using System.Diagnostics;
using Microsoft.AspNetCore.SignalR.Client;

namespace Solia.Dashboard.Live;

/// <summary>
/// <see cref="ILiveFeed"/> over one SignalR connection to the backend hub. <b>One connection per app, not per
/// circuit</b>: the hub broadcasts the same events to every client and has no per-user state, so a connection per
/// browser tab would only multiply sockets and reconnect storms against the backend. Every circuit's components
/// subscribe to this singleton and dispose their subscriptions with themselves.
/// <para>
/// Runs as a hosted service: starting never blocks the host. A failed start is retried, and a dropped connection
/// is re-established by SignalR, both with the retry policy (<see cref="LiveFeedRetryPolicy"/>: never giving up). A
/// connection the server closed is restarted with the same backoff, which only resets once a connection has stayed
/// up for <see cref="StableConnection"/> — a server that accepts and then drops every connection is not hammered.
/// Every transition to <see cref="LiveFeedState.Connected"/> raises the resynced signal.
/// </para>
/// <para>
/// Subscribers are started in order but never awaited: SignalR handles one message at a time, so a slow circuit
/// must not stall delivery (or pings) for the whole app.
/// </para>
/// </summary>
internal sealed class LiveFeed : ILiveFeed, IHostedService, IAsyncDisposable
{
    /// <summary>How long a connection must stay up before a later close restarts the backoff from its first step.</summary>
    internal static readonly TimeSpan StableConnection = TimeSpan.FromSeconds(30);

    private readonly ILiveConnection _connection;
    private readonly IRetryPolicy _retryPolicy;
    private readonly TimeSpan _stableAfter;
    private readonly ILogger<LiveFeed> _logger;
    private readonly Dictionary<string, object> _events = [];
    private readonly Subscribers<LiveFeedState> _stateSubscribers;
    private readonly Subscribers<ValueTuple> _resyncSubscribers;
    private readonly CancellationTokenSource _stopping = new();
    private readonly Lock _stateGate = new();
    private readonly Lock _lifecycleGate = new();

    private volatile LiveFeedState _state = LiveFeedState.Disconnected;
    private TaskCompletionSource<Exception?> _closed = NewSignal();
    private Task? _run;
    private Task? _stop;

    public LiveFeed(ILiveConnection connection, IRetryPolicy retryPolicy, ILogger<LiveFeed> logger, TimeSpan? stableAfter = null)
    {
        _connection = connection;
        _retryPolicy = retryPolicy;
        _stableAfter = stableAfter ?? StableConnection;
        _logger = logger;
        _stateSubscribers = new Subscribers<LiveFeedState>("state", logger);
        _resyncSubscribers = new Subscribers<ValueTuple>("resynced", logger);

        Register(LiveFeedEvents.PredictionCreated);
        Register(LiveFeedEvents.TradeOpened);
        Register(LiveFeedEvents.TradeClosed);
        Register(LiveFeedEvents.LogWritten);
        Register(LiveFeedEvents.StatsChanged);

        _connection.Reconnecting += OnReconnectingAsync;
        _connection.Reconnected += OnReconnectedAsync;
        _connection.Closed += OnClosedAsync;
    }

    public LiveFeedState State => _state;

    public IDisposable Subscribe<T>(LiveEvent<T> liveEvent, Func<T, Task> handler)
    {
        ArgumentNullException.ThrowIfNull(liveEvent);
        return Events(liveEvent).Add(handler);
    }

    public IDisposable SubscribeState(Func<LiveFeedState, Task> handler) => _stateSubscribers.Add(handler);

    public IDisposable SubscribeResynced(Func<Task> handler)
    {
        ArgumentNullException.ThrowIfNull(handler);
        return _resyncSubscribers.Add(_ => handler());
    }

    /// <summary>Subscribers of <paramref name="liveEvent"/> (tests check nothing is left behind).</summary>
    internal Subscribers<T> Events<T>(LiveEvent<T> liveEvent) => (Subscribers<T>)_events[liveEvent.Name];

    public Task StartAsync(CancellationToken cancellationToken)
    {
        lock (_lifecycleGate)
        {
            _run ??= Task.Run(() => RunAsync(_stopping.Token), CancellationToken.None);
        }

        return Task.CompletedTask;
    }

    /// <summary>
    /// Stops the feed. The stop itself always runs to the end — <paramref name="cancellationToken"/> only bounds how
    /// long this call waits — and every call (<see cref="DisposeAsync"/> included) waits for that same stop.
    /// </summary>
    public Task StopAsync(CancellationToken cancellationToken)
    {
        Task stop;
        lock (_lifecycleGate)
        {
            if (_run is null)
            {
                return Task.CompletedTask;
            }

            Task run = _run;
            stop = _stop ??= Task.Run(() => StopCoreAsync(run), CancellationToken.None);
        }

        return stop.WaitAsync(cancellationToken);
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync(CancellationToken.None);
        await _connection.DisposeAsync();
        _stopping.Dispose();
    }

    private async Task StopCoreAsync(Task run)
    {
        await _stopping.CancelAsync();
        try
        {
            await run;
            await _connection.StopAsync(CancellationToken.None);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "live: error while stopping the hub connection");
        }

        SetState(LiveFeedState.Disconnected);
    }

    private void Register<T>(LiveEvent<T> liveEvent)
    {
        Subscribers<T> subscribers = new(liveEvent.Name, _logger);
        _events[liveEvent.Name] = subscribers;
        _connection.On<T>(liveEvent.Name, payload =>
        {
            if (payload is not null)
            {
                subscribers.Publish(payload);
            }

            return Task.CompletedTask;
        });
    }

    private async Task RunAsync(CancellationToken stopping)
    {
        long failures = 0;
        while (!stopping.IsCancellationRequested)
        {
            Volatile.Write(ref _closed, NewSignal());
            Exception reason;
            try
            {
                await _connection.StartAsync(stopping);
                _logger.LogInformation("live: connected to the backend hub");
                long connectedAt = Stopwatch.GetTimestamp();
                OnConnected();

                Exception? error = await Volatile.Read(ref _closed).Task.WaitAsync(stopping);
                if (Stopwatch.GetElapsedTime(connectedAt) >= _stableAfter)
                {
                    failures = 0;
                }

                reason = error ?? new IOException("The backend closed the hub connection.");
            }
            catch (OperationCanceledException) when (stopping.IsCancellationRequested)
            {
                return;
            }
            catch (Exception ex)
            {
                reason = ex;
                SetState(LiveFeedState.Disconnected);
            }

            TimeSpan? delay = _retryPolicy.NextRetryDelay(new RetryContext { PreviousRetryCount = failures++, RetryReason = reason });
            if (delay is null)
            {
                _logger.LogError(reason, "live: hub connection lost and the retry policy gave up; live updates stay offline");
                return;
            }

            _logger.LogWarning(reason, "live: no connection to the backend hub, retrying in {Delay}", delay);
            try
            {
                await Task.Delay(delay.Value, stopping);
            }
            catch (OperationCanceledException)
            {
                return;
            }
        }
    }

    private Task OnReconnectingAsync(Exception? error)
    {
        _logger.LogWarning(error, "live: hub connection lost, reconnecting");
        SetState(LiveFeedState.Reconnecting);
        return Task.CompletedTask;
    }

    private Task OnReconnectedAsync(string? connectionId)
    {
        _logger.LogInformation("live: reconnected to the backend hub");
        OnConnected();
        return Task.CompletedTask;
    }

    // Closed for good (by the server, or a reconnect that gave up): the run loop backs off and starts over.
    private Task OnClosedAsync(Exception? error)
    {
        SetState(LiveFeedState.Disconnected);
        Volatile.Read(ref _closed).TrySetResult(error);
        return Task.CompletedTask;
    }

    private void OnConnected()
    {
        SetState(LiveFeedState.Connected);
        _resyncSubscribers.Publish(default);
    }

    private void SetState(LiveFeedState state)
    {
        lock (_stateGate)
        {
            if (_state == state)
            {
                return;
            }

            _state = state;

            // Started under the lock so every subscriber sees the transitions in the order they happened.
            _stateSubscribers.Publish(state);
        }
    }

    private static TaskCompletionSource<Exception?> NewSignal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
}
