using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Api;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

public class LiveFeedTests : IAsyncLifetime
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(5);

    private static readonly TradeDto Trade = new(
        7, "So1abc", 42, DateTimeOffset.Parse("2026-10-26T12:00:00Z"), null, 0.0000001234m, null, 0.5m, null, null, null,
        "buy", 74.0, null, "open");

    private readonly FakeLiveConnection _connection = new();
    private readonly LiveFeed _feed;
    private readonly List<LiveFeedState> _states = [];
    private int _resyncs;

    public LiveFeedTests()
    {
        _feed = new LiveFeed(_connection, new ZeroDelayPolicy(), NullLogger<LiveFeed>.Instance);
        _feed.SubscribeState(s =>
        {
            lock (_states)
            {
                _states.Add(s);
            }

            return Task.CompletedTask;
        });
        _feed.SubscribeResynced(() =>
        {
            Interlocked.Increment(ref _resyncs);
            return Task.CompletedTask;
        });
    }

    Task IAsyncLifetime.InitializeAsync() => Task.CompletedTask;

    async Task IAsyncLifetime.DisposeAsync() => await _feed.DisposeAsync();

    // --- Dispatch ---

    [Fact]
    public void Every_hub_event_is_registered_on_the_connection()
    {
        foreach (string name in new[] { "prediction_created", "trade_opened", "trade_closed", "log_written", "stats_changed" })
        {
            Assert.True(_connection.Handles(name), name);
        }
    }

    [Fact]
    public async Task An_event_reaches_its_typed_subscribers_only()
    {
        List<TradeDto> opened = [];
        int closed = 0;
        _feed.Subscribe(LiveFeedEvents.TradeOpened, t => Add(opened, t));
        _feed.Subscribe(LiveFeedEvents.TradeClosed, _ => Task.FromResult(Interlocked.Increment(ref closed)));

        await _connection.RaiseAsync("trade_opened", Trade);

        Assert.Equal([Trade], opened);
        Assert.Equal(0, closed);
    }

    [Fact]
    public async Task Every_subscriber_of_an_event_receives_it()
    {
        List<StatsChangedDto> first = [], second = [];
        StatsChangedDto stats = new(7, DateTimeOffset.Parse("2026-10-26T12:01:30Z"));
        _feed.Subscribe(LiveFeedEvents.StatsChanged, s => Add(first, s));
        _feed.Subscribe(LiveFeedEvents.StatsChanged, s => Add(second, s));

        await _connection.RaiseAsync("stats_changed", stats);

        Assert.Equal([stats], first);
        Assert.Equal([stats], second);
    }

    [Fact]
    public async Task A_disposed_subscriber_no_longer_receives_events()
    {
        List<TradeDto> received = [];
        IDisposable subscription = _feed.Subscribe(LiveFeedEvents.TradeOpened, t => Add(received, t));
        await _connection.RaiseAsync("trade_opened", Trade);

        subscription.Dispose();
        await _connection.RaiseAsync("trade_opened", Trade with { Id = 8 });

        Assert.Equal([Trade], received);
        Assert.Equal(0, _feed.Events(LiveFeedEvents.TradeOpened).Count);
    }

    [Fact]
    public async Task A_subscriber_disposed_during_a_dispatch_is_not_called_by_it()
    {
        int laterCalls = 0;
        IDisposable? later = null;
        _feed.Subscribe(LiveFeedEvents.TradeOpened, _ =>
        {
            later!.Dispose();
            return Task.CompletedTask;
        });
        later = _feed.Subscribe(LiveFeedEvents.TradeOpened, _ => Task.FromResult(Interlocked.Increment(ref laterCalls)));

        await _connection.RaiseAsync("trade_opened", Trade);

        Assert.Equal(0, laterCalls);
    }

    [Fact]
    public async Task Disposing_a_subscription_twice_is_harmless()
    {
        IDisposable subscription = _feed.Subscribe(LiveFeedEvents.LogWritten, _ => Task.CompletedTask);
        IDisposable other = _feed.Subscribe(LiveFeedEvents.LogWritten, _ => Task.CompletedTask);

        subscription.Dispose();
        subscription.Dispose();

        Assert.Equal(1, _feed.Events(LiveFeedEvents.LogWritten).Count);
        other.Dispose();
        await _connection.RaiseAsync("log_written", new LogDto(1, DateTimeOffset.UtcNow, "backend", "info", "m", null));
    }

    [Fact]
    public async Task A_failing_subscriber_does_not_stop_the_others_or_the_feed()
    {
        List<TradeDto> received = [];
        _feed.Subscribe(LiveFeedEvents.TradeOpened, _ => throw new InvalidOperationException("component bug"));
        _feed.Subscribe(LiveFeedEvents.TradeOpened, _ => Task.FromException(new InvalidOperationException("async bug")));
        _feed.Subscribe(LiveFeedEvents.TradeOpened, t => Add(received, t));

        await _connection.RaiseAsync("trade_opened", Trade);

        Assert.Equal([Trade], received);
    }

    [Fact]
    public async Task A_null_payload_is_ignored()
    {
        int calls = 0;
        _feed.Subscribe(LiveFeedEvents.PredictionCreated, _ => Task.FromResult(Interlocked.Increment(ref calls)));

        await _connection.RaiseAsync("prediction_created", null);

        Assert.Equal(0, calls);
    }

    [Fact]
    public void Subscribing_a_null_handler_is_rejected()
    {
        Assert.Throws<ArgumentNullException>(() => _feed.Subscribe(LiveFeedEvents.TradeOpened, null!));
        Assert.Throws<ArgumentNullException>(() => _feed.SubscribeResynced(null!));
    }

    // --- Connection state ---

    [Fact]
    public void Starts_disconnected()
    {
        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
    }

    [Fact]
    public async Task Connecting_moves_to_connected_and_raises_resynced()
    {
        await _feed.StartAsync(CancellationToken.None);

        await Eventually(() => _feed.State == LiveFeedState.Connected && _resyncs == 1);
        Assert.Equal([LiveFeedState.Connected], States());
    }

    [Fact]
    public async Task Starting_does_not_wait_for_the_connection()
    {
        TaskCompletionSource never = new();
        _connection.EnqueueStart(ct => never.Task.WaitAsync(ct));

        Task start = _feed.StartAsync(CancellationToken.None);

        Assert.True(start.IsCompleted);
        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
    }

    [Fact]
    public async Task A_failed_start_is_retried_until_it_connects()
    {
        _connection.FailNextStart(times: 3);

        await _feed.StartAsync(CancellationToken.None);

        await Eventually(() => _feed.State == LiveFeedState.Connected);
        Assert.Equal(4, _connection.StartCalls);
        Assert.Equal(1, _resyncs);
    }

    [Fact]
    public async Task A_dropped_connection_reports_reconnecting_then_connected_and_resyncs()
    {
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _feed.State == LiveFeedState.Connected);

        await _connection.RaiseReconnectingAsync();
        Assert.Equal(LiveFeedState.Reconnecting, _feed.State);

        await _connection.RaiseReconnectedAsync();
        Assert.Equal(LiveFeedState.Connected, _feed.State);
        Assert.Equal(2, _resyncs);
        Assert.Equal([LiveFeedState.Connected, LiveFeedState.Reconnecting, LiveFeedState.Connected], States());
    }

    [Fact]
    public async Task A_closed_connection_goes_offline_and_is_restarted()
    {
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _feed.State == LiveFeedState.Connected);
        _connection.FailNextStart();

        await _connection.RaiseClosedAsync();

        await Eventually(() => _connection.StartCalls == 3 && _feed.State == LiveFeedState.Connected);
        Assert.Equal([LiveFeedState.Connected, LiveFeedState.Disconnected, LiveFeedState.Connected], States());
        Assert.Equal(2, _resyncs);
    }

    [Fact]
    public async Task Stopping_disconnects_and_does_not_restart()
    {
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _feed.State == LiveFeedState.Connected);

        await _feed.StopAsync(CancellationToken.None);
        await _connection.RaiseClosedAsync();

        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
        Assert.Equal(1, _connection.StopCalls);
        Assert.Equal(1, _connection.StartCalls);
    }

    [Fact]
    public async Task Stopping_cancels_a_start_in_progress()
    {
        TaskCompletionSource never = new();
        _connection.EnqueueStart(ct => never.Task.WaitAsync(ct));
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _connection.StartCalls == 1);

        await _feed.StopAsync(CancellationToken.None).WaitAsync(Timeout);

        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
        Assert.Equal(1, _connection.StartCalls);
    }

    [Fact]
    public async Task Disposing_disposes_the_connection()
    {
        await _feed.StartAsync(CancellationToken.None);

        await _feed.DisposeAsync();

        Assert.True(_connection.Disposed);
    }

    [Fact]
    public async Task A_slow_subscriber_does_not_hold_up_the_connection_or_the_others()
    {
        TaskCompletionSource never = new();
        List<TradeDto> received = [];
        _feed.Subscribe(LiveFeedEvents.TradeOpened, _ => never.Task);
        _feed.Subscribe(LiveFeedEvents.TradeOpened, t => Add(received, t));

        await _connection.RaiseAsync("trade_opened", Trade).WaitAsync(Timeout);
        await _connection.RaiseAsync("trade_opened", Trade with { Id = 8 }).WaitAsync(Timeout);

        Assert.Equal([7L, 8L], received.Select(t => t.Id));
    }

    [Fact]
    public async Task A_slow_resync_handler_does_not_hold_up_restarts_or_stopping()
    {
        TaskCompletionSource never = new();
        _feed.SubscribeResynced(() => never.Task);
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _feed.State == LiveFeedState.Connected);

        await _connection.RaiseClosedAsync();
        await Eventually(() => _connection.StartCalls == 2 && _feed.State == LiveFeedState.Connected);

        await _feed.StopAsync(CancellationToken.None).WaitAsync(Timeout);
        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
    }

    [Fact]
    public async Task Repeated_server_closes_back_off_instead_of_restarting_at_once()
    {
        RecordingPolicy policy = new(TimeSpan.Zero);
        await using LiveFeed feed = new(_connection, policy, NullLogger<LiveFeed>.Instance, stableAfter: TimeSpan.MaxValue);
        await feed.StartAsync(CancellationToken.None);

        for (int closes = 1; closes <= 3; closes++)
        {
            int expected = closes;
            await Eventually(() => _connection.StartCalls == expected && feed.State == LiveFeedState.Connected);
            await _connection.RaiseClosedAsync();
        }

        await Eventually(() => policy.Retries.Count == 3);
        Assert.Equal([0L, 1L, 2L], policy.Retries);
    }

    [Fact]
    public async Task The_backoff_restarts_after_a_connection_that_stayed_up()
    {
        RecordingPolicy policy = new(TimeSpan.Zero);
        await using LiveFeed feed = new(_connection, policy, NullLogger<LiveFeed>.Instance, stableAfter: TimeSpan.Zero);
        _connection.FailNextStart(times: 2);
        await feed.StartAsync(CancellationToken.None);
        await Eventually(() => feed.State == LiveFeedState.Connected);

        await _connection.RaiseClosedAsync();

        await Eventually(() => policy.Retries.Count == 3);
        Assert.Equal([0L, 1L, 0L], policy.Retries);
    }

    [Fact]
    public async Task A_policy_that_gives_up_leaves_the_feed_offline()
    {
        RecordingPolicy policy = new(null);
        await using LiveFeed feed = new(_connection, policy, NullLogger<LiveFeed>.Instance);
        _connection.FailNextStart(times: 5);

        await feed.StartAsync(CancellationToken.None);
        await Eventually(() => policy.Retries.Count == 1);
        await Task.Delay(100);

        Assert.Equal(1, _connection.StartCalls);
        Assert.Equal(LiveFeedState.Disconnected, feed.State);
    }

    [Fact]
    public async Task Concurrent_starts_run_a_single_connection_loop()
    {
        await Task.WhenAll(Enumerable.Range(0, 16).Select(_ => Task.Run(() => _feed.StartAsync(CancellationToken.None))));

        await Eventually(() => _feed.State == LiveFeedState.Connected);
        await Task.Delay(50);
        Assert.Equal(1, _connection.StartCalls);
    }

    [Fact]
    public async Task A_stop_whose_wait_is_cancelled_still_completes_and_later_stops_wait_for_it()
    {
        TaskCompletionSource stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);
        _connection.OnStop = _ => stopped.Task;
        await _feed.StartAsync(CancellationToken.None);
        await Eventually(() => _feed.State == LiveFeedState.Connected);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _feed.StopAsync(new CancellationToken(canceled: true)));
        Task second = _feed.StopAsync(CancellationToken.None);
        await Eventually(() => _connection.StopCalls == 1);
        Assert.False(second.IsCompleted);

        stopped.SetResult();
        await second.WaitAsync(Timeout);

        Assert.Equal(LiveFeedState.Disconnected, _feed.State);
        Assert.Equal(1, _connection.StopCalls);
    }

    [Fact]
    public async Task Stopping_a_feed_that_never_started_is_a_no_op()
    {
        await _feed.StopAsync(CancellationToken.None);

        Assert.Equal(0, _connection.StopCalls);
    }

    private List<LiveFeedState> States()
    {
        lock (_states)
        {
            return [.. _states];
        }
    }

    private static Task Add<T>(List<T> list, T item)
    {
        lock (list)
        {
            list.Add(item);
        }

        return Task.CompletedTask;
    }

    private static async Task Eventually(Func<bool> condition)
    {
        using CancellationTokenSource timeout = new(Timeout);
        while (!condition())
        {
            await Task.Delay(10, timeout.Token);
        }
    }

    /// <summary>Records each retry's <see cref="RetryContext.PreviousRetryCount"/> and answers the given delay.</summary>
    private sealed class RecordingPolicy(TimeSpan? delay) : IRetryPolicy
    {
        private readonly List<long> _retries = [];

        public List<long> Retries
        {
            get
            {
                lock (_retries)
                {
                    return [.. _retries];
                }
            }
        }

        public TimeSpan? NextRetryDelay(RetryContext retryContext)
        {
            lock (_retries)
            {
                _retries.Add(retryContext.PreviousRetryCount);
            }

            return delay;
        }
    }

    private sealed class ZeroDelayPolicy : IRetryPolicy
    {
        public TimeSpan? NextRetryDelay(RetryContext retryContext) => TimeSpan.Zero;
    }
}
