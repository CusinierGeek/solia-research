using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

/// <summary>An <see cref="ILiveFeed"/> for component tests: the test sets its state, pushes events and resyncs.</summary>
internal sealed class FakeLiveFeed(LiveFeedState state = LiveFeedState.Connected) : ILiveFeed
{
    private readonly Subscribers<LiveFeedState> _state = new("state", NullLogger.Instance);
    private readonly Subscribers<bool> _resynced = new("resynced", NullLogger.Instance);
    private readonly Dictionary<string, object> _events = [];

    public LiveFeedState State { get; private set; } = state;

    public int StateSubscribers => _state.Count;

    public int ResyncSubscribers => _resynced.Count;

    public int Subscribers<T>(LiveEvent<T> liveEvent) => Events(liveEvent).Count;

    public Task SetStateAsync(LiveFeedState state)
    {
        State = state;
        return _state.InvokeAsync(state);
    }

    /// <summary>Notifies subscribers of <paramref name="stale"/> without changing <see cref="State"/> — a late notification.</summary>
    public Task NotifyAsync(LiveFeedState stale) => _state.InvokeAsync(stale);

    /// <summary>Delivers <paramref name="payload"/> to the <paramref name="liveEvent"/> subscribers, as the hub would.</summary>
    public Task PushAsync<T>(LiveEvent<T> liveEvent, T payload) => Events(liveEvent).InvokeAsync(payload);

    /// <summary>Signals a (re)connect: views re-fetch.</summary>
    public Task ResyncAsync() => _resynced.InvokeAsync(true);

    public IDisposable Subscribe<T>(LiveEvent<T> liveEvent, Func<T, Task> handler) => Events(liveEvent).Add(handler);

    public IDisposable SubscribeState(Func<LiveFeedState, Task> handler) => _state.Add(handler);

    public IDisposable SubscribeResynced(Func<Task> handler) => _resynced.Add(_ => handler());

    private Subscribers<T> Events<T>(LiveEvent<T> liveEvent)
    {
        if (!_events.TryGetValue(liveEvent.Name, out object? subscribers))
        {
            subscribers = new Subscribers<T>(liveEvent.Name, NullLogger.Instance);
            _events[liveEvent.Name] = subscribers;
        }
        return (Subscribers<T>)subscribers;
    }
}
