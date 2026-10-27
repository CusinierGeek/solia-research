using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

/// <summary>An <see cref="ILiveFeed"/> for component tests: the test sets its state.</summary>
internal sealed class FakeLiveFeed(LiveFeedState state = LiveFeedState.Connected) : ILiveFeed
{
    private readonly Subscribers<LiveFeedState> _state = new("state", NullLogger.Instance);

    public LiveFeedState State { get; private set; } = state;

    public int StateSubscribers => _state.Count;

    public Task SetStateAsync(LiveFeedState state)
    {
        State = state;
        return _state.InvokeAsync(state);
    }

    /// <summary>Notifies subscribers of <paramref name="stale"/> without changing <see cref="State"/> — a late notification.</summary>
    public Task NotifyAsync(LiveFeedState stale) => _state.InvokeAsync(stale);

    public IDisposable Subscribe<T>(LiveEvent<T> liveEvent, Func<T, Task> handler) => new Nothing();

    public IDisposable SubscribeState(Func<LiveFeedState, Task> handler) => _state.Add(handler);

    public IDisposable SubscribeResynced(Func<Task> handler) => new Nothing();

    private sealed class Nothing : IDisposable
    {
        public void Dispose()
        {
        }
    }
}
