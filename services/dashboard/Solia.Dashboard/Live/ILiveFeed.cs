namespace Solia.Dashboard.Live;

/// <summary>Whether the dashboard is receiving the backend's live events right now.</summary>
public enum LiveFeedState
{
    /// <summary>Not connected (backend unreachable, or not started yet); the feed keeps retrying with backoff.</summary>
    Disconnected,

    /// <summary>The connection dropped and is being re-established automatically.</summary>
    Reconnecting,

    /// <summary>Connected: events flow.</summary>
    Connected,
}

/// <summary>
/// The dashboard's one shared subscription to the backend's live hub (<see cref="LiveFeedEvents"/>). Every
/// <c>Subscribe…</c> returns a handle that stops delivery once disposed — components dispose theirs in
/// <c>Dispose</c>. Handlers run on a background thread, not the component's: marshal UI work with
/// <c>InvokeAsync(StateHasChanged)</c>. Handlers are started in order but not awaited, so a slow one never delays
/// the others or the connection. Delivery is at most once, so views re-fetch over REST on
/// <see cref="SubscribeResynced"/>.
/// </summary>
public interface ILiveFeed
{
    /// <summary>The current connection state.</summary>
    LiveFeedState State { get; }

    /// <summary>Calls <paramref name="handler"/> with every <paramref name="liveEvent"/> payload received.</summary>
    IDisposable Subscribe<T>(LiveEvent<T> liveEvent, Func<T, Task> handler);

    /// <summary>
    /// Calls <paramref name="handler"/> on every <see cref="State"/> change, in order. A handler that renders later
    /// (e.g. through <c>InvokeAsync</c>) should read <see cref="State"/> then rather than keep the value it was given.
    /// </summary>
    IDisposable SubscribeState(Func<LiveFeedState, Task> handler);

    /// <summary>
    /// Calls <paramref name="handler"/> each time the feed becomes <see cref="LiveFeedState.Connected"/> (the first
    /// connect included): events sent while it was down are lost, so the view should re-fetch its data over REST.
    /// </summary>
    IDisposable SubscribeResynced(Func<Task> handler);
}
