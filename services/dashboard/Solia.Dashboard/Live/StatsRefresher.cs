namespace Solia.Dashboard.Live;

/// <summary>
/// The live wiring of the pages built from stats cards: every <c>trade_closed</c> or <c>stats_changed</c> asks for one
/// debounced refresh, a resync refreshes at once (and drops the pending one), and a periodic reconcile catches what
/// sends nothing on the hub (a lost event, an ageing window, a promotion). Created by the page once it is interactive
/// and disposed with it; <c>refresh</c> runs through <c>dispatch</c>, the page's <c>InvokeAsync</c>.
/// </summary>
public sealed class StatsRefresher : IDisposable
{
    private readonly Func<Func<Task>, Task> _dispatch;
    private readonly Func<Task> _refresh;
    private readonly Debouncer _debouncer;
    private readonly List<IDisposable> _subscriptions = [];
    private readonly ITimer _reconcileTimer;
    private volatile bool _disposed;

    public StatsRefresher(
        ILiveFeed feed,
        TimeProvider clock,
        TimeSpan refreshDelay,
        TimeSpan reconcileInterval,
        Func<Func<Task>, Task> dispatch,
        Func<Task> refresh,
        ILogger logger)
    {
        _dispatch = dispatch;
        _refresh = refresh;
        _debouncer = new Debouncer(clock, refreshDelay, () => _dispatch(RefreshAsync), logger);
        // Every close is followed by a stats_changed (and either may be lost): both just ask for one refresh soon.
        _subscriptions.Add(feed.Subscribe(LiveFeedEvents.StatsChanged, _ => Trigger()));
        _subscriptions.Add(feed.Subscribe(LiveFeedEvents.TradeClosed, _ => Trigger()));
        _subscriptions.Add(feed.SubscribeResynced(() => _dispatch(OnResyncedAsync)));
        _reconcileTimer = clock.CreateTimer(_ => _debouncer.Trigger(), null, reconcileInterval, reconcileInterval);
    }

    private Task Trigger()
    {
        _debouncer.Trigger();
        return Task.CompletedTask;
    }

    // Events sent while the feed was down are lost: re-fetch now, which also covers a refresh still pending.
    private Task OnResyncedAsync()
    {
        _debouncer.Cancel();
        return RefreshAsync();
    }

    private Task RefreshAsync() => _disposed ? Task.CompletedTask : _refresh();

    public void Dispose()
    {
        _disposed = true;
        _subscriptions.ForEach(s => s.Dispose());
        _reconcileTimer.Dispose();
        _debouncer.Dispose();
    }
}
