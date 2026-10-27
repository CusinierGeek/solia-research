using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Api;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

public class StatsRefresherTests
{
    private static readonly TimeSpan Delay = TimeSpan.FromSeconds(1);
    private static readonly TimeSpan Reconcile = TimeSpan.FromSeconds(60);
    private static readonly DateTimeOffset Now = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private readonly ManualClock _clock = new(Now);
    private readonly FakeLiveFeed _feed = new();
    private int _refreshes;
    private int _dispatched;

    private StatsRefresher Create() => new(_feed, _clock, Delay, Reconcile,
        work => { _dispatched++; return work(); },
        () => { _refreshes++; return Task.CompletedTask; },
        NullLogger.Instance);

    [Fact]
    public async Task A_burst_of_closes_refreshes_once_after_the_delay_through_the_dispatcher()
    {
        using StatsRefresher refresher = Create();

        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Trade(1));
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(2, Now));

        Assert.Equal(0, _refreshes);
        _clock.Elapse(Delay);
        Assert.Equal(1, _refreshes);
        Assert.Equal(1, _dispatched);
    }

    [Fact]
    public async Task A_resync_refreshes_at_once_and_drops_the_pending_refresh()
    {
        using StatsRefresher refresher = Create();
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));

        await _feed.ResyncAsync();

        Assert.Equal(1, _refreshes);
        Assert.False(_clock.IsArmed(Delay));
        _clock.Elapse(Delay);
        Assert.Equal(1, _refreshes);
    }

    [Fact]
    public void With_no_event_it_still_refreshes_every_reconcile_interval()
    {
        using StatsRefresher refresher = Create();

        _clock.Tick(Reconcile);
        _clock.Elapse(Delay);

        Assert.Equal(1, _refreshes);
    }

    [Fact]
    public async Task Disposing_unsubscribes_and_stops_every_refresh()
    {
        StatsRefresher refresher = Create();
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));

        refresher.Dispose();
        _clock.Tick(Reconcile);
        _clock.Elapse(Delay);

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.StatsChanged));
        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.TradeClosed));
        Assert.Equal(0, _feed.ResyncSubscribers);
        Assert.Equal(0, _refreshes);
    }

    [Fact]
    public async Task A_refresh_already_dispatched_when_it_is_disposed_does_nothing()
    {
        Func<Task>? queued = null;
        var refresher = new StatsRefresher(_feed, _clock, Delay, Reconcile,
            work => { queued = work; return Task.CompletedTask; },
            () => { _refreshes++; return Task.CompletedTask; },
            NullLogger.Instance);
        await _feed.ResyncAsync();

        refresher.Dispose();
        await queued!();

        Assert.Equal(0, _refreshes);
    }

    private static TradeDto Trade(long id) =>
        new(id, "mint", null, Now.AddMinutes(-5), Now, 1m, 1.1m, 1m, 300, 0.1m, 0.1, "buy", 70, "tp", "closed");
}
