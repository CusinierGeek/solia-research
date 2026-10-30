using AngleSharp.Dom;
using Bunit.TestDoubles;
using Microsoft.Extensions.DependencyInjection;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Charts;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Stats;
using Solia.Dashboard.Tests.Api;
using Solia.Dashboard.Tests.Live;

namespace Solia.Dashboard.Tests.Pages;

/// <summary>The chart wrappers are stubbed: these tests check what the page hands them, not what the library draws.</summary>
public class PerformanceTests : MudTestContext
{
    private static readonly DateTimeOffset Now = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();
    private readonly FakeLiveFeed _feed = new();
    private readonly ManualClock _clock = new(Now);

    public PerformanceTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Services.AddSingleton<ILiveFeed>(_feed);
        Services.AddSingleton<TimeProvider>(_clock);
        ComponentFactories.AddStub<EquityChart>();
        ComponentFactories.AddStub<WinRateDonut>();
        _api.Performance = (_, _) => Task.FromResult(Curve(3.2m, (Now.AddHours(-2), 101.5m, 1.5m), (Now.AddHours(-1), 103.2m, 1.7m)));
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0.58, 290, 190, 20, "all"));
    }

    private static PerformanceResponse Curve(decimal totalPnl, params (DateTimeOffset Ts, decimal Equity, decimal Pnl)[] points) =>
        new(points.Select(p => new EquityPoint(p.Ts, p.Equity, p.Pnl)).ToList(), totalPnl);

    private IRenderedComponent<Performance> RenderLoaded()
    {
        IRenderedComponent<Performance> page = Render<Performance>();
        page.WaitForAssertion(() =>
        {
            Assert.Single(page.FindComponents<Stub<EquityChart>>());
            Assert.Single(page.FindComponents<Stub<WinRateDonut>>());
        });
        return page;
    }

    private static EquityCurve ChartCurve(IRenderedComponent<Performance> page) =>
        page.FindComponent<Stub<EquityChart>>().Instance.Parameters.Get(c => c.Curve);

    private static WinRateSlices DonutSlices(IRenderedComponent<Performance> page) =>
        page.FindComponent<Stub<WinRateDonut>>().Instance.Parameters.Get(c => c.Slices);

    private static IElement Card(IRenderedComponent<Performance> page, string name) => page.Find($"[data-card={name}]");

    private static string Text(IElement card, string selector) => card.QuerySelector(selector)!.TextContent.Trim();

    private int Calls => _api.PerformanceCalls.Count + _api.WinRateCalls.Count;

    [Fact]
    public void The_equity_chart_receives_the_mapped_curve()
    {
        _api.Performance = (_, _) => Task.FromResult(Curve(2.5m,
            (Now.AddHours(-1), 102.5m, 1m),
            (Now.AddHours(-3), 101.5m, 1.5m)));

        IRenderedComponent<Performance> page = RenderLoaded();

        EquityCurve curve = ChartCurve(page);
        Assert.Equal([101.5m, 102.5m], curve.Points.Select(p => p.Equity)); // ordered by ts
        Assert.Equal(100m, curve.StartingBalance);
        Assert.Equal(2.5m, curve.TotalPnl);
        Assert.Equal([1001], _api.PerformanceCalls); // the default length, one more to tell older trades exist
    }

    [Fact]
    public void The_summary_shows_the_starting_balance_the_equity_and_the_total_pnl()
    {
        IRenderedComponent<Performance> page = RenderLoaded();

        IElement card = Card(page, "equity");
        Assert.Equal("Starting balance 100", Text(card, "[data-part=start]"));
        Assert.Equal("Equity 103.2", Text(card, "[data-part=equity]"));
        Assert.Equal("Total P&L +3.2", Text(card, "[data-part=total]"));
        Assert.Contains("solia-pnl--gain", card.QuerySelector("[data-part=total] strong")!.ClassList);
        Assert.Equal("All 2 closed trades", Text(card, "[data-part=shown]")); // fewer than the 1000 asked for
    }

    // The backend's answer when `closed` trades exist in all: the latest `limit` of them.
    private static PerformanceResponse Trades(int closed, int? limit)
    {
        int count = Math.Min(closed, limit!.Value);
        return Curve(closed * 0.01m,
            Enumerable.Range(closed - count, count).Select(i => (Now.AddMinutes(i - closed), 100m + ((i + 1) * 0.01m), 0.01m)).ToArray());
    }

    [Fact]
    public void More_trades_than_shown_says_it_shows_the_latest_trades_only()
    {
        _api.Performance = (limit, _) => Task.FromResult(Trades(150, limit));
        IRenderedComponent<Performance> page = RenderLoaded();

        page.Find("[data-limit='100']").Click();

        page.WaitForAssertion(() => Assert.Equal("Latest 100 closed trades", Text(Card(page, "equity"), "[data-part=shown]")));
        Assert.Equal(100, ChartCurve(page).Points.Count);
        Assert.Equal(101.5m, ChartCurve(page).Points[^1].Equity);
    }

    [Fact]
    public void Exactly_as_many_trades_as_shown_says_all_of_them_are_on_the_curve()
    {
        _api.Performance = (limit, _) => Task.FromResult(Trades(100, limit));
        IRenderedComponent<Performance> page = RenderLoaded();

        page.Find("[data-limit='100']").Click();

        page.WaitForAssertion(() => Assert.Equal([1001, 101], _api.PerformanceCalls));
        page.WaitForAssertion(() => Assert.Equal("All 100 closed trades", Text(Card(page, "equity"), "[data-part=shown]")));
    }

    [Fact]
    public void While_a_longer_curve_loads_the_one_on_screen_keeps_its_own_label()
    {
        _api.Performance = (limit, _) => Task.FromResult(Trades(2000, limit));
        IRenderedComponent<Performance> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal("Latest 1,000 closed trades", Text(Card(page, "equity"), "[data-part=shown]")));
        var longer = new TaskCompletionSource<PerformanceResponse>();
        _api.Performance = (_, _) => longer.Task;

        page.Find("[data-limit='5000']").Click();

        page.WaitForAssertion(() => Assert.Equal([1001, 5001], _api.PerformanceCalls));
        Assert.Equal("Latest 1,000 closed trades", Text(Card(page, "equity"), "[data-part=shown]"));

        longer.SetResult(Trades(2000, 5001));
        page.WaitForAssertion(() => Assert.Equal("All 2,000 closed trades", Text(Card(page, "equity"), "[data-part=shown]")));
    }

    [Fact]
    public void The_donut_receives_the_mapped_breakdown_over_every_closed_trade()
    {
        IRenderedComponent<Performance> page = RenderLoaded();

        WinRateSlices slices = DonutSlices(page);
        Assert.Equal([290.0, 190.0, 20.0], slices.Values);
        Assert.Equal("58.0%", Text(Card(page, "winrate"), ".solia-stat-card__value"));
        Assert.Contains("over 500 closed trades", Card(page, "winrate").TextContent);
        Assert.Equal([StatsWindow.All], _api.WinRateCalls);
    }

    [Fact]
    public void Both_panels_show_a_loading_state_until_the_backend_answers()
    {
        var curve = new TaskCompletionSource<PerformanceResponse>();
        var winRate = new TaskCompletionSource<WinRateResponse>();
        _api.Performance = (_, _) => curve.Task;
        _api.WinRate = (_, _) => winRate.Task;

        IRenderedComponent<Performance> page = Render<Performance>();

        page.WaitForAssertion(() => Assert.Equal(2, Calls));
        Assert.NotNull(Card(page, "equity").QuerySelector(".solia-loading"));
        Assert.NotNull(Card(page, "winrate").QuerySelector(".solia-loading"));
        Assert.Empty(page.FindComponents<Stub<EquityChart>>());

        curve.SetResult(Curve(1m, (Now, 101m, 1m)));
        winRate.SetResult(new WinRateResponse(1, 1, 0, 0, "all"));

        page.WaitForAssertion(() => Assert.Single(page.FindComponents<Stub<EquityChart>>()));
        Assert.Null(Card(page, "equity").QuerySelector(".solia-loading"));
    }

    [Fact]
    public void With_no_closed_trades_both_panels_say_so_and_draw_no_chart()
    {
        _api.Performance = (_, _) => Task.FromResult(Curve(0m));
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0, 0, 0, 0, "all"));

        IRenderedComponent<Performance> page = Render<Performance>();

        page.WaitForAssertion(() =>
        {
            Assert.Contains("No closed trades in this session yet.", Card(page, "equity").TextContent);
            Assert.Contains("No closed trades in this session yet.", Card(page, "winrate").TextContent);
        });
        Assert.Empty(page.FindComponents<Stub<EquityChart>>());
        Assert.Empty(page.FindComponents<Stub<WinRateDonut>>());
    }

    [Fact]
    public void A_failing_panel_shows_its_own_error_and_recovers_on_retry()
    {
        int attempts = 0;
        _api.Performance = (_, _) => ++attempts == 1
            ? Task.FromException<PerformanceResponse>(new BackendUnavailableException())
            : Task.FromResult(Curve(1m, (Now, 101m, 1m)));

        IRenderedComponent<Performance> page = Render<Performance>();

        page.WaitForAssertion(() => Assert.NotNull(Card(page, "equity").QuerySelector(".solia-error__retry")));
        Assert.Contains("Could not load the equity curve", Card(page, "equity").TextContent);
        page.WaitForAssertion(() => Assert.Single(page.FindComponents<Stub<WinRateDonut>>()));

        Card(page, "equity").QuerySelector(".solia-error__retry")!.Click();

        page.WaitForAssertion(() => Assert.Single(page.FindComponents<Stub<EquityChart>>()));
        Assert.Equal(101m, ChartCurve(page).Points.Single().Equity);
    }

    [Fact]
    public void Choosing_a_length_re_queries_the_curve_only()
    {
        IRenderedComponent<Performance> page = RenderLoaded();
        _api.Performance = (_, _) => Task.FromResult(Curve(1m, (Now, 101m, 1m)));

        page.Find("[data-limit='100']").Click();

        page.WaitForAssertion(() => Assert.Equal([1001, 101], _api.PerformanceCalls));
        page.WaitForAssertion(() => Assert.Single(ChartCurve(page).Points));
        Assert.Single(_api.WinRateCalls);
        Assert.Contains("mud-button-filled", page.Find("[data-limit='100']").ClassName);
        Assert.DoesNotContain("mud-button-filled", page.Find("[data-limit='1000']").ClassName);
    }

    [Fact]
    public void Every_offered_length_is_queried_one_point_longer()
    {
        IRenderedComponent<Performance> page = RenderLoaded();

        foreach (int limit in new[] { 500, 5000, 1000 })
        {
            page.Find($"[data-limit='{limit}']").Click();
        }

        page.WaitForAssertion(() => Assert.Equal([1001, 501, 5001, 1001], _api.PerformanceCalls));
    }

    [Fact]
    public async Task A_burst_of_closes_refreshes_both_panels_once_after_the_delay_without_a_reload()
    {
        IRenderedComponent<Performance> page = RenderLoaded();
        page.Find("[data-limit='500']").Click();
        page.WaitForAssertion(() => Assert.Equal(2, _api.PerformanceCalls.Count));
        _api.Performance = (_, _) => Task.FromResult(Curve(4.5m, (Now.AddHours(-1), 103.2m, 1.7m), (Now, 104.5m, 1.3m)));
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0.6, 3, 2, 0, "all"));

        for (int i = 0; i < 3; i++)
        {
            await _feed.PushAsync(LiveFeedEvents.TradeClosed, Trade(i));
            await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(i, Now));
        }

        Assert.Equal(3, Calls); // nothing re-fetched yet: the burst is gathered
        _clock.Elapse(Performance.RefreshDelay);

        page.WaitForAssertion(() => Assert.Equal(104.5m, ChartCurve(page).Points[^1].Equity));
        Assert.Equal([3.0, 2.0, 0.0], DonutSlices(page).Values);
        Assert.Equal([1001, 501, 501], _api.PerformanceCalls); // the chosen length is kept
        Assert.Equal(2, _api.WinRateCalls.Count);
        Assert.Equal("Total P&L +4.5", Text(Card(page, "equity"), "[data-part=total]"));
    }

    [Fact]
    public async Task A_resync_refreshes_both_panels_at_once_and_drops_the_pending_refresh()
    {
        IRenderedComponent<Performance> page = RenderLoaded();
        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Trade(1));

        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.Equal(4, Calls));
        Assert.False(_clock.IsArmed(Performance.RefreshDelay));
        _clock.Elapse(Performance.RefreshDelay);
        Assert.Equal(4, Calls);
    }

    [Fact]
    public void With_no_event_the_panels_still_re_fetch_periodically()
    {
        IRenderedComponent<Performance> page = RenderLoaded();

        _clock.Tick(Performance.ReconcileInterval);
        _clock.Elapse(Performance.RefreshDelay);

        page.WaitForAssertion(() => Assert.Equal(4, Calls));
    }

    [Fact]
    public async Task Leaving_the_page_unsubscribes_and_stops_the_pending_refresh()
    {
        RenderLoaded();
        Assert.Equal(1, _feed.Subscribers(LiveFeedEvents.StatsChanged));
        Assert.Equal(1, _feed.Subscribers(LiveFeedEvents.TradeClosed));
        Assert.Equal(1, _feed.ResyncSubscribers);
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));

        await DisposeComponentsAsync();
        _clock.Tick(Performance.ReconcileInterval);
        _clock.Elapse(Performance.RefreshDelay);

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.StatsChanged));
        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.TradeClosed));
        Assert.Equal(0, _feed.ResyncSubscribers);
        Assert.Equal(2, Calls);
    }

    private static TradeDto Trade(long id) =>
        new(id, "mint", null, Now.AddMinutes(-5), Now, 1m, 1.1m, 1m, 300, 0.1m, 0.1, "buy", 70, "tp", "closed");

    [Fact]
    public void The_win_rate_card_breaks_the_exits_down()
    {
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0.5, 2, 2, 0, "all", 3, new ExitBreakdown(2, 1, 0, 1)));

        IRenderedComponent<Performance> page = RenderLoaded();

        IElement card = Card(page, "winrate");
        Assert.Equal("2 take profit", Text(card, "[data-exit=tp]"));
        Assert.Equal("1 stop loss", Text(card, "[data-exit=sl]"));
        Assert.Equal("0 sell signal", Text(card, "[data-exit=signal]"));
        Assert.Equal("1 timeout", Text(card, "[data-exit=timeout]"));
    }

    [Fact]
    public async Task Picking_a_session_reloads_the_curve_and_the_win_rate_for_it()
    {
        IRenderedComponent<Performance> page = RenderLoaded();
        Assert.Equal([SessionSelection.Current], _api.PerformanceSessions);

        await page.InvokeAsync(() =>
            page.FindComponent<MudBlazor.MudSelect<SessionSelection>>().Instance.ValueChanged.InvokeAsync(SessionSelection.Of(1)));

        page.WaitForAssertion(() =>
        {
            Assert.Equal(SessionSelection.Of(1), _api.PerformanceSessions[^1]);
            Assert.Equal(SessionSelection.Of(1), _api.WinRateSessions[^1]);
        });
        Assert.Equal(Performance.DefaultLimit + 1, _api.PerformanceCalls[^1]); // the length is kept
    }
}
