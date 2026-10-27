using System.Text.Json;
using AngleSharp.Dom;
using Microsoft.Extensions.DependencyInjection;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Tests.Api;
using Solia.Dashboard.Tests.Live;

namespace Solia.Dashboard.Tests.Pages;

public class OverviewTests : MudTestContext
{
    private static readonly DateTimeOffset Now = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();
    private readonly FakeLiveFeed _feed = new();
    private readonly ManualClock _clock = new(Now);

    public OverviewTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Services.AddSingleton<ILiveFeed>(_feed);
        Services.AddSingleton<TimeProvider>(_clock);
        _api.WinRate = (w, _) => Task.FromResult(new WinRateResponse(0.58, 290, 190, 20, "all"));
        _api.Performance = (_, _) => Task.FromResult(new PerformanceResponse([], 3.2m));
        _api.ModelStatus = _ => Task.FromResult(Status("v3"));
    }

    private static ModelStatusResponse Status(string? champion, long sinceTrain = 142, int threshold = 500) =>
        new(champion, champion is null ? null : Now.AddHours(-3), new Dictionary<string, JsonElement>(), sinceTrain, threshold);

    private IRenderedComponent<Overview> RenderLoaded()
    {
        IRenderedComponent<Overview> page = Render<Overview>();
        page.WaitForAssertion(() =>
        {
            Assert.NotNull(page.Find("[data-card=winrate] .solia-stat-card__value"));
            Assert.NotNull(page.Find("[data-card=pnl] .solia-stat-card__value"));
            Assert.NotNull(page.Find("[data-card=model] .solia-stat-card__value"));
        });
        return page;
    }

    private static IElement Card(IRenderedComponent<Overview> page, string name) => page.Find($"[data-card={name}]");

    private static string Text(IElement card, string selector) => card.QuerySelector(selector)!.TextContent.Trim();

    private int Calls => _api.WinRateCalls.Count + _api.PerformanceCalls.Count + _api.ModelStatusCalls;

    [Fact]
    public void Win_rate_shows_the_backends_ratio_and_its_breakdown()
    {
        IRenderedComponent<Overview> page = RenderLoaded();

        IElement card = Card(page, "winrate");
        Assert.Equal("58.0%", Text(card, ".solia-stat-card__value"));
        Assert.Equal("290 wins", Text(card, "[data-part=wins]"));
        Assert.Equal("190 losses", Text(card, "[data-part=losses]"));
        Assert.Equal("20 break-even", Text(card, "[data-part=break-even]"));
        Assert.Null(card.QuerySelector(".solia-winrate__empty"));
        Assert.Equal([StatsWindow.All], _api.WinRateCalls);
    }

    [Fact]
    public void Win_rate_is_a_placeholder_when_nothing_closed_in_the_window()
    {
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0, 0, 0, 0, "24h"));

        IRenderedComponent<Overview> page = RenderLoaded();

        IElement card = Card(page, "winrate");
        Assert.Equal("—", Text(card, ".solia-stat-card__value"));
        Assert.Equal("No closed trades in this window.", Text(card, ".solia-winrate__empty"));
        Assert.Null(card.QuerySelector(".solia-winrate__breakdown")); // no redundant 0 · 0 · 0
    }

    [Theory]
    [InlineData("3.2", "+3.2", "solia-pnl--gain")]
    [InlineData("-0.045", "-0.045", "solia-pnl--loss")]
    [InlineData("0", "0", null)]
    public void Total_pnl_is_signed_and_coloured_by_sign(string pnl, string expected, string? cssClass)
    {
        _api.Performance = (_, _) => Task.FromResult(new PerformanceResponse([], decimal.Parse(pnl, System.Globalization.CultureInfo.InvariantCulture)));

        IRenderedComponent<Overview> page = RenderLoaded();

        IElement value = Card(page, "pnl").QuerySelector(".solia-stat-card__value")!;
        Assert.Equal(expected, value.TextContent.Trim());
        Assert.Equal(cssClass is not null, value.ClassList.Contains("solia-pnl--gain") || value.ClassList.Contains("solia-pnl--loss"));
        if (cssClass is not null)
        {
            Assert.Contains(cssClass, value.ClassList);
        }
        Assert.Equal([1], _api.PerformanceCalls); // only total_pnl is shown: the smallest curve
    }

    [Fact]
    public void Switching_the_window_re_queries_the_win_rate_only()
    {
        IRenderedComponent<Overview> page = RenderLoaded();

        page.Find("[data-window='7d']").Click();

        page.WaitForAssertion(() => Assert.Equal([StatsWindow.All, StatsWindow.Last7Days], _api.WinRateCalls));
        Assert.Single(_api.PerformanceCalls);
        Assert.Equal(1, _api.ModelStatusCalls);
        Assert.Contains("mud-button-filled", page.Find("[data-window='7d']").ClassName);
        Assert.DoesNotContain("mud-button-filled", page.Find("[data-window='all']").ClassName);
    }

    [Fact]
    public void Every_window_maps_to_its_wire_value()
    {
        IRenderedComponent<Overview> page = RenderLoaded();

        foreach (string window in new[] { "24h", "30d", "all" })
        {
            page.Find($"[data-window='{window}']").Click();
        }

        page.WaitForAssertion(() => Assert.Equal(
            [StatsWindow.All, StatsWindow.Last24Hours, StatsWindow.Last30Days, StatsWindow.All], _api.WinRateCalls));
    }

    [Fact]
    public async Task A_burst_of_closes_refreshes_every_card_once_after_the_delay()
    {
        IRenderedComponent<Overview> page = RenderLoaded();
        _api.WinRate = (_, _) => Task.FromResult(new WinRateResponse(0.6, 3, 2, 0, "all"));
        _api.Performance = (_, _) => Task.FromResult(new PerformanceResponse([], 4.5m));

        for (int i = 0; i < 3; i++)
        {
            await _feed.PushAsync(LiveFeedEvents.TradeClosed, Trade(i));
            await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(i, Now));
        }

        Assert.Equal(3, Calls); // nothing re-fetched yet: the burst is gathered
        _clock.Elapse(Overview.RefreshDelay);

        page.WaitForAssertion(() => Assert.Equal("+4.5", Text(Card(page, "pnl"), ".solia-stat-card__value")));
        Assert.Equal("60.0%", Text(Card(page, "winrate"), ".solia-stat-card__value"));
        Assert.Equal(2, _api.WinRateCalls.Count);
        Assert.Equal(2, _api.PerformanceCalls.Count);
        Assert.Equal(2, _api.ModelStatusCalls);
    }

    [Fact]
    public async Task A_live_refresh_keeps_the_selected_window()
    {
        IRenderedComponent<Overview> page = RenderLoaded();
        page.Find("[data-window='24h']").Click();
        page.WaitForAssertion(() => Assert.Equal(2, _api.WinRateCalls.Count));

        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));
        _clock.Elapse(Overview.RefreshDelay);

        page.WaitForAssertion(() => Assert.Equal(3, _api.WinRateCalls.Count));
        Assert.Equal(StatsWindow.Last24Hours, _api.WinRateCalls[^1]);
    }

    [Fact]
    public async Task A_resync_refreshes_every_card_at_once_and_drops_the_pending_refresh()
    {
        IRenderedComponent<Overview> page = RenderLoaded();
        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Trade(1));

        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.Equal(6, Calls));
        Assert.False(_clock.IsArmed(Overview.RefreshDelay));
        _clock.Elapse(Overview.RefreshDelay);
        Assert.Equal(6, Calls);
    }

    [Fact]
    public async Task A_background_refresh_while_the_backend_is_down_keeps_the_figures_marked_stale()
    {
        IRenderedComponent<Overview> page = RenderLoaded();
        _api.Performance = (_, _) => Task.FromException<PerformanceResponse>(new BackendUnavailableException());

        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.NotNull(Card(page, "pnl").QuerySelector(".solia-stale")));
        Assert.Equal("+3.2", Text(Card(page, "pnl"), ".solia-stat-card__value"));
        Assert.Null(Card(page, "winrate").QuerySelector(".solia-stale"));

        _api.Performance = (_, _) => Task.FromResult(new PerformanceResponse([], 4.5m));
        page.Find("[data-card=pnl] .solia-stale button.solia-stale__retry").Click();

        page.WaitForAssertion(() => Assert.Equal("+4.5", Text(Card(page, "pnl"), ".solia-stat-card__value")));
        Assert.Null(Card(page, "pnl").QuerySelector(".solia-stale"));
    }

    [Fact]
    public void A_failing_card_shows_its_own_error_and_leaves_the_others()
    {
        _api.WinRate = (_, _) => Task.FromException<WinRateResponse>(new BackendUnavailableException());

        IRenderedComponent<Overview> page = Render<Overview>();

        page.WaitForAssertion(() => Assert.NotNull(Card(page, "winrate").QuerySelector(".solia-error")));
        page.WaitForAssertion(() => Assert.Equal("+3.2", Text(Card(page, "pnl"), ".solia-stat-card__value")));
        Assert.Equal("v3", Text(Card(page, "model"), ".solia-model__champion"));
        Assert.Contains("Could not load the win-rate", Card(page, "winrate").TextContent);
        Assert.Null(Card(page, "pnl").QuerySelector(".solia-error"));
        Assert.Null(Card(page, "model").QuerySelector(".solia-error"));
    }

    [Fact]
    public void A_failed_card_recovers_on_retry()
    {
        int attempts = 0;
        _api.ModelStatus = _ => ++attempts == 1
            ? Task.FromException<ModelStatusResponse>(new BackendUnavailableException())
            : Task.FromResult(Status("v4"));
        IRenderedComponent<Overview> page = Render<Overview>();
        page.WaitForAssertion(() => Assert.NotNull(Card(page, "model").QuerySelector(".solia-error__retry")));

        Card(page, "model").QuerySelector(".solia-error__retry")!.Click();

        page.WaitForAssertion(() => Assert.Equal("v4", Text(Card(page, "model"), ".solia-model__champion")));
    }

    [Fact]
    public void The_model_card_shows_the_champion_and_the_retrain_progress()
    {
        IRenderedComponent<Overview> page = RenderLoaded();

        IElement card = Card(page, "model");
        Assert.Equal("v3", Text(card, ".solia-model__champion"));
        Assert.Equal("Trained 2026-10-27 09:00:00 UTC", Text(card, ".solia-model__trained"));
        Assert.Contains("142 / 500 trades since last train", Text(card, ".solia-model__retrain"));
        Assert.Null(card.QuerySelector(".solia-model__none"));
    }

    [Fact]
    public void A_champion_without_a_training_time_omits_that_line()
    {
        _api.ModelStatus = _ => Task.FromResult(Status("v3") with { CreatedAt = null });

        IRenderedComponent<Overview> page = RenderLoaded();

        IElement card = Card(page, "model");
        Assert.Equal("v3", Text(card, ".solia-model__champion"));
        Assert.Null(card.QuerySelector(".solia-model__trained"));
        Assert.DoesNotContain("—", card.TextContent);
    }

    [Fact]
    public void With_no_event_the_cards_still_re_fetch_periodically()
    {
        IRenderedComponent<Overview> page = RenderLoaded();
        page.Find("[data-window='24h']").Click();
        page.WaitForAssertion(() => Assert.Equal(2, _api.WinRateCalls.Count));
        _api.ModelStatus = _ => Task.FromResult(Status("v4", sinceTrain: 0));

        _clock.Tick(Overview.ReconcileInterval);
        _clock.Elapse(Overview.RefreshDelay);

        page.WaitForAssertion(() => Assert.Equal("v4", Text(Card(page, "model"), ".solia-model__champion")));
        Assert.Contains("0 / 500 trades since last train", Text(Card(page, "model"), ".solia-model__retrain"));
        Assert.Equal(StatsWindow.Last24Hours, _api.WinRateCalls[^1]);
        Assert.Equal(3, _api.WinRateCalls.Count);
        Assert.Equal(2, _api.PerformanceCalls.Count);
    }

    [Fact]
    public async Task A_reconcile_tick_rides_along_a_pending_live_refresh()
    {
        RenderLoaded();
        await _feed.PushAsync(LiveFeedEvents.StatsChanged, new StatsChangedDto(1, Now));

        _clock.Tick(Overview.ReconcileInterval);
        _clock.Elapse(Overview.RefreshDelay);

        Assert.Equal(6, Calls); // one refresh, not two
    }

    [Fact]
    public void A_null_champion_degrades_to_a_placeholder_and_keeps_the_retrain_progress()
    {
        _api.ModelStatus = _ => Task.FromResult(Status(null, sinceTrain: 1234, threshold: 2000));

        IRenderedComponent<Overview> page = RenderLoaded();

        IElement card = Card(page, "model");
        Assert.Equal("—", Text(card, ".solia-model__champion"));
        Assert.Contains("No champion serving", Text(card, ".solia-model__none"));
        Assert.Null(card.QuerySelector(".solia-model__trained"));
        Assert.Contains("1,234 / 2,000 trades since last train", Text(card, ".solia-model__retrain"));
        Assert.Null(card.QuerySelector(".solia-error"));
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
        _clock.Tick(Overview.ReconcileInterval);
        _clock.Elapse(Overview.RefreshDelay);

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.StatsChanged));
        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.TradeClosed));
        Assert.Equal(0, _feed.ResyncSubscribers);
        Assert.Equal(3, Calls);
    }

    private static TradeDto Trade(long id) =>
        new(id, "mint", null, Now.AddMinutes(-5), Now, 1m, 1.1m, 1m, 300, 0.1m, 0.1, "buy", 70, "tp", "closed");
}
