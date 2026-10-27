using System.Globalization;
using System.Net;
using AngleSharp.Dom;
using Bunit.TestDoubles;
using Microsoft.Extensions.DependencyInjection;
using Solia.Dashboard.Components.Charts;
using Solia.Dashboard.Components.Layout;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Navigation;
using Solia.Dashboard.Stats;

namespace Solia.Dashboard.Tests.EndToEnd;

/// <summary>
/// The Sprint-5 Definition of Done, end to end: the real dashboard host (<see cref="DashboardHost"/>) against a
/// contract-faithful backend (<see cref="StubBackend"/>) seeded with documented JSON. Every view renders the seeded
/// data through the real REST client, pushed hub events appear without a reload, and an outage of either channel
/// degrades gracefully and recovers. The pages render under bUnit on the host's own services (bUnit only supplies
/// the renderer, MudBlazor and navigation), since the views fetch only once interactive — a plain HTTP GET sees
/// their loading skeleton. Everything runs in memory (TestServer, no socket, no Docker), so it is part of the fast
/// local suite, not <c>Category=Integration</c>.
/// </summary>
public class DashboardEndToEndTests : MudTestContext, IAsyncLifetime
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(15);

    // --- Seed (docs/API-CONTRACTS.md shapes, raw snake_case) ---

    private const string MintA = "So1aNaMintAddressAAAAAAAAAAAAAAAAAAAAAAAaaa1";
    private const string MintB = "So1aNaMintAddressBBBBBBBBBBBBBBBBBBBBBBBbbb2";
    private const string MintC = "So1aNaMintAddressCCCCCCCCCCCCCCCCCCCCCCCccc3";

    private static string Prediction(long id, string mint, string symbol, double pump, double rug, string strategy) =>
        string.Create(CultureInfo.InvariantCulture, $$"""
        { "id": {{id}}, "mint": "{{mint}}", "symbol": "{{symbol}}", "ts": "{{DateTimeOffset.UtcNow.AddSeconds(-30):O}}",
          "horizon": "1m", "pump_prob": {{pump}}, "rug_prob": {{rug}}, "strategy": "{{strategy}}", "confidence": 74.0,
          "position_size": 0.5, "take_profit": 0.0000001500, "stop_loss": 0.0000001100, "model_version": "v3" }
        """);

    private static readonly string LivePredictionsJson =
        $"[ {Prediction(801, MintA, "ALPHA", 0.81, 0.07, "buy")}, {Prediction(802, MintB, "BETA", 0.2, 0.6, "hold")} ]";

    private static readonly string PredictionHistoryJson =
        $$"""{ "items": [ {{Prediction(801, MintA, "ALPHA", 0.81, 0.07, "buy")}} ], "page": 1, "page_size": 50, "total": 1 }""";

    private const string ClosedTradeJson = """
        { "id": 7, "mint": "So1aNaMintAddressAAAAAAAAAAAAAAAAAAAAAAAaaa1", "prediction_id": 801,
          "opened_at": "2026-10-26T12:00:00Z", "closed_at": "2026-10-26T12:01:30Z", "entry_price": 0.0000001234,
          "exit_price": 0.0000001500, "size": 0.5, "duration_sec": 90, "pnl": 0.0789, "pnl_pct": 0.2166,
          "strategy": "buy", "confidence": 74.0, "exit_reason": "tp", "status": "closed" }
        """;

    private const string OpenedTradeJson = """
        { "id": 8, "mint": "So1aNaMintAddressCCCCCCCCCCCCCCCCCCCCCCCccc3", "prediction_id": 803,
          "opened_at": "2026-10-27T09:00:00Z", "closed_at": null, "entry_price": 0.0000002000, "exit_price": null,
          "size": 0.25, "duration_sec": null, "pnl": null, "pnl_pct": null, "strategy": "buy", "confidence": 80.0,
          "exit_reason": null, "status": "open" }
        """;

    private const string TradeHistoryJson =
        $$"""{ "items": [ {{ClosedTradeJson}} ], "page": 1, "page_size": 50, "total": 1 }""";

    private const string WinRateJson = """{ "winrate": 0.58, "wins": 29, "losses": 19, "break_even": 2, "window": "all" }""";

    private const string PerformanceJson = """
        { "points": [ { "ts": "2026-10-26T12:01:30Z", "equity": 10.5, "pnl": 0.5 },
                      { "ts": "2026-10-26T13:00:00Z", "equity": 10.2, "pnl": -0.3 },
                      { "ts": "2026-10-26T14:00:00Z", "equity": 11.0, "pnl": 0.8 } ],
          "total_pnl": 1.0 }
        """;

    private const string ModelStatusJson = """
        { "champion_version": "v3", "created_at": "2026-10-26T09:00:00Z", "metrics": { "auc": 0.71 },
          "trades_since_last_train": 142, "retrain_threshold": 500 }
        """;

    private const string LogsJson = """
        [ { "id": 2, "ts": "2026-10-27T10:00:05Z", "service": "bot", "level": "warn", "message": "Skipped a stale prediction", "context": { "mint": "So1a" } },
          { "id": 1, "ts": "2026-10-27T10:00:00Z", "service": "backend", "level": "info", "message": "Bot started", "context": null } ]
        """;

    private StubBackend _backend = null!;
    private DashboardHost _dashboard = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        _backend = await StubBackend.StartAsync();
        _backend.Respond("predictions/live", LivePredictionsJson);
        _backend.Respond("predictions/history", PredictionHistoryJson);
        _backend.Respond("trades/history", TradeHistoryJson);
        _backend.Respond("stats/winrate", WinRateJson);
        _backend.Respond("stats/performance", PerformanceJson);
        _backend.Respond("model/status", ModelStatusJson);
        _backend.Respond("logs/bot", LogsJson);

        _dashboard = await DashboardHost.StartAsync(_backend, Timeout);
        Services.AddFallbackServiceProvider(_dashboard.Services);
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        await DisposeAsync(); // the renderer first: its components unsubscribe from the feed
        await _dashboard.DisposeAsync();
        await _backend.DisposeAsync();
    }

    private static string Cell(IElement row, string column) => row.QuerySelector($"td[data-col={column}]")!.TextContent.Trim();

    // ------------------------------------------------------------------ boot

    [Fact]
    public async Task Every_view_is_served_by_the_booted_host()
    {
        HttpClient client = _dashboard.CreateClient();

        foreach (NavItem item in DashboardNav.Items)
        {
            HttpResponseMessage response = await client.GetAsync(item.Href);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            // The nav lists every title on every page: the page's own <title> shows the route rendered its view.
            Assert.Contains($"<title>{item.Title} · Solia</title>", await response.Content.ReadAsStringAsync());
        }
        Assert.Equal(LiveFeedState.Connected, _dashboard.LiveFeed.State);
    }

    // ------------------------------------------------------------------ views render the seeded data

    [Fact]
    public void Overview_shows_the_seeded_win_rate_pnl_and_model()
    {
        IRenderedComponent<Overview> page = Render<Overview>();

        page.WaitForAssertion(() =>
        {
            Assert.Equal("58.0%", page.Find("[data-card=winrate] .solia-stat-card__value").TextContent.Trim());
            Assert.Equal("+1", page.Find("[data-card=pnl] .solia-stat-card__value").TextContent.Trim());
            Assert.Equal("v3", page.Find("[data-card=model] .solia-model__champion").TextContent.Trim());
        }, Timeout);
        Assert.Empty(page.FindAll(".solia-error"));
    }

    [Fact]
    public async Task Live_predictions_show_the_seeded_rows_and_a_pushed_prediction_appears()
    {
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(2, page.FindAll("tbody tr").Count), Timeout);
        Assert.Equal("81.0%", Cell(page.Find($"tbody tr[data-mint={MintA}]"), "pump"));
        Assert.Equal("HOLD", Cell(page.Find($"tbody tr[data-mint={MintB}]"), "strategy"));

        await _backend.PushAsync(LiveFeedEvents.PredictionCreated.Name, Prediction(803, MintC, "GAMMA", 0.9, 0.05, "buy"));

        page.WaitForAssertion(() =>
        {
            Assert.Equal(3, page.FindAll("tbody tr").Count);
            Assert.Equal("90.0%", Cell(page.Find($"tbody tr[data-mint={MintC}]"), "pump"));
        }, Timeout);
        Assert.Equal(1, _backend.RequestCount("predictions/live")); // pushed, not re-fetched
    }

    [Fact]
    public void Prediction_history_shows_the_seeded_page()
    {
        IRenderedComponent<PredictionHistory> page = Render<PredictionHistory>();

        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr[data-id='801']")), Timeout);
        Assert.Contains("ALPHA", page.Find("tbody tr[data-id='801'] .solia-token").TextContent);
    }

    [Fact]
    public async Task Trade_history_shows_the_seeded_page_and_a_pushed_trade_appears_on_top()
    {
        IRenderedComponent<TradeHistory> page = Render<TradeHistory>();
        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr[data-id='7']")), Timeout);
        Assert.Equal("+0.0789", Cell(page.Find("tbody tr[data-id='7']"), "pnl"));

        await _backend.PushAsync(LiveFeedEvents.TradeOpened.Name, OpenedTradeJson);

        page.WaitForAssertion(() =>
            Assert.Equal(["8", "7"], page.FindAll("tbody tr[data-id]").Select(r => r.GetAttribute("data-id")!)), Timeout);
        Assert.Equal("open", page.Find("tbody tr[data-id='8']").GetAttribute("data-status"));
        Assert.Equal(1, _backend.RequestCount("trades/history"));
    }

    [Fact]
    public void Performance_maps_the_seeded_curve_and_win_rate_into_the_charts()
    {
        ComponentFactories.AddStub<EquityChart>();
        ComponentFactories.AddStub<WinRateDonut>();

        IRenderedComponent<Performance> page = Render<Performance>();

        page.WaitForAssertion(() =>
        {
            EquityCurve curve = page.FindComponent<Stub<EquityChart>>().Instance.Parameters.Get(c => c.Curve);
            Assert.Equal([10.5m, 10.2m, 11.0m], curve.Points.Select(p => p.Equity));
            Assert.Equal(10.0m, curve.StartingBalance);
            WinRateSlices slices = page.FindComponent<Stub<WinRateDonut>>().Instance.Parameters.Get(c => c.Slices);
            Assert.Equal(0.58, slices.Winrate);
        }, Timeout);
        Assert.Empty(page.FindAll(".solia-error"));
    }

    [Fact]
    public async Task Bot_logs_show_the_seeded_entries_and_a_pushed_entry_appears()
    {
        IRenderedComponent<BotLogs> page = Render<BotLogs>();
        page.WaitForAssertion(() => Assert.Equal(2, page.FindAll("tbody tr[data-id]").Count), Timeout);
        Assert.Equal("Skipped a stale prediction", Cell(page.Find("tbody tr[data-id='2']"), "message"));

        await _backend.PushAsync(LiveFeedEvents.LogWritten.Name, """
            { "id": 3, "ts": "2026-10-27T10:00:09Z", "service": "bot", "level": "error", "message": "Scoring failed", "context": null }
            """);

        page.WaitForAssertion(() =>
            Assert.Equal(["3", "2", "1"], page.FindAll("tbody tr[data-id]").Select(r => r.GetAttribute("data-id")!)), Timeout);
    }

    // ------------------------------------------------------------------ backend down

    [Fact]
    public void Rest_down_shows_each_views_error_state_and_retry_recovers_once_it_is_back()
    {
        _backend.RestDown = true;

        IRenderedComponent<TradeHistory> trades = Render<TradeHistory>();
        IRenderedComponent<LivePredictions> predictions = Render<LivePredictions>();
        IRenderedComponent<Overview> overview = Render<Overview>();

        trades.WaitForAssertion(() => Assert.NotNull(trades.Find(".solia-error button.solia-error__retry")), Timeout);
        predictions.WaitForAssertion(() => Assert.NotNull(predictions.Find(".solia-error button.solia-error__retry")), Timeout);
        overview.WaitForAssertion(() => Assert.Equal(3, overview.FindAll("[data-card] .solia-error").Count), Timeout);
        Assert.Equal(LiveFeedState.Connected, _dashboard.LiveFeed.State); // REST down alone leaves live updates on

        _backend.RestDown = false;
        trades.Find(".solia-error button.solia-error__retry").Click();
        predictions.Find(".solia-error button.solia-error__retry").Click();

        trades.WaitForAssertion(() => Assert.Single(trades.FindAll("tbody tr[data-id='7']")), Timeout);
        predictions.WaitForAssertion(() => Assert.Equal(2, predictions.FindAll("tbody tr").Count), Timeout);
        Assert.Empty(trades.FindAll(".solia-error"));
    }

    [Fact]
    public async Task Hub_down_shows_the_indicator_keeps_the_views_and_a_reconnect_resyncs_them()
    {
        IRenderedComponent<LiveFeedIndicator> indicator = Render<LiveFeedIndicator>();
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(2, page.FindAll("tbody tr").Count), Timeout);
        Assert.Equal("connected", indicator.Find(".solia-live-indicator").GetAttribute("data-state"));

        _backend.TakeHubDown();

        indicator.WaitForAssertion(() =>
        {
            Assert.Equal("reconnecting", indicator.Find(".solia-live-indicator").GetAttribute("data-state"));
            Assert.Contains("RECONNECTING", indicator.Markup);
        }, Timeout);
        Assert.Equal(2, page.FindAll("tbody tr").Count); // degraded, not blanked
        Assert.Empty(page.FindAll(".solia-error"));

        // A prediction made while the hub was down never reaches the dashboard as an event...
        _backend.Respond("predictions/live",
            $"[ {Prediction(801, MintA, "ALPHA", 0.81, 0.07, "buy")}, {Prediction(802, MintB, "BETA", 0.2, 0.6, "hold")}, {Prediction(803, MintC, "GAMMA", 0.9, 0.05, "buy")} ]");
        _backend.BringHubUp();

        // ...the reconnect's resync re-fetches it.
        indicator.WaitForAssertion(
            () => Assert.Equal("connected", indicator.Find(".solia-live-indicator").GetAttribute("data-state")), Timeout);
        page.WaitForAssertion(() => Assert.NotNull(page.Find($"tbody tr[data-mint={MintC}]")), Timeout);
        Assert.Equal(2, _backend.RequestCount("predictions/live"));

        // And live pushes flow again.
        await _backend.PushAsync(LiveFeedEvents.PredictionCreated.Name, Prediction(804, MintB, "BETA", 0.95, 0.01, "buy"));
        page.WaitForAssertion(() => Assert.Equal("95.0%", Cell(page.Find($"tbody tr[data-mint={MintB}]"), "pump")), Timeout);
    }
}
