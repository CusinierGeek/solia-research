using AngleSharp.Dom;
using Microsoft.Extensions.DependencyInjection;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Tests.Api;
using Solia.Dashboard.Tests.Live;

namespace Solia.Dashboard.Tests.Pages;

public class LivePredictionsTests : MudTestContext
{
    private static readonly DateTimeOffset Now = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();
    private readonly FakeLiveFeed _feed = new();
    private readonly ManualClock _clock = new(Now);

    public LivePredictionsTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Services.AddSingleton<ILiveFeed>(_feed);
        Services.AddSingleton<TimeProvider>(_clock);
    }

    private static PredictionRecordDto Prediction(
        long id, string mint, string horizon = "1m", double? pump = 0.3, double? rug = 0.1, string? strategy = "hold",
        double? confidence = 50, int secondsAgo = 10, string? symbol = "ABC", string? model = "v3") =>
        new(id, mint, symbol, Now.AddSeconds(-secondsAgo), horizon, pump, rug, strategy, confidence, null, null, null, model);

    private void Returns(params PredictionRecordDto[] rows) =>
        _api.LivePredictions = (_, _) => Task.FromResult<IReadOnlyList<PredictionRecordDto>>(rows);

    private static IReadOnlyList<IElement> Rows(IRenderedComponent<LivePredictions> page) =>
        page.FindAll("tbody tr");

    private static string Cell(IElement row, string column) =>
        row.QuerySelector($"td[data-col={column}]")!.TextContent.Trim();

    [Fact]
    public void Renders_the_fetched_predictions()
    {
        const string mint = "So1aNaMintAddress1111111111111111111111xyz9";
        Returns(Prediction(1, mint, "5m", pump: 0.8123, rug: 0.07, strategy: "buy", confidence: 74.4, secondsAgo: 125));

        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        IElement row = Rows(page)[0];
        Assert.Equal("ABC", row.QuerySelector(".solia-token strong")!.TextContent);
        Assert.Equal("So1a…xyz9", row.QuerySelector(".solia-mint")!.TextContent);
        Assert.Equal(mint, row.QuerySelector(".solia-token")!.GetAttribute("title"));
        Assert.Equal("5m", row.GetAttribute("data-horizon"));
        Assert.Equal("81.2%", Cell(row, "pump"));
        Assert.Equal("7.0%", Cell(row, "rug"));
        Assert.Equal("BUY", Cell(row, "strategy"));
        Assert.Equal("74", Cell(row, "confidence"));
        Assert.Equal("v3", Cell(row, "model"));
        Assert.Equal("2m", Cell(row, "age"));
        Assert.Equal(LivePredictions.Capacity, Assert.Single(_api.LivePredictionsCalls).Limit);
    }

    [Fact]
    public void Missing_values_show_a_placeholder()
    {
        Returns(Prediction(1, "m1", pump: null, rug: null, strategy: null, confidence: null, symbol: null, model: null));

        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        IElement row = Rows(page)[0];
        Assert.Equal("?", row.QuerySelector(".solia-token strong")!.TextContent);
        Assert.All(["pump", "rug", "strategy", "confidence", "model"], c => Assert.Equal("—", Cell(row, c)));
    }

    [Fact]
    public void High_pump_and_high_rug_probabilities_are_emphasised()
    {
        Returns(
            Prediction(1, "pumpy", pump: 0.9, rug: 0.05),
            Prediction(2, "ruggy", pump: 0.2, rug: 0.6),
            Prediction(3, "calm", pump: 0.3, rug: 0.1));

        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        page.WaitForAssertion(() => Assert.Equal(3, Rows(page).Count));
        IElement Row(string mint) => page.Find($"tbody tr[data-mint={mint}]");
        Assert.NotNull(Row("pumpy").QuerySelector(".solia-high-pump"));
        Assert.Contains("solia-row--pump", Row("pumpy").ClassList);
        Assert.NotNull(Row("ruggy").QuerySelector(".solia-high-rug"));
        Assert.Contains("solia-row--rug", Row("ruggy").ClassList);
        Assert.Null(Row("calm").QuerySelector(".solia-high-pump, .solia-high-rug"));
        Assert.DoesNotContain(Row("calm").ClassList, c => c.StartsWith("solia-row--", StringComparison.Ordinal));
    }

    [Fact]
    public async Task A_pushed_prediction_inserts_a_new_token_and_updates_an_existing_one()
    {
        Returns(Prediction(1, "m1", pump: 0.2, secondsAgo: 30));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(2, "m2", pump: 0.4, secondsAgo: 5));
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(3, "m1", pump: 0.85, secondsAgo: 1));
        page.WaitForAssertion(() =>
        {
            Assert.Equal(2, Rows(page).Count);
            Assert.Equal("85.0%", Cell(page.Find("tbody tr[data-mint=m1]"), "pump"));
        });
        Assert.Single(_api.LivePredictionsCalls); // no reload
    }

    [Fact]
    public async Task A_pushed_prediction_shows_up_on_an_empty_page()
    {
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Contains("No tracked tokens yet.", page.Find(".solia-empty").TextContent));

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(1, "m1"));

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        Assert.Empty(page.FindAll(".solia-empty"));
    }

    [Fact]
    public async Task A_prediction_pushed_while_loading_is_kept_on_top_of_the_snapshot()
    {
        var pending = new TaskCompletionSource<IReadOnlyList<PredictionRecordDto>>();
        _api.LivePredictions = (_, _) => pending.Task;
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(9, "m1", pump: 0.9, secondsAgo: 1));
        pending.SetResult([Prediction(1, "m1", pump: 0.1, secondsAgo: 20), Prediction(2, "m2")]);

        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        Assert.Equal("90.0%", Cell(page.Find("tbody tr[data-mint=m1]"), "pump"));
    }

    [Fact]
    public void Shows_loading_until_the_data_arrives()
    {
        _api.LivePredictions = (_, _) => new TaskCompletionSource<IReadOnlyList<PredictionRecordDto>>().Task;

        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        Assert.NotNull(page.Find(".solia-loading"));
        Assert.Empty(Rows(page));
    }

    [Fact]
    public void An_empty_result_shows_the_empty_state()
    {
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();

        page.WaitForAssertion(() => Assert.Contains("No tracked tokens yet.", page.Find(".solia-empty").TextContent));
        Assert.Empty(page.FindAll("table"));
    }

    [Fact]
    public void A_failure_shows_the_error_state_and_retry_calls_the_client_again()
    {
        _api.LivePredictions = (_, _) => Task.FromException<IReadOnlyList<PredictionRecordDto>>(new BackendUnavailableException());
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Contains("Could not load live predictions", page.Find(".solia-error").TextContent));

        Returns(Prediction(1, "m1"));
        page.Find(".solia-error__retry").Click();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        Assert.Equal(2, _api.LivePredictionsCalls.Count);
        Assert.Empty(page.FindAll(".solia-error"));
    }

    [Fact]
    public async Task A_feed_resync_refetches_the_snapshot()
    {
        Returns(Prediction(1, "m1"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        Returns(Prediction(1, "m1"), Prediction(2, "m2"));
        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        Assert.Equal(2, _api.LivePredictionsCalls.Count);
    }

    [Fact]
    public async Task The_horizon_selector_refetches_that_horizon_and_filters_pushes()
    {
        Returns(Prediction(1, "m1", "1m"), Prediction(2, "m1", "5m"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        Returns(Prediction(2, "m1", "5m"));
        page.Find("button[data-horizon='5m']").Click();

        page.WaitForAssertion(() => Assert.Equal("5m", Assert.Single(Rows(page)).GetAttribute("data-horizon")));
        Assert.Equal(PredictionHorizon.Minutes5, _api.LivePredictionsCalls[^1].Horizon);

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(3, "m2", "1m"));
        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(4, "m3", "5m"));
        page.WaitForAssertion(() => Assert.Equal(["m1", "m3"], Rows(page).Select(r => r.GetAttribute("data-mint")).Order()));

        Returns(Prediction(1, "m1", "1m"), Prediction(2, "m1", "5m"));
        page.Find("button[data-horizon='all']").Click();
        page.WaitForAssertion(() => Assert.Null(_api.LivePredictionsCalls[^1].Horizon));
    }

    [Fact]
    public void Rows_are_newest_first_and_a_header_click_sorts_by_that_column()
    {
        Returns(
            Prediction(1, "a", pump: 0.5, secondsAgo: 30),
            Prediction(2, "b", pump: null, secondsAgo: 1),
            Prediction(3, "c", pump: 0.9, secondsAgo: 60));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(3, Rows(page).Count));
        string[] Order() => Rows(page).Select(r => r.GetAttribute("data-mint")!).ToArray();

        Assert.Equal(["b", "a", "c"], Order());

        page.Find("th[data-sort=pump_prob] button").Click();
        Assert.Equal(["c", "a", "b"], Order());
        Assert.Equal("descending", page.Find("th[data-sort=pump_prob]").GetAttribute("aria-sort"));

        page.Find("th[data-sort=pump_prob] button").Click();
        Assert.Equal(["a", "c", "b"], Order()); // ascending, missing value still last
        Assert.Equal("ascending", page.Find("th[data-sort=pump_prob]").GetAttribute("aria-sort"));
        Assert.Single(_api.LivePredictionsCalls); // client-side
    }

    [Fact]
    public async Task Unsubscribes_when_disposed()
    {
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(1, _feed.Subscribers(LiveFeedEvents.PredictionCreated)));
        Assert.Equal(1, _feed.ResyncSubscribers);

        await DisposeComponentsAsync();

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.PredictionCreated));
        Assert.Equal(0, _feed.ResyncSubscribers);
    }

    [Fact]
    public void The_age_header_reads_youngest_first_by_default_and_flips_on_click()
    {
        Returns(Prediction(1, "a", secondsAgo: 30), Prediction(2, "b", secondsAgo: 1));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        string AgeSort() => page.Find("th[data-sort=ts]").GetAttribute("aria-sort")!;

        Assert.Equal("ascending", AgeSort()); // newest ts first = smallest age first

        page.Find("th[data-sort=ts] button").Click();
        Assert.Equal("descending", AgeSort());
        Assert.Equal(["a", "b"], Rows(page).Select(r => r.GetAttribute("data-mint")));
    }

    [Fact]
    public async Task Pushes_of_another_horizon_never_evict_the_filtered_rows()
    {
        PredictionRecordDto[] fiveMinute = Enumerable.Range(0, LivePredictions.Capacity)
            .Select(i => Prediction(i + 1, $"m{i}", "5m", secondsAgo: 1000 + i))
            .ToArray();
        Returns(fiveMinute);
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(LivePredictions.Capacity, Rows(page).Count));
        page.Find("button[data-horizon='5m']").Click();
        page.WaitForAssertion(() => Assert.Equal(PredictionHorizon.Minutes5, _api.LivePredictionsCalls[^1].Horizon));

        for (int i = 0; i < LivePredictions.Capacity; i++)
        {
            await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(10_000 + i, $"x{i}", "1m", secondsAgo: 1));
        }
        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(20_000, "fresh", "5m", secondsAgo: 1));

        page.WaitForAssertion(() => Assert.Equal("fresh", Rows(page)[0].GetAttribute("data-mint")));
        Assert.Equal(LivePredictions.Capacity, Rows(page).Count);
        Assert.All(Rows(page), r => Assert.Equal("5m", r.GetAttribute("data-horizon")));
        Assert.Empty(page.FindAll($"tbody tr[data-mint=m{LivePredictions.Capacity - 1}]")); // the oldest made room
    }

    [Fact]
    public async Task A_failed_refresh_keeps_the_rows_under_a_warning_and_pushes_still_show()
    {
        Returns(Prediction(1, "m1"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        _api.LivePredictions = (_, _) => Task.FromException<IReadOnlyList<PredictionRecordDto>>(new BackendUnavailableException());
        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-stale")));
        Assert.Single(Rows(page));

        await _feed.PushAsync(LiveFeedEvents.PredictionCreated, Prediction(2, "m2"));
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        Returns(Prediction(1, "m1"));
        page.Find(".solia-stale .solia-stale__retry").Click();
        page.WaitForAssertion(() => Assert.Empty(page.FindAll(".solia-stale")));
        Assert.Single(Rows(page));
    }

    [Fact]
    public void The_snapshot_is_refetched_periodically_and_drops_untracked_tokens()
    {
        Returns(Prediction(1, "m1"), Prediction(2, "gone"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        Returns(Prediction(1, "m1"));
        _clock.Tick(LivePredictions.ReconcileInterval);

        page.WaitForAssertion(() => Assert.Equal("m1", Assert.Single(Rows(page)).GetAttribute("data-mint")));
        Assert.Equal(2, _api.LivePredictionsCalls.Count);
    }

    [Fact]
    public void Age_ticks_redraw_the_table_but_not_the_error_state()
    {
        Returns(Prediction(1, "m1"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        int renders = page.RenderCount;
        _clock.Tick(TimeSpan.FromSeconds(5));
        page.WaitForAssertion(() => Assert.True(page.RenderCount > renders));

        _api.LivePredictions = (_, _) => Task.FromException<IReadOnlyList<PredictionRecordDto>>(new BackendUnavailableException());
        IRenderedComponent<LivePredictions> failed = Render<LivePredictions>();
        failed.WaitForAssertion(() => Assert.NotNull(failed.Find(".solia-error")));
        renders = failed.RenderCount;
        _clock.Tick(TimeSpan.FromSeconds(5));
        Assert.Equal(renders, failed.RenderCount);
    }

    [Fact]
    public async Task A_tick_already_in_flight_when_disposed_does_nothing()
    {
        Returns(Prediction(1, "m1"));
        IRenderedComponent<LivePredictions> page = Render<LivePredictions>();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await DisposeComponentsAsync();
        _clock.Tick(LivePredictions.ReconcileInterval, includeDisposed: true);
        _clock.Tick(TimeSpan.FromSeconds(5), includeDisposed: true);

        Assert.Single(_api.LivePredictionsCalls);
    }
}
