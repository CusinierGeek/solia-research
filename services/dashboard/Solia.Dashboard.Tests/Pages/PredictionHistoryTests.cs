using AngleSharp.Dom;
using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.DependencyInjection;
using MudBlazor;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Tests.Api;

namespace Solia.Dashboard.Tests.Pages;

public class PredictionHistoryTests : MudTestContext
{
    private static readonly DateTimeOffset Ts = new(2026, 10, 27, 11, 58, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();

    public PredictionHistoryTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Navigation.NavigateTo("predictions/history");
    }

    private NavigationManager Navigation => Services.GetRequiredService<NavigationManager>();

    private static PredictionRecordDto Prediction(long id, string mint = "So1aNaMintAddress1111111111111111111111xyz9") =>
        new(id, mint, "ABC", Ts, "5m", 0.8123, 0.07, "buy", 74.4, 0.5m, 0.0000002500m, 0.0000000900m, "v3");

    /// <summary>Answers every request with <paramref name="rows"/> rows (ids from 1) out of <paramref name="total"/>.</summary>
    private void Returns(long total, int rows = 1) =>
        _api.PredictionHistory = (q, _) => Task.FromResult(new PagedResult<PredictionRecordDto>(
            Enumerable.Range(1, rows).Select(i => Prediction(i)).ToArray(), q.Page ?? 1, q.PageSize ?? 50, total));

    private IRenderedComponent<PredictionHistory> RenderLoaded()
    {
        Render<MudPopoverProvider>();
        IRenderedComponent<PredictionHistory> page = Render<PredictionHistory>();
        page.WaitForAssertion(() => Assert.NotEmpty(_api.PredictionHistoryCalls));
        return page;
    }

    private static string Cell(IElement row, string column) =>
        row.QuerySelector($"td[data-col={column}]")!.TextContent.Trim();

    private static IElement PagerButton(IRenderedComponent<PredictionHistory> page, string which) =>
        page.Find($"button[data-pager={which}]");

    private string Query => new Uri(Navigation.Uri).Query;

    [Fact]
    public void Renders_a_page_of_predictions_with_their_sizing()
    {
        Returns(total: 1);

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr")));
        IElement row = page.Find("tbody tr");
        Assert.Equal("2026-10-27 11:58:00", Cell(row, "ts"));
        Assert.Equal("ABC", row.QuerySelector(".solia-token strong")!.TextContent);
        Assert.Equal("So1a…xyz9", row.QuerySelector(".solia-mint")!.TextContent);
        Assert.Equal("5m", Cell(row, "horizon"));
        Assert.Equal("81.2%", Cell(row, "pump"));
        Assert.Equal("7.0%", Cell(row, "rug"));
        Assert.Equal("BUY", Cell(row, "strategy"));
        Assert.Equal("74", Cell(row, "confidence"));
        Assert.Equal("0.5", Cell(row, "size"));
        Assert.Equal("0.00000025", Cell(row, "tp"));
        Assert.Equal("0.00000009", Cell(row, "sl"));
        Assert.Equal("v3", Cell(row, "model"));
        Assert.Equal("1–1 of 1", page.Find(".solia-pager__range").TextContent.Trim());

        PredictionHistoryQuery query = Assert.Single(_api.PredictionHistoryCalls);
        Assert.Equal(new PredictionHistoryQuery { Page = 1, PageSize = 50 }, query);
    }

    [Fact]
    public void Filters_and_page_are_read_from_the_url()
    {
        Navigation.NavigateTo("predictions/history?mint=So1abc&horizon=1h&strategy=sell&from=2026-10-01T00:00:00Z&page=3&pageSize=25");
        Returns(total: 100);

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        Assert.Equal(new PredictionHistoryQuery
        {
            Mint = "So1abc",
            Horizon = PredictionHorizon.Hour1,
            Strategy = TradeStrategy.Sell,
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            Page = 3,
            PageSize = 25,
        }, Assert.Single(_api.PredictionHistoryCalls));
        Assert.Equal("So1abc", page.Find(".solia-filter-mint input").GetAttribute("value"));
        Assert.Equal("2026-10-01T00:00:00", page.Find(".solia-filter-from input").GetAttribute("value"));
        page.WaitForAssertion(() => Assert.Equal("Page 3 of 4", page.Find(".solia-pager__page").TextContent.Trim()));
    }

    [Fact]
    public void Applying_filters_updates_the_url_back_to_page_one_and_queries_the_backend()
    {
        Navigation.NavigateTo("predictions/history?page=4");
        Returns(total: 500);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.Find(".solia-filter-mint input").Input("  So1abc ");
        page.InvokeAsync(() => page.FindComponent<MudSelect<PredictionHorizon?>>().Instance.ValueChanged.InvokeAsync(PredictionHorizon.Minutes5));
        page.InvokeAsync(() => page.FindComponent<MudSelect<TradeStrategy?>>().Instance.ValueChanged.InvokeAsync(TradeStrategy.Buy));
        page.Find(".solia-filter-from input").Input("2026-10-01T00:00");
        page.Find(".solia-filter-to input").Input("2026-10-02T12:30");
        Assert.Single(_api.PredictionHistoryCalls); // editing alone doesn't query
        page.Find("form.solia-filter-bar").Submit();

        page.WaitForAssertion(() => Assert.Equal(
            "?mint=So1abc&horizon=5m&strategy=buy&from=2026-10-01T00%3A00%3A00Z&to=2026-10-02T12%3A30%3A00Z", Query));
        page.WaitForAssertion(() => Assert.Equal(2, _api.PredictionHistoryCalls.Count));
        Assert.Equal(new PredictionHistoryQuery
        {
            Mint = "So1abc",
            Horizon = PredictionHorizon.Minutes5,
            Strategy = TradeStrategy.Buy,
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            To = new DateTimeOffset(2026, 10, 2, 12, 30, 0, TimeSpan.Zero),
            Page = 1,
            PageSize = 50,
        }, _api.PredictionHistoryCalls[^1]);
    }

    [Fact]
    public void Reset_clears_every_filter_from_the_url()
    {
        Navigation.NavigateTo("predictions/history?mint=So1abc&horizon=1h&page=2");
        Returns(total: 500);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.Find("button.solia-filter-reset").Click();

        Assert.Equal("", Query);
        page.WaitForAssertion(() => Assert.Equal(new PredictionHistoryQuery { Page = 1, PageSize = 50 }, _api.PredictionHistoryCalls[^1]));
        page.WaitForAssertion(() => Assert.Null(page.Find(".solia-filter-mint input").GetAttribute("value")));
    }

    [Fact]
    public void Paging_updates_the_url_and_queries_the_next_page()
    {
        Returns(total: 120, rows: 50);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal("1–50 of 120", page.Find(".solia-pager__range").TextContent.Trim()));

        PagerButton(page, "next").Click();

        Assert.Equal("?page=2", Query);
        page.WaitForAssertion(() => Assert.Equal(2, _api.PredictionHistoryCalls[^1].Page));
        page.WaitForAssertion(() => Assert.Equal("51–100 of 120", page.Find(".solia-pager__range").TextContent.Trim()));

        PagerButton(page, "last").Click();
        Assert.Equal("?page=3", Query);
        PagerButton(page, "first").Click();
        Assert.Equal("", Query);
    }

    [Fact]
    public void Changing_the_page_size_goes_back_to_page_one()
    {
        Navigation.NavigateTo("predictions/history?page=2");
        Returns(total: 500);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.InvokeAsync(() => page.FindComponent<MudSelect<int>>().Instance.ValueChanged.InvokeAsync(100));

        Assert.Equal("?pageSize=100", Query);
        page.WaitForAssertion(() => Assert.Equal(new PredictionHistoryQuery { Page = 1, PageSize = 100 }, _api.PredictionHistoryCalls[^1]));
    }

    [Fact]
    public void Browser_navigation_reloads_the_view_and_the_filter_bar_follows()
    {
        Returns(total: 10);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        Navigation.NavigateTo("predictions/history?mint=So1back");

        page.WaitForAssertion(() => Assert.Equal("So1back", _api.PredictionHistoryCalls[^1].Mint));
        page.WaitForAssertion(() => Assert.Equal("So1back", page.Find(".solia-filter-mint input").GetAttribute("value")));
    }

    [Fact]
    public void On_the_first_page_first_and_previous_are_disabled()
    {
        Returns(total: 120);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.True(PagerButton(page, "first").HasAttribute("disabled")));
        Assert.True(PagerButton(page, "previous").HasAttribute("disabled"));
        Assert.False(PagerButton(page, "next").HasAttribute("disabled"));
        Assert.False(PagerButton(page, "last").HasAttribute("disabled"));
    }

    [Fact]
    public void On_the_last_page_next_and_last_are_disabled()
    {
        Navigation.NavigateTo("predictions/history?page=3");
        Returns(total: 120);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.True(PagerButton(page, "next").HasAttribute("disabled")));
        Assert.True(PagerButton(page, "last").HasAttribute("disabled"));
        Assert.False(PagerButton(page, "previous").HasAttribute("disabled"));
        Assert.Equal("Page 3 of 3", page.Find(".solia-pager__page").TextContent.Trim());
    }

    [Fact]
    public void A_single_page_disables_the_whole_pager()
    {
        Returns(total: 1);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal("Page 1 of 1", page.Find(".solia-pager__page").TextContent.Trim()));
        Assert.All(["first", "previous", "next", "last"], b => Assert.True(PagerButton(page, b).HasAttribute("disabled")));
    }

    [Fact]
    public void The_pager_stops_at_the_api_depth_cap_and_says_so()
    {
        Returns(total: 50_000);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal("Page 1 of 200", page.Find(".solia-pager__page").TextContent.Trim()));
        Assert.Contains("10,000", page.Find(".solia-pager__cap").TextContent);

        PagerButton(page, "last").Click();

        Assert.Equal("?page=200", Query);
        page.WaitForAssertion(() => Assert.True(PagerButton(page, "next").HasAttribute("disabled")));
    }

    [Fact]
    public void An_invalid_request_is_shown_as_an_inline_validation_message()
    {
        Navigation.NavigateTo("predictions/history?from=2026-10-02T00:00:00Z&to=2026-10-01T00:00:00Z");
        _api.PredictionHistory = (_, _) => Task.FromException<PagedResult<PredictionRecordDto>>(
            new BackendErrorException(400, "invalid_request", "'from' must not be after 'to'."));

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal(
            "Check the filters: 'from' must not be after 'to'.", page.Find(".solia-invalid").TextContent.Trim()));
        Assert.Empty(page.FindAll(".solia-error"));
        Assert.Empty(page.FindAll("table"));
        Assert.NotNull(page.Find("form.solia-filter-bar")); // still there to fix the filters
    }

    [Fact]
    public void No_match_shows_the_empty_state()
    {
        Returns(total: 0, rows: 0);

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Contains("No predictions match these filters.", page.Find(".solia-empty").TextContent));
        Assert.Empty(page.FindAll("table"));
    }

    [Fact]
    public void A_page_past_the_end_says_so_and_keeps_the_pager_to_go_back()
    {
        Navigation.NavigateTo("predictions/history?page=9");
        Returns(total: 120, rows: 0);

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-past-end")));
        Assert.Equal("0 of 120", page.Find(".solia-pager__range").TextContent.Trim());
        PagerButton(page, "first").Click();
        Assert.Equal("", Query);
    }

    [Fact]
    public void A_backend_failure_shows_the_error_state_with_a_retry()
    {
        int calls = 0;
        _api.PredictionHistory = (q, _) => ++calls == 1
            ? Task.FromException<PagedResult<PredictionRecordDto>>(new BackendUnavailableException())
            : Task.FromResult(new PagedResult<PredictionRecordDto>([Prediction(1)], 1, 50, 1));

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Contains("Could not load the prediction history", page.Find(".solia-error").TextContent));
        Assert.Empty(page.FindAll(".solia-invalid"));

        page.Find("button.solia-error__retry").Click();

        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr")));
    }

    [Fact]
    public void Shows_the_loading_state_until_the_page_arrives()
    {
        var pending = new TaskCompletionSource<PagedResult<PredictionRecordDto>>();
        _api.PredictionHistory = (_, _) => pending.Task;

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        Assert.NotNull(page.Find(".solia-loading"));
        pending.SetResult(new PagedResult<PredictionRecordDto>([Prediction(1)], 1, 50, 1));
        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr")));
    }

    [Fact]
    public async Task Leaving_the_page_stops_following_the_url()
    {
        Returns(total: 1);
        RenderLoaded();

        await DisposeComponentsAsync();
        Navigation.NavigateTo("predictions/history?mint=after");

        Assert.Single(_api.PredictionHistoryCalls);
    }

    [Fact]
    public void A_typed_mint_is_applied_on_enter_without_leaving_the_field()
    {
        Returns(total: 10);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        // Immediate: each keystroke reaches the draft, so Enter (a submit straight after the input) applies it.
        page.Find(".solia-filter-mint input").Input("So1typed");
        page.Find("form.solia-filter-bar").Submit();

        page.WaitForAssertion(() => Assert.Equal("?mint=So1typed", Query));
        page.WaitForAssertion(() => Assert.Equal("So1typed", _api.PredictionHistoryCalls[^1].Mint));
    }

    [Fact]
    public void High_probabilities_are_emphasised()
    {
        _api.PredictionHistory = (_, _) => Task.FromResult(new PagedResult<PredictionRecordDto>(
            [Prediction(1) with { PumpProb = 0.7, RugProb = 0.49 }, Prediction(2) with { PumpProb = 0.69, RugProb = 0.5 }], 1, 50, 2));

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal(2, page.FindAll("tbody tr").Count));
        IElement high = page.Find("tr[data-id='1']"), low = page.Find("tr[data-id='2']");
        Assert.True(high.QuerySelector("td[data-col=pump]")!.ClassList.Contains("solia-high-pump"));
        Assert.False(high.QuerySelector("td[data-col=rug]")!.ClassList.Contains("solia-high-rug"));
        Assert.False(low.QuerySelector("td[data-col=pump]")!.ClassList.Contains("solia-high-pump"));
        Assert.True(low.QuerySelector("td[data-col=rug]")!.ClassList.Contains("solia-high-rug"));
    }

    [Fact]
    public void While_the_next_page_loads_the_pager_still_describes_the_rows_on_screen()
    {
        var next = new TaskCompletionSource<PagedResult<PredictionRecordDto>>();
        _api.PredictionHistory = (q, _) => q.Page == 1
            ? Task.FromResult(new PagedResult<PredictionRecordDto>(Enumerable.Range(1, 50).Select(i => Prediction(i)).ToArray(), 1, 50, 120))
            : next.Task;
        IRenderedComponent<PredictionHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal("Page 1 of 3", page.Find(".solia-pager__page").TextContent.Trim()));
        Assert.Empty(page.FindAll(".solia-refreshing"));

        PagerButton(page, "next").Click();

        page.WaitForAssertion(() => Assert.Equal(2, _api.PredictionHistoryCalls[^1].Page));
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-refreshing")));
        Assert.Equal("Page 1 of 3", page.Find(".solia-pager__page").TextContent.Trim());
        Assert.Equal("1–50 of 120", page.Find(".solia-pager__range").TextContent.Trim());

        next.SetResult(new PagedResult<PredictionRecordDto>(Enumerable.Range(51, 50).Select(i => Prediction(i)).ToArray(), 2, 50, 120));

        page.WaitForAssertion(() => Assert.Equal("Page 2 of 3", page.Find(".solia-pager__page").TextContent.Trim()));
        Assert.Equal("51–100 of 120", page.Find(".solia-pager__range").TextContent.Trim());
        Assert.Empty(page.FindAll(".solia-refreshing"));
    }

    [Fact]
    public void A_page_too_deep_offers_a_way_back_to_the_first_page()
    {
        Navigation.NavigateTo("predictions/history?mint=So1abc&page=300");
        _api.PredictionHistory = (q, _) => q.Page > 200
            ? Task.FromException<PagedResult<PredictionRecordDto>>(new BackendErrorException(400, "invalid_request", "Page too deep."))
            : Task.FromResult(new PagedResult<PredictionRecordDto>([Prediction(1)], 1, 50, 1));
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.NotNull(page.Find("button.solia-invalid__first-page")));
        page.Find("button.solia-invalid__first-page").Click();

        Assert.Equal("?mint=So1abc", Query); // the filters stay
        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr")));
    }

    [Fact]
    public void A_rejected_page_size_goes_back_to_the_default_one()
    {
        Navigation.NavigateTo("predictions/history?page=2&pageSize=100");
        _api.PredictionHistory = (q, _) => q.PageSize == 100
            ? Task.FromException<PagedResult<PredictionRecordDto>>(new BackendErrorException(400, "invalid_request", "No."))
            : Task.FromResult(new PagedResult<PredictionRecordDto>([Prediction(1)], 1, 50, 1));
        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.NotNull(page.Find("button.solia-invalid__first-page")));
        page.Find("button.solia-invalid__first-page").Click();

        Assert.Equal("", Query);
        page.WaitForAssertion(() => Assert.Equal(new PredictionHistoryQuery { Page = 1, PageSize = 50 }, _api.PredictionHistoryCalls[^1]));
    }

    [Fact]
    public void An_invalid_filter_on_the_first_page_offers_no_paging_way_out()
    {
        Navigation.NavigateTo("predictions/history?from=2026-10-02T00:00:00Z&to=2026-10-01T00:00:00Z");
        _api.PredictionHistory = (_, _) => Task.FromException<PagedResult<PredictionRecordDto>>(
            new BackendErrorException(400, "invalid_request", "'from' must not be after 'to'."));

        IRenderedComponent<PredictionHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-invalid")));
        Assert.Empty(page.FindAll("button.solia-invalid__first-page"));
    }

    [Fact]
    public void An_oversized_page_size_in_the_url_is_capped_to_the_api_maximum()
    {
        Navigation.NavigateTo("predictions/history?pageSize=500");
        Returns(total: 1);

        RenderLoaded();

        Assert.Equal(200, Assert.Single(_api.PredictionHistoryCalls).PageSize);
    }

    [Fact]
    public void Navigating_to_another_page_does_not_reload_the_history()
    {
        Returns(total: 1);
        IRenderedComponent<PredictionHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(page.FindAll("tbody tr")));

        Navigation.NavigateTo("trades/history?page=3&from=2026-10-01T00:00:00Z");

        Assert.Single(_api.PredictionHistoryCalls);
        Assert.Equal("Page 1 of 1", page.Find(".solia-pager__page").TextContent.Trim());
    }
}
