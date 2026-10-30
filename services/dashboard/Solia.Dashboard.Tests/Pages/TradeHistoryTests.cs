using AngleSharp.Dom;
using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.DependencyInjection;
using MudBlazor;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Tests.Api;
using Solia.Dashboard.Tests.Live;
using Solia.Dashboard.Trades;

namespace Solia.Dashboard.Tests.Pages;

public class TradeHistoryTests : MudTestContext
{
    private const string Mint = "So1aNaMintAddress1111111111111111111111xyz9";

    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 11, 58, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();
    private readonly FakeLiveFeed _feed = new();

    public TradeHistoryTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Services.AddSingleton<ILiveFeed>(_feed);
        Navigation.NavigateTo("trades");
    }

    private NavigationManager Navigation => Services.GetRequiredService<NavigationManager>();

    private static TradeDto Opened(long id, long? predictionId = 42) =>
        new(id, Mint, predictionId, T0.AddMinutes(id), null, 0.000000123400m, null, 0.5m, null, null, null, "buy", 74.4, null, "open");

    private static TradeDto Closed(long id, decimal pnl, double pnlPct, string reason = "tp") =>
        Opened(id) with
        {
            ClosedAt = T0.AddMinutes(id).AddSeconds(187),
            ExitPrice = 0.0000001500m,
            DurationSec = 187,
            Pnl = pnl,
            PnlPct = pnlPct,
            ExitReason = reason,
            Status = "closed",
        };

    /// <summary>Answers every request with <paramref name="rows"/> (newest first) out of <paramref name="total"/>.</summary>
    private void Returns(long total, params TradeDto[] rows) =>
        _api.TradeHistory = (q, _) => Task.FromResult(new PagedResult<TradeDto>(rows, q.Page ?? 1, q.PageSize ?? 50, total));

    private IRenderedComponent<TradeHistory> RenderLoaded()
    {
        Render<MudPopoverProvider>();
        IRenderedComponent<TradeHistory> page = Render<TradeHistory>();
        page.WaitForAssertion(() => Assert.NotEmpty(_api.TradeHistoryCalls));
        return page;
    }

    private static IReadOnlyList<IElement> Rows(IRenderedComponent<TradeHistory> page) => page.FindAll("tbody tr");

    private static long[] Ids(IRenderedComponent<TradeHistory> page) =>
        [.. Rows(page).Select(r => long.Parse(r.GetAttribute("data-id")!))];

    private static string Cell(IElement row, string column) =>
        row.QuerySelector($"td[data-col={column}]")!.TextContent.Trim();

    private string Query => new Uri(Navigation.Uri).Query;

    [Fact]
    public async Task A_resync_while_the_backend_is_down_keeps_the_live_page_marked_stale()
    {
        Returns(total: 1, Closed(7, pnl: 0.0123m, pnlPct: 0.2156));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal([7L], Ids(page)));
        _api.TradeHistory = (_, _) => Task.FromException<PagedResult<TradeDto>>(new BackendUnavailableException());

        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-stale")));
        Assert.Equal([7L], Ids(page));

        Returns(total: 2, Opened(8), Closed(7, pnl: 0.0123m, pnlPct: 0.2156));
        page.Find(".solia-stale button.solia-stale__retry").Click();

        page.WaitForAssertion(() => Assert.Equal([8L, 7L], Ids(page)));
        Assert.Empty(page.FindAll(".solia-stale"));
    }

    [Fact]
    public void Renders_a_closed_trade_with_exact_prices_and_signed_pnl()
    {
        Returns(total: 1, Closed(7, pnl: 0.0123m, pnlPct: 0.2156));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        IElement row = Rows(page)[0];
        Assert.Equal("7", Cell(row, "id"));
        Assert.Equal("So1a…xyz9", Cell(row, "token"));
        Assert.Equal(Mint, row.QuerySelector(".solia-token")!.GetAttribute("title"));
        Assert.Equal("2026-10-27 12:05:00", Cell(row, "opened"));
        Assert.Equal("2026-10-27 12:08:07", Cell(row, "closed"));
        Assert.Equal("0.0000001234", Cell(row, "entry"));
        Assert.Equal("0.00000015", Cell(row, "exit"));
        Assert.Equal("0.5", Cell(row, "size"));
        Assert.Equal("3m 07s", Cell(row, "duration"));
        Assert.Equal("+0.0123", Cell(row, "pnl"));
        Assert.Equal("+21.6%", Cell(row, "pnl-pct"));
        Assert.Contains("solia-pnl--gain", row.QuerySelector("td[data-col=pnl]")!.ClassList);
        Assert.Equal("BUY", Cell(row, "strategy"));
        Assert.Equal("74", Cell(row, "confidence"));
        Assert.Equal("Take profit", Cell(row, "exit-reason"));
        Assert.Equal("closed", Cell(row, "status"));
        Assert.Equal(new TradeHistoryQuery { Page = 1, PageSize = 50 }, Assert.Single(_api.TradeHistoryCalls));
    }

    [Fact]
    public void A_loss_is_negative_and_marked_as_such()
    {
        Returns(total: 1, Closed(7, pnl: -0.0000000045m, pnlPct: -0.034, reason: "sl"));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        IElement row = Rows(page)[0];
        Assert.Equal("-0.0000000045", Cell(row, "pnl"));
        Assert.Equal("-3.4%", Cell(row, "pnl-pct"));
        Assert.Contains("solia-pnl--loss", row.QuerySelector("td[data-col=pnl]")!.ClassList);
        Assert.Contains("solia-pnl--loss", row.QuerySelector("td[data-col=pnl-pct]")!.ClassList);
        Assert.Equal("Stop loss", Cell(row, "exit-reason"));
    }

    [Fact]
    public void An_open_trade_shows_placeholders_for_what_is_not_known_yet()
    {
        Returns(total: 1, Opened(7));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        IElement row = Rows(page)[0];
        foreach (string column in new[] { "closed", "exit", "duration", "pnl", "pnl-pct", "exit-reason" })
        {
            Assert.Equal("—", Cell(row, column));
        }
        Assert.Empty(row.QuerySelectorAll("td[data-col=pnl].solia-pnl--gain, td[data-col=pnl].solia-pnl--loss"));
        Assert.Equal("open", Cell(row, "status"));
    }

    [Fact]
    public void A_trade_links_to_its_prediction_context_only_when_it_has_one()
    {
        Returns(total: 2, Opened(8, predictionId: 42), Opened(7, predictionId: null));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        IElement link = Rows(page)[0].QuerySelector("a.solia-prediction-link")!;
        Assert.Equal($"predictions/history?mint={Mint}", link.GetAttribute("href"));
        Assert.Equal("#42", link.TextContent.Trim());
        Assert.Null(Rows(page)[1].QuerySelector("a.solia-prediction-link"));
        Assert.Equal("—", Cell(Rows(page)[1], "prediction"));
    }

    [Fact]
    public void Filters_and_page_are_read_from_the_url()
    {
        Navigation.NavigateTo("trades?status=closed&from=2026-10-01T00:00:00Z&to=2026-10-02T00:00:00Z&page=2&pageSize=25");
        Returns(total: 100, Opened(1));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        Assert.Equal(new TradeHistoryQuery
        {
            Status = TradeStatus.Closed,
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            To = new DateTimeOffset(2026, 10, 2, 0, 0, 0, TimeSpan.Zero),
            Page = 2,
            PageSize = 25,
        }, Assert.Single(_api.TradeHistoryCalls));
        Assert.Equal("2026-10-01T00:00:00", page.Find(".solia-filter-from input").GetAttribute("value"));
        page.WaitForAssertion(() => Assert.Equal("Page 2 of 4", page.Find(".solia-pager__page").TextContent.Trim()));
    }

    [Fact]
    public void Applying_filters_updates_the_url_back_to_page_one_and_queries_the_backend()
    {
        Navigation.NavigateTo("trades?page=3");
        Returns(total: 500, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.InvokeAsync(() => page.FindComponent<MudSelect<TradeStatus?>>().Instance.ValueChanged.InvokeAsync(TradeStatus.Open));
        page.Find(".solia-filter-from input").Input("2026-10-01T00:00");
        page.Find(".solia-filter-to input").Input("2026-10-02T12:30");
        Assert.Single(_api.TradeHistoryCalls); // editing alone doesn't query
        page.Find("form.solia-filter-bar").Submit();

        page.WaitForAssertion(() => Assert.Equal(
            "?status=open&from=2026-10-01T00%3A00%3A00Z&to=2026-10-02T12%3A30%3A00Z", Query));
        page.WaitForAssertion(() => Assert.Equal(2, _api.TradeHistoryCalls.Count));
        Assert.Equal(new TradeHistoryQuery
        {
            Status = TradeStatus.Open,
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            To = new DateTimeOffset(2026, 10, 2, 12, 30, 0, TimeSpan.Zero),
            Page = 1,
            PageSize = 50,
        }, _api.TradeHistoryCalls[^1]);
    }

    [Fact]
    public void Picking_a_session_puts_it_in_the_url_back_to_page_one_and_keeps_it_across_filters()
    {
        Navigation.NavigateTo("trades?status=closed&page=3");
        Returns(total: 500, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.InvokeAsync(() => page.FindComponent<MudSelect<SessionSelection>>().Instance.ValueChanged.InvokeAsync(SessionSelection.Of(2)));

        page.WaitForAssertion(() => Assert.Equal("?status=closed&session=2", Query));
        page.WaitForAssertion(() => Assert.Equal(
            new TradeHistoryQuery { Status = TradeStatus.Closed, Session = SessionSelection.Of(2), Page = 1, PageSize = 50 },
            _api.TradeHistoryCalls[^1]));

        page.Find("button.solia-filter-reset").Click();
        Assert.Equal("?session=2", Query);

        // Back to the current session: the default, so it leaves the URL.
        page.InvokeAsync(() => page.FindComponent<MudSelect<SessionSelection>>().Instance.ValueChanged.InvokeAsync(SessionSelection.Current));
        page.WaitForAssertion(() => Assert.Equal("", Query));
    }

    [Fact]
    public void Reset_clears_every_filter_from_the_url()
    {
        Navigation.NavigateTo("trades?status=closed&page=2");
        Returns(total: 500, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.Find("button.solia-filter-reset").Click();

        Assert.Equal("", Query);
        page.WaitForAssertion(() => Assert.Equal(new TradeHistoryQuery { Page = 1, PageSize = 50 }, _api.TradeHistoryCalls[^1]));
    }

    [Fact]
    public void Paging_updates_the_url_and_queries_the_next_page()
    {
        Returns(total: 120, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        page.Find("button[data-pager=next]").Click();

        Assert.Equal("?page=2", Query);
        page.WaitForAssertion(() => Assert.Equal(2, _api.TradeHistoryCalls[^1].Page));
    }

    [Fact]
    public async Task On_the_live_page_an_opened_trade_is_prepended_and_its_close_updates_the_row_in_place()
    {
        Returns(total: 2, Opened(2), Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(3));

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
        Assert.Equal("open", Cell(Rows(page)[0], "status"));
        Assert.Equal("1–3 of 3", page.Find(".solia-pager__range").TextContent.Trim());

        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(3, pnl: -0.01m, pnlPct: -0.02));

        page.WaitForAssertion(() => Assert.Equal("closed", Cell(Rows(page)[0], "status")));
        Assert.Equal([3, 2, 1], Ids(page));
        Assert.Equal("-0.01", Cell(Rows(page)[0], "pnl"));

        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(1, pnl: 0.5m, pnlPct: 1.0));

        page.WaitForAssertion(() => Assert.Equal("closed", Cell(Rows(page)[2], "status")));
        Assert.Equal("+0.5", Cell(Rows(page)[2], "pnl"));
        Assert.Single(_api.TradeHistoryCalls); // applied live, not re-fetched
        Assert.Empty(page.FindAll(".solia-new-trades"));
    }

    [Fact]
    public async Task On_a_filtered_page_live_trades_only_raise_a_refresh_hint()
    {
        Navigation.NavigateTo("trades?status=open");
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(2));
        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(1, pnl: 1m, pnlPct: 1));

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-new-trades")));
        Assert.Equal([1], Ids(page));
        Assert.Equal("open", Cell(Rows(page)[0], "status")); // not mutated

        Returns(total: 1, Opened(2));
        page.Find("button.solia-new-trades__refresh").Click();

        page.WaitForAssertion(() => Assert.Equal([2], Ids(page)));
        Assert.Empty(page.FindAll(".solia-new-trades"));
        Assert.Equal(2, _api.TradeHistoryCalls.Count);
    }

    [Fact]
    public async Task On_a_later_page_live_trades_only_raise_a_refresh_hint()
    {
        Navigation.NavigateTo("trades?page=2");
        Returns(total: 60, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(99));

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-new-trades")));
        Assert.Equal([1], Ids(page));
    }

    [Fact]
    public async Task Leaving_for_the_live_page_drops_the_hint()
    {
        Navigation.NavigateTo("trades?status=closed");
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(2));
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-new-trades")));

        page.Find("button.solia-filter-reset").Click();

        page.WaitForAssertion(() => Assert.Empty(page.FindAll(".solia-new-trades")));
    }

    [Fact]
    public async Task A_trade_opened_on_an_empty_live_page_replaces_the_empty_state_without_a_refetch()
    {
        Returns(total: 0);
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-empty")));

        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(1)); // the backend may not show it yet

        page.WaitForAssertion(() => Assert.Equal([1], Ids(page)));
        Assert.Empty(page.FindAll(".solia-empty"));
        Assert.Equal("1–1 of 1", page.Find(".solia-pager__range").TextContent.Trim());
        Assert.Single(_api.TradeHistoryCalls);
    }

    [Fact]
    public async Task A_trade_closed_on_an_empty_live_page_keeps_the_empty_state()
    {
        Returns(total: 0);
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-empty")));

        await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(1, pnl: 1m, pnlPct: 1));

        Assert.NotNull(page.Find(".solia-empty"));
    }

    [Fact]
    public async Task A_trade_pushed_while_the_first_page_loads_goes_on_top_of_it()
    {
        var pending = new TaskCompletionSource<PagedResult<TradeDto>>();
        _api.TradeHistory = (_, _) => pending.Task;
        IRenderedComponent<TradeHistory> page = RenderLoaded();

        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(3));
        pending.SetResult(new PagedResult<TradeDto>([Opened(2), Opened(1)], 1, 50, 2));

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
    }

    [Fact]
    public async Task Going_back_to_the_live_page_lays_nothing_over_the_page_still_on_screen()
    {
        Navigation.NavigateTo("trades?page=2");
        Returns(total: 60, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        var pending = new TaskCompletionSource<PagedResult<TradeDto>>();
        _api.TradeHistory = (_, _) => pending.Task;
        page.Find("button[data-pager=first]").Click();
        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(99));

        Assert.Equal([1], Ids(page)); // still page 2's rows, untouched
        Assert.Equal("51–51 of 60", page.Find(".solia-pager__range").TextContent.Trim());

        pending.SetResult(new PagedResult<TradeDto>([Opened(51)], 1, 50, 60));

        page.WaitForAssertion(() => Assert.Equal([99, 51], Ids(page))); // pushed while page 1 was in flight
    }

    [Fact]
    public async Task A_superseded_fetch_does_not_drop_the_trades_laid_over_the_page()
    {
        Returns(total: 2, Opened(2), Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        await _feed.PushAsync(LiveFeedEvents.TradeOpened, Opened(3));
        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));

        var first = new TaskCompletionSource<PagedResult<TradeDto>>();
        var second = new TaskCompletionSource<PagedResult<TradeDto>>();
        var fetches = new Queue<TaskCompletionSource<PagedResult<TradeDto>>>([first, second]);
        _api.TradeHistory = (_, _) => fetches.Dequeue().Task;
        // A resync completes with its re-fetch, so neither is awaited until the fetches are answered.
        Task firstResync = _feed.ResyncAsync();
        Task secondResync = _feed.ResyncAsync(); // supersedes the first re-fetch
        page.WaitForAssertion(() => Assert.Equal(3, _api.TradeHistoryCalls.Count));

        first.SetResult(new PagedResult<TradeDto>([Opened(2), Opened(1)], 1, 50, 2)); // stale: discarded
        await firstResync;
        Assert.Equal([3, 2, 1], Ids(page));

        second.SetResult(new PagedResult<TradeDto>([Opened(3), Opened(2), Opened(1)], 1, 50, 3));
        await secondResync;
        page.WaitForAssertion(() => Assert.Equal("1–3 of 3", page.Find(".solia-pager__range").TextContent.Trim()));
        Assert.Equal([3, 2, 1], Ids(page));
    }

    [Fact]
    public async Task A_live_page_left_open_refetches_once_the_pushes_pile_up()
    {
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        // Closes of trades not on the page are held (they may show up after a re-fetch) without being shown.
        for (long id = 1000; id < 1000 + LiveTradeOverlay.DefaultCapacity; id++)
        {
            await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(id, pnl: 0m, pnlPct: 0));
        }

        page.WaitForAssertion(() => Assert.Equal(2, _api.TradeHistoryCalls.Count));
        Assert.Equal([1], Ids(page));
    }

    [Fact]
    public async Task A_failed_catch_up_is_not_retried_on_every_push()
    {
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        _api.TradeHistory = (_, _) => Task.FromException<PagedResult<TradeDto>>(new BackendUnavailableException());

        for (long id = 1000; id < 1000 + LiveTradeOverlay.DefaultCapacity + 10; id++)
        {
            await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(id, pnl: 0m, pnlPct: 0));
        }

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-stale")));
        Assert.Equal(2, _api.TradeHistoryCalls.Count); // the first load and one catch-up, not one per push

        // Once a fetch succeeds again, a page that piles up pushes catches up again.
        Returns(total: 1, Opened(1));
        page.Find(".solia-stale button.solia-stale__retry").Click();
        page.WaitForAssertion(() => Assert.Empty(page.FindAll(".solia-stale")));
        for (long id = 2000; id < 2000 + LiveTradeOverlay.DefaultCapacity; id++)
        {
            await _feed.PushAsync(LiveFeedEvents.TradeClosed, Closed(id, pnl: 0m, pnlPct: 0));
        }
        page.WaitForAssertion(() => Assert.Equal(4, _api.TradeHistoryCalls.Count));
    }

    [Fact]
    public async Task A_resync_refetches_the_live_page()
    {
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        Returns(total: 2, Opened(2), Opened(1));
        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.Equal([2, 1], Ids(page)));
        Assert.Equal(2, _api.TradeHistoryCalls.Count);
    }

    [Fact]
    public async Task A_resync_on_a_filtered_page_raises_the_hint_instead()
    {
        Navigation.NavigateTo("trades?status=closed");
        Returns(total: 1, Opened(1));
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await _feed.ResyncAsync();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-new-trades")));
        Assert.Single(_api.TradeHistoryCalls);
    }

    [Fact]
    public void An_invalid_request_is_shown_as_an_inline_validation_message()
    {
        Navigation.NavigateTo("trades?from=2026-10-02T00:00:00Z&to=2026-10-01T00:00:00Z");
        _api.TradeHistory = (_, _) => Task.FromException<PagedResult<TradeDto>>(
            new BackendErrorException(400, "invalid_request", "'from' must not be after 'to'."));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal(
            "Check the filters: 'from' must not be after 'to'.", page.Find(".solia-invalid").TextContent.Trim()));
        Assert.Empty(page.FindAll("table"));
        Assert.NotNull(page.Find("form.solia-filter-bar"));
    }

    [Fact]
    public void No_match_shows_the_empty_state()
    {
        Returns(total: 0);

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Contains("No trades match these filters.", page.Find(".solia-empty").TextContent));
        Assert.Empty(page.FindAll("table"));
    }

    [Fact]
    public void A_backend_failure_shows_the_error_state_with_a_retry()
    {
        int calls = 0;
        _api.TradeHistory = (_, _) => ++calls == 1
            ? Task.FromException<PagedResult<TradeDto>>(new BackendUnavailableException())
            : Task.FromResult(new PagedResult<TradeDto>([Opened(1)], 1, 50, 1));

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Contains("Could not load the trade history", page.Find(".solia-error").TextContent));

        page.Find("button.solia-error__retry").Click();

        page.WaitForAssertion(() => Assert.Single(Rows(page)));
    }

    [Fact]
    public void Shows_the_loading_state_until_the_page_arrives()
    {
        var pending = new TaskCompletionSource<PagedResult<TradeDto>>();
        _api.TradeHistory = (_, _) => pending.Task;

        IRenderedComponent<TradeHistory> page = RenderLoaded();

        Assert.NotNull(page.Find(".solia-loading"));
        pending.SetResult(new PagedResult<TradeDto>([Opened(1)], 1, 50, 1));
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
    }

    [Fact]
    public async Task Leaving_the_page_stops_listening_to_the_feed()
    {
        IRenderedComponent<TradeHistory> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(1, _feed.Subscribers(LiveFeedEvents.TradeOpened)));

        await DisposeComponentsAsync();

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.TradeOpened));
        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.TradeClosed));
        Assert.Equal(0, _feed.ResyncSubscribers);
    }
}
