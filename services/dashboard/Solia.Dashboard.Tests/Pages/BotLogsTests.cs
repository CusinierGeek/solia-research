using System.Text.Json;
using AngleSharp.Dom;
using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.DependencyInjection;
using MudBlazor;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Pages;
using Solia.Dashboard.Live;
using Solia.Dashboard.Logs;
using Solia.Dashboard.Tests.Api;
using Solia.Dashboard.Tests.Live;

namespace Solia.Dashboard.Tests.Pages;

public class BotLogsTests : MudTestContext
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();
    private readonly FakeLiveFeed _feed = new();
    private readonly ManualClock _clock = new(T0);

    public BotLogsTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        Services.AddSingleton<ILiveFeed>(_feed);
        Services.AddSingleton<TimeProvider>(_clock);
        Navigation.NavigateTo("logs");
    }

    private NavigationManager Navigation => Services.GetRequiredService<NavigationManager>();

    private static LogDto Log(long id, string level = "info", string service = "bot", string? message = null, string? context = null) =>
        new(id, T0.AddSeconds(id), service, level, message ?? $"entry {id}",
            context is null ? null : JsonDocument.Parse(context).RootElement.Clone());

    private void Returns(params LogDto[] rows) => _api.BotLogs = (_, _) => Task.FromResult<IReadOnlyList<LogDto>>(rows);

    private IRenderedComponent<BotLogs> RenderLoaded(string uri = "logs")
    {
        Navigation.NavigateTo(uri);
        Render<MudPopoverProvider>();
        IRenderedComponent<BotLogs> page = Render<BotLogs>();
        page.WaitForAssertion(() => Assert.NotEmpty(_api.BotLogsCalls));
        return page;
    }

    private static IReadOnlyList<IElement> Rows(IRenderedComponent<BotLogs> page) => page.FindAll("tbody tr[data-id]");

    private static long[] Ids(IRenderedComponent<BotLogs> page) =>
        [.. Rows(page).Select(r => long.Parse(r.GetAttribute("data-id")!))];

    private static string Cell(IElement row, string column) =>
        row.QuerySelector($"td[data-col={column}]")!.TextContent.Trim();

    // Pushes, then lets the batched render happen.
    private async Task PushAsync(IRenderedComponent<BotLogs> page, LogDto log)
    {
        await PushUnrenderedAsync(page, log);
        _clock.Elapse(BotLogs.RenderInterval);
    }

    private Task PushUnrenderedAsync(IRenderedComponent<BotLogs> page, LogDto log) =>
        page.InvokeAsync(() => _feed.PushAsync(LiveFeedEvents.LogWritten, log));

    private TaskCompletionSource<IReadOnlyList<LogDto>> HoldNextFetch()
    {
        var pending = new TaskCompletionSource<IReadOnlyList<LogDto>>();
        _api.BotLogs = (_, _) => pending.Task;
        return pending;
    }

    [Fact]
    public void Renders_an_entry_with_its_time_service_and_colored_level()
    {
        Returns(Log(2, "error", "ai-service", "model failed"), Log(1, "warn", "data-pipeline"));

        IRenderedComponent<BotLogs> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        IElement row = Rows(page)[0];
        Assert.Equal("2026-10-27 12:00:02", Cell(row, "ts"));
        Assert.Equal("ai-service", Cell(row, "service"));
        Assert.Equal("error", Cell(row, "level"));
        Assert.Equal("model failed", Cell(row, "message"));
        Assert.Contains("mud-chip-color-error", row.QuerySelector("td[data-col=level] .mud-chip")!.ClassName);
        Assert.Contains("mud-chip-color-warning", Rows(page)[1].QuerySelector("td[data-col=level] .mud-chip")!.ClassName);
        Assert.Equal(new BotLogsQuery(), Assert.Single(_api.BotLogsCalls));
    }

    [Fact]
    public void Messages_and_context_are_rendered_as_text_never_as_markup()
    {
        Returns(Log(1, message: "<script>alert(1)</script><b>bold</b>", context: """{"html":"<img src=x onerror=alert(1)>"}"""));

        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        page.Find("button.solia-log__expand").Click();

        Assert.Empty(page.FindAll("tbody script, tbody b, tbody img"));
        Assert.Equal("<script>alert(1)</script><b>bold</b>", Cell(Rows(page)[0], "message"));
        Assert.Contains("<img src=x onerror=alert(1)>", page.Find("pre.solia-log__context").TextContent);
    }

    [Fact]
    public void The_context_json_is_shown_on_expand_and_hidden_again()
    {
        Returns(Log(1, context: """{"mint":"abc","attempt":3}"""), Log(0));

        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));
        Assert.Empty(page.FindAll(".solia-log__context"));
        Assert.Null(Rows(page)[1].QuerySelector("button.solia-log__expand")); // nothing more to show

        page.Find("button.solia-log__expand").Click();

        IElement details = page.Find("tr.solia-log__details");
        Assert.Equal("1", details.GetAttribute("data-details-for"));
        Assert.Equal("{\n  \"mint\": \"abc\",\n  \"attempt\": 3\n}", page.Find("pre.solia-log__context").TextContent.ReplaceLineEndings("\n"));
        Assert.Equal("true", page.Find("button.solia-log__expand").GetAttribute("aria-expanded"));

        page.Find("button.solia-log__expand").Click();

        Assert.Empty(page.FindAll(".solia-log__context"));
    }

    [Fact]
    public void A_long_message_is_truncated_until_expanded()
    {
        string message = new string('x', BotLogs.MessagePreviewLength) + "TAIL";
        Returns(Log(1, message: message));

        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        Assert.Equal(new string('x', BotLogs.MessagePreviewLength) + "…", Cell(Rows(page)[0], "message"));

        page.Find("button.solia-log__expand").Click();

        Assert.Equal(message, Cell(Rows(page)[0], "message"));
        Assert.Empty(page.FindAll(".solia-log__context")); // no context to show
    }

    [Fact]
    public void Filters_come_from_the_url_and_reach_the_backend()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded("logs?level=warn&service=bot&limit=250");

        Assert.Equal(new BotLogsQuery { Level = BotLogLevel.Warn, Service = LogService.Bot, Limit = 250 }, Assert.Single(_api.BotLogsCalls));
        Assert.Equal("warn", page.Find(".solia-filter-level input").GetAttribute("value"));
        Assert.Equal("bot", page.Find(".solia-filter-service input").GetAttribute("value"));
        Assert.Equal("250", page.Find(".solia-filter-limit input").GetAttribute("value"));
    }

    [Fact]
    public void Applying_filters_updates_the_url_and_re_fetches()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();

        page.InvokeAsync(() => page.FindComponent<MudSelect<BotLogLevel?>>().Instance.ValueChanged.InvokeAsync(BotLogLevel.Error));
        page.InvokeAsync(() => page.FindComponent<MudSelect<LogService?>>().Instance.ValueChanged.InvokeAsync(LogService.DataPipeline));
        page.InvokeAsync(() => page.FindComponent<MudSelect<int>>().Instance.ValueChanged.InvokeAsync(500));
        page.Find("form.solia-filter-bar").Submit();

        page.WaitForAssertion(() => Assert.Equal(2, _api.BotLogsCalls.Count));
        Assert.Equal("?level=error&service=data-pipeline&limit=500", new Uri(Navigation.Uri).Query);
        Assert.Equal(new BotLogsQuery { Level = BotLogLevel.Error, Service = LogService.DataPipeline, Limit = 500 }, _api.BotLogsCalls[1]);

        page.Find("button.solia-filter-reset").Click();

        page.WaitForAssertion(() => Assert.Equal(3, _api.BotLogsCalls.Count));
        Assert.Equal("", new Uri(Navigation.Uri).Query);
        Assert.Equal(new BotLogsQuery(), _api.BotLogsCalls[2]);
    }

    [Fact]
    public async Task A_pushed_entry_matching_the_filters_is_prepended()
    {
        Returns(Log(2, "warn"), Log(1, "error"));
        IRenderedComponent<BotLogs> page = RenderLoaded("logs?level=warn&service=bot");
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        await PushAsync(page, Log(3, "error"));
        await PushAsync(page, Log(4, "info"));            // below the minimum level
        await PushAsync(page, Log(5, "error", "backend")); // another service
        await PushAsync(page, Log(3, "error"));            // already shown

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
        Assert.Single(_api.BotLogsCalls);
    }

    [Fact]
    public async Task The_tail_is_capped_to_the_limit_dropping_the_oldest()
    {
        Returns([.. Enumerable.Range(1, 50).Reverse().Select(i => Log(i))]);
        IRenderedComponent<BotLogs> page = RenderLoaded("logs?limit=50");
        page.WaitForAssertion(() => Assert.Equal(50, Rows(page).Count));

        await PushAsync(page, Log(51));
        await PushAsync(page, Log(52));

        page.WaitForAssertion(() => Assert.Equal(50, Rows(page).Count));
        Assert.Equal([52, 51], Ids(page)[..2]);
        Assert.Equal(3, Ids(page)[^1]);
    }

    [Fact]
    public async Task Pausing_stops_appending_and_resuming_catches_up()
    {
        Returns(Log(1));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        Assert.Equal("Live", page.Find(".solia-log-tail__state").TextContent.Trim());

        page.Find("button.solia-log-tail__toggle").Click();
        await PushAsync(page, Log(2));
        await PushAsync(page, Log(3));

        Assert.Equal("Paused", page.Find(".solia-log-tail__state").TextContent.Trim());
        Assert.Equal([1], Ids(page));
        Assert.Equal("2 new entries since paused", page.Find(".solia-log-tail__missed").TextContent.Trim());

        Returns(Log(3), Log(2), Log(1));
        page.Find("button.solia-log-tail__toggle").Click();

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
        Assert.Equal(2, _api.BotLogsCalls.Count);
        Assert.Empty(page.FindAll(".solia-log-tail__missed"));

        await PushAsync(page, Log(4));
        page.WaitForAssertion(() => Assert.Equal([4, 3, 2, 1], Ids(page)));
    }

    [Fact]
    public async Task A_burst_of_pushes_is_rendered_once()
    {
        Returns(Log(1));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        await PushUnrenderedAsync(page, Log(2));
        await PushUnrenderedAsync(page, Log(3));

        Assert.Equal([1], Ids(page));
        Assert.True(_clock.IsArmed(BotLogs.RenderInterval));

        _clock.Elapse(BotLogs.RenderInterval);

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
    }

    [Fact]
    public async Task Pausing_while_a_fetch_runs_keeps_the_rows_still_and_loses_nothing()
    {
        Returns(Log(1));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        TaskCompletionSource<IReadOnlyList<LogDto>> pending = HoldNextFetch();
        _ = page.InvokeAsync(_feed.ResyncAsync); // completes with the held fetch
        page.WaitForAssertion(() => Assert.Equal(2, _api.BotLogsCalls.Count));
        page.Find("button.solia-log-tail__toggle").Click();
        await PushAsync(page, Log(3));
        await page.InvokeAsync(() => pending.SetResult([Log(2), Log(1)])); // taken before 3 was written

        Assert.Equal([1], Ids(page));
        Assert.Equal("1 new entry since paused", page.Find(".solia-log-tail__missed").TextContent.Trim());

        Returns(Log(3), Log(2), Log(1));
        page.Find("button.solia-log-tail__toggle").Click();

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
        Assert.Equal(3, _api.BotLogsCalls.Count);
    }

    [Fact]
    public async Task Retrying_while_paused_leaves_the_rows_for_the_resume()
    {
        Returns(Log(1));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));
        _api.BotLogs = (_, _) => Task.FromException<IReadOnlyList<LogDto>>(new BackendUnavailableException());
        await page.InvokeAsync(_feed.ResyncAsync);
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-stale")));

        page.Find("button.solia-log-tail__toggle").Click();
        Returns(Log(2), Log(1));
        page.Find(".solia-stale button.solia-stale__retry").Click();

        page.WaitForAssertion(() => Assert.Empty(page.FindAll(".solia-stale")));
        Assert.Equal([1], Ids(page));

        page.Find("button.solia-log-tail__toggle").Click();

        page.WaitForAssertion(() => Assert.Equal([2, 1], Ids(page)));
    }

    [Fact]
    public async Task A_filter_change_drops_the_old_rows_and_a_failure_shows_the_error()
    {
        Returns(Log(2, "info"), Log(1, "error"));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        TaskCompletionSource<IReadOnlyList<LogDto>> pending = HoldNextFetch();
        await page.InvokeAsync(() => Navigation.NavigateTo("logs?level=error"));

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-loading")));
        Assert.Empty(Rows(page));

        await PushAsync(page, Log(3, "error"));
        await page.InvokeAsync(() => pending.SetException(new BackendUnavailableException()));

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-error")));
        Assert.Empty(Rows(page));
        Assert.Empty(page.FindAll(".solia-stale"));

        Returns(Log(1, "error"));
        page.Find("button.solia-error__retry").Click();

        page.WaitForAssertion(() => Assert.Equal([1], Ids(page)));
    }

    [Fact]
    public async Task A_push_during_a_filter_change_joins_the_new_rows()
    {
        Returns(Log(2, "info"), Log(1, "error"));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Equal(2, Rows(page).Count));

        TaskCompletionSource<IReadOnlyList<LogDto>> pending = HoldNextFetch();
        await page.InvokeAsync(() => Navigation.NavigateTo("logs?level=error"));
        await PushAsync(page, Log(3, "error"));
        await page.InvokeAsync(() => pending.SetResult([Log(1, "error")]));

        page.WaitForAssertion(() => Assert.Equal([3, 1], Ids(page)));
    }

    [Fact]
    public void Resuming_with_nothing_missed_does_not_re_fetch()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();

        page.Find("button.solia-log-tail__toggle").Click();
        page.Find("button.solia-log-tail__toggle").Click();

        Assert.Single(_api.BotLogsCalls);
    }

    [Fact]
    public async Task A_resync_re_fetches_or_waits_for_resume_when_paused()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();

        await page.InvokeAsync(_feed.ResyncAsync);
        page.WaitForAssertion(() => Assert.Equal(2, _api.BotLogsCalls.Count));

        page.Find("button.solia-log-tail__toggle").Click();
        await page.InvokeAsync(_feed.ResyncAsync);
        Assert.Equal(2, _api.BotLogsCalls.Count);

        page.Find("button.solia-log-tail__toggle").Click();
        page.WaitForAssertion(() => Assert.Equal(3, _api.BotLogsCalls.Count));
    }

    [Fact]
    public void Shows_the_loading_state_while_the_first_fetch_runs()
    {
        _api.BotLogs = (_, _) => new TaskCompletionSource<IReadOnlyList<LogDto>>().Task;

        IRenderedComponent<BotLogs> page = RenderLoaded();

        Assert.NotNull(page.Find(".solia-loading"));
    }

    [Fact]
    public void Shows_the_empty_state_when_nothing_matches()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-empty")));
        Assert.Empty(page.FindAll("table"));
    }

    [Fact]
    public async Task An_empty_view_fills_in_with_pushed_entries()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-empty")));

        await PushAsync(page, Log(1));

        page.WaitForAssertion(() => Assert.Equal([1], Ids(page)));
    }

    [Fact]
    public void Shows_the_error_state_and_retries()
    {
        _api.BotLogs = (_, _) => Task.FromException<IReadOnlyList<LogDto>>(new BackendUnavailableException());
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-error")));

        Returns(Log(1));
        page.Find("button.solia-error__retry").Click();

        page.WaitForAssertion(() => Assert.Equal([1], Ids(page)));
        Assert.Empty(page.FindAll(".solia-error"));
    }

    [Fact]
    public async Task A_failed_refresh_keeps_the_entries_under_a_warning()
    {
        Returns(Log(1));
        IRenderedComponent<BotLogs> page = RenderLoaded();
        page.WaitForAssertion(() => Assert.Single(Rows(page)));

        _api.BotLogs = (_, _) => Task.FromException<IReadOnlyList<LogDto>>(new BackendUnavailableException());
        await page.InvokeAsync(_feed.ResyncAsync);

        page.WaitForAssertion(() => Assert.NotNull(page.Find(".solia-stale")));
        Assert.Equal([1], Ids(page));
    }

    [Fact]
    public async Task An_entry_pushed_while_the_fetch_runs_is_kept_on_top()
    {
        var pending = new TaskCompletionSource<IReadOnlyList<LogDto>>();
        _api.BotLogs = (_, _) => pending.Task;
        IRenderedComponent<BotLogs> page = RenderLoaded();

        await PushAsync(page, Log(3));
        await page.InvokeAsync(() => pending.SetResult([Log(2), Log(1)]));

        page.WaitForAssertion(() => Assert.Equal([3, 2, 1], Ids(page)));
    }

    [Fact]
    public void Leaving_the_page_unsubscribes_from_the_feed()
    {
        IRenderedComponent<BotLogs> page = RenderLoaded();
        Assert.Equal(1, _feed.Subscribers(LiveFeedEvents.LogWritten));

        page.Instance.Dispose();

        Assert.Equal(0, _feed.Subscribers(LiveFeedEvents.LogWritten));
        Assert.Equal(0, _feed.ResyncSubscribers);
    }
}
