using Microsoft.Extensions.DependencyInjection;
using MudBlazor;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Sessions;
using Solia.Dashboard.Tests.Api;

namespace Solia.Dashboard.Tests.Sessions;

public class BotSessionSelectorTests : MudTestContext
{
    private static readonly DateTimeOffset Cutover = new(2026, 10, 30, 20, 26, 42, TimeSpan.Zero);

    private readonly FakeSoliaApiClient _api = new();

    public BotSessionSelectorTests()
    {
        Services.AddSingleton<ISoliaApiClient>(_api);
        _api.BotSessions = _ => Task.FromResult<IReadOnlyList<BotSessionDto>>(
        [
            Session(3, label: null, current: true, open: 2, orphaned: 1, committed: 0.4m),
            Session(2, label: "naive fills (pre-#165)", current: false, open: 0, orphaned: 3, committed: 0m),
        ]);
    }

    private static BotSessionDto Session(long id, string? label, bool current, long open, long orphaned, decimal committed) =>
        new(id, Cutover.AddHours(id - 3), current ? null : Cutover, label, 1, $"v{id}", null, current,
            10, 7, open, orphaned, 1.5m, committed);

    private IRenderedComponent<BotSessionSelector> RenderSelector(SessionSelection value, Action<SessionSelection>? changed = null)
    {
        IRenderedComponent<BotSessionSelector> selector = Render<BotSessionSelector>(p => p
            .Add(s => s.Value, value)
            .Add(s => s.ValueChanged, changed ?? (_ => { })));
        selector.WaitForAssertion(() => Assert.Equal(1, _api.BotSessionsCalls));
        return selector;
    }

    private static string Part(IRenderedComponent<BotSessionSelector> selector, string part) =>
        selector.Find($"[data-part={part}]").TextContent.Trim();

    [Fact]
    public void The_current_session_is_described_with_its_open_positions_and_orphans()
    {
        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Current);

        selector.WaitForAssertion(() => Assert.Equal("Champion v3", Part(selector, "model")));
        Assert.Equal("current", selector.Find(".solia-session").GetAttribute("data-session"));
        Assert.Equal("2 open (0.4 SOL committed)", Part(selector, "open"));
        Assert.Equal("1 orphaned", Part(selector, "orphaned"));
        Assert.Contains("now UTC", Part(selector, "period"));
        Assert.Equal("Current session (#3 · v3)", selector.Instance.Describe(SessionSelection.Current));
    }

    [Fact]
    public void A_past_session_is_named_by_its_label_and_shows_its_end()
    {
        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Of(2));

        selector.WaitForAssertion(() => Assert.Equal("Champion v2", Part(selector, "model")));
        Assert.Equal("0 open (0 SOL committed)", Part(selector, "open"));
        Assert.Equal("3 orphaned", Part(selector, "orphaned"));
        Assert.DoesNotContain("now", Part(selector, "period"));
        Assert.Equal("#2 · naive fills (pre-#165)", selector.Instance.Describe(SessionSelection.Of(2)));
    }

    [Fact]
    public void Every_session_is_offered_between_current_and_all()
    {
        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.All);

        selector.WaitForAssertion(() => Assert.Equal("Every paper trade, across every session.", Part(selector, "all")));
        Assert.Equal(
            [SessionSelection.Current, SessionSelection.Of(3), SessionSelection.Of(2), SessionSelection.All],
            selector.FindComponents<MudSelectItem<SessionSelection>>().Select(i => i.Instance.Value));
        Assert.Equal("All sessions", selector.Instance.Describe(SessionSelection.All));
    }

    [Fact]
    public async Task Picking_a_session_reports_it()
    {
        SessionSelection? picked = null;
        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Current, s => picked = s);

        await selector.InvokeAsync(() =>
            selector.FindComponent<MudSelect<SessionSelection>>().Instance.ValueChanged.InvokeAsync(SessionSelection.Of(2)));

        Assert.Equal(SessionSelection.Of(2), picked);
    }

    [Fact]
    public void Without_any_session_the_current_one_reads_every_trade()
    {
        _api.BotSessions = _ => Task.FromResult<IReadOnlyList<BotSessionDto>>([]);

        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Current);

        selector.WaitForAssertion(() => Assert.Contains("every trade is shown", Part(selector, "none")));
        Assert.Equal("Current session", selector.Instance.Describe(SessionSelection.Current));
    }

    [Fact]
    public void A_backend_failure_still_offers_current_and_all()
    {
        _api.BotSessions = _ => throw new BackendUnavailableException();

        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Current);

        selector.WaitForAssertion(() => Assert.Equal("The session list could not be loaded.", Part(selector, "unavailable")));
        Assert.Equal(
            [SessionSelection.Current, SessionSelection.All],
            selector.FindComponents<MudSelectItem<SessionSelection>>().Select(i => i.Instance.Value));
    }

    [Fact]
    public async Task A_reload_picks_up_the_moved_figures()
    {
        IRenderedComponent<BotSessionSelector> selector = RenderSelector(SessionSelection.Current);
        selector.WaitForAssertion(() => Assert.Equal("2 open (0.4 SOL committed)", Part(selector, "open")));
        _api.BotSessions = _ => Task.FromResult<IReadOnlyList<BotSessionDto>>(
            [Session(3, label: null, current: true, open: 0, orphaned: 1, committed: 0m)]);

        await selector.InvokeAsync(selector.Instance.ReloadAsync);

        Assert.Equal("0 open (0 SOL committed)", Part(selector, "open"));
        Assert.Equal(2, _api.BotSessionsCalls);
    }
}
