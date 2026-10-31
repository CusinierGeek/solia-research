using Microsoft.AspNetCore.Components;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.JSInterop;
using Solia.Dashboard.Formatting;
using Solia.Dashboard.Components.Layout;
using Solia.Dashboard.Live;
using Solia.Dashboard.Tests.Live;
using Solia.Dashboard.Navigation;

namespace Solia.Dashboard.Tests;

public class LayoutTests : MudTestContext
{
    public LayoutTests() => Services.AddSingleton<ILiveFeed>(new FakeLiveFeed(LiveFeedState.Reconnecting));

    private IRenderedComponent<MainLayout> RenderLayout() =>
        Render<MainLayout>(p => p.Add(l => l.Body, (RenderFragment)(b => b.AddMarkupContent(0, "<p id=\"page-body\">body</p>"))));

    [Fact]
    public void Layout_renders_a_nav_link_for_every_view()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        var links = layout.FindAll("nav a").Select(a => (Href: a.GetAttribute("href"), Text: a.TextContent.Trim())).ToList();

        Assert.Equal(6, DashboardNav.Items.Count);
        foreach (NavItem item in DashboardNav.Items)
        {
            Assert.Contains(links, l => l.Href == item.Href && l.Text == item.Title);
        }
    }

    [Fact]
    public void Layout_renders_the_page_body()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.Equal("body", layout.Find("#page-body").TextContent);
    }

    [Fact]
    public void Layout_always_shows_the_paper_trading_indicator()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.Contains("PAPER TRADING", layout.Find(".solia-paper-badge").TextContent);
    }

    [Fact]
    public void Layout_shows_the_live_feed_status()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.Equal("reconnecting", layout.Find(".solia-live-indicator").GetAttribute("data-state"));
    }

    [Fact]
    public void Times_are_shown_in_the_browser_zone_once_it_is_known()
    {
        JSInterop.Setup<string?>(DisplayTimeZone.BrowserFunction).SetResult("America/Toronto");

        IRenderedComponent<MainLayout> layout = RenderLayout();

        layout.WaitForAssertion(() => Assert.Equal("local", DisplayTimeZone.Label(layout.Instance.TimeZone)));
    }

    [Fact]
    public void An_unknown_or_unreadable_browser_zone_keeps_utc()
    {
        JSInterop.Setup<string?>(DisplayTimeZone.BrowserFunction).SetException(new JSException("no Intl"));

        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.Same(TimeZoneInfo.Utc, layout.Instance.TimeZone);
    }

    [Fact]
    public void Dark_mode_is_the_default_without_a_stored_choice()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.True(layout.Instance.IsDarkMode);
    }

    [Fact]
    public void Stored_light_theme_is_restored_on_load()
    {
        JSInterop.Setup<string?>("localStorage.getItem", MainLayout.DarkModeStorageKey).SetResult("false");

        IRenderedComponent<MainLayout> layout = RenderLayout();

        layout.WaitForAssertion(() => Assert.False(layout.Instance.IsDarkMode));
    }

    [Fact]
    public void Theme_toggle_persists_the_choice()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();

        layout.Find("button[aria-label='Toggle dark mode']").Click();

        Assert.False(layout.Instance.IsDarkMode);
        Assert.Equal([MainLayout.DarkModeStorageKey, "false"], JSInterop.VerifyInvoke("localStorage.setItem").Arguments);
    }

    [Fact]
    public void Unavailable_storage_keeps_the_default_theme()
    {
        JSInterop.Setup<string?>("localStorage.getItem", MainLayout.DarkModeStorageKey)
            .SetException(new JSException("storage blocked"));

        IRenderedComponent<MainLayout> layout = RenderLayout();

        Assert.True(layout.Instance.IsDarkMode);
    }

    [Fact]
    public void Drawer_toggle_flips_the_navigation_open_and_closed()
    {
        IRenderedComponent<MainLayout> layout = RenderLayout();
        // The responsive drawer's first state depends on the browser's breakpoint (unknown under bUnit), so only
        // the toggling is asserted.
        bool DrawerOpen() => layout.Find(".mud-drawer").ClassList.Contains("mud-drawer--open");
        bool initial = DrawerOpen();

        layout.Find("button[aria-label='Toggle navigation']").Click();
        Assert.Equal(!initial, DrawerOpen());

        layout.Find("button[aria-label='Toggle navigation']").Click();
        Assert.Equal(initial, DrawerOpen());
    }
}
