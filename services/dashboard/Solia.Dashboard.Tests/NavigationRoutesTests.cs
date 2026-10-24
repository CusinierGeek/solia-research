using System.Reflection;
using Microsoft.AspNetCore.Components;
using Solia.Dashboard.Navigation;

namespace Solia.Dashboard.Tests;

/// <summary>The nav list and the pages' @page routes can't drift apart.</summary>
public class NavigationRoutesTests
{
    // Framework pages that are routable but deliberately not in the navigation.
    private static readonly string[] NonNavRoutes = ["/not-found", "/Error"];

    private static readonly string[] PageRoutes = typeof(Program).Assembly.GetTypes()
        .SelectMany(t => t.GetCustomAttributes<RouteAttribute>())
        .Select(r => r.Template)
        .ToArray();

    [Fact]
    public void Every_nav_item_points_at_an_existing_page()
    {
        foreach (NavItem item in DashboardNav.Items)
        {
            Assert.Contains("/" + item.Href, PageRoutes);
        }
    }

    [Fact]
    public void Every_view_page_has_a_nav_item()
    {
        string[] navRoutes = DashboardNav.Items.Select(i => "/" + i.Href).ToArray();

        foreach (string route in PageRoutes.Except(NonNavRoutes))
        {
            Assert.Contains(route, navRoutes);
        }
    }

    [Fact]
    public void Nav_hrefs_are_unique_and_relative()
    {
        Assert.Equal(DashboardNav.Items.Count, DashboardNav.Items.Select(i => i.Href).Distinct().Count());
        Assert.All(DashboardNav.Items, i => Assert.False(i.Href.StartsWith('/'), $"'{i.Href}' must be relative to <base href>."));
    }
}
