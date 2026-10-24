using MudBlazor;

namespace Solia.Dashboard.Navigation;

/// <summary>One entry of the side navigation.</summary>
/// <param name="Href">The page route, without a leading slash ("" = the home page), as the &lt;base href="/"&gt; resolves it.</param>
public sealed record NavItem(string Title, string Href, string Icon);

/// <summary>
/// The dashboard's views, in navigation order — the single list the nav menu renders, and that the tests
/// check every page route against, so a view can't be added without a nav entry (or vice versa).
/// </summary>
public static class DashboardNav
{
    public const string Overview = "";
    public const string LivePredictions = "predictions/live";
    public const string PredictionHistory = "predictions/history";
    public const string TradeHistory = "trades";
    public const string Performance = "performance";
    public const string BotLogs = "logs";

    public static IReadOnlyList<NavItem> Items { get; } =
    [
        new("Overview", Overview, Icons.Material.Outlined.Dashboard),
        new("Live predictions", LivePredictions, Icons.Material.Outlined.Bolt),
        new("Prediction history", PredictionHistory, Icons.Material.Outlined.History),
        new("Trade history", TradeHistory, Icons.Material.Outlined.SwapHoriz),
        new("Performance", Performance, Icons.Material.Outlined.ShowChart),
        new("Bot logs", BotLogs, Icons.Material.Outlined.Article),
    ];
}
