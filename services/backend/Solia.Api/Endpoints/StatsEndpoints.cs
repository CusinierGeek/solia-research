using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Solia.Bot.Configuration;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary>
/// <c>GET /stats/winrate</c> and <c>GET /stats/performance</c>, derived from closed <c>trades</c>. Both read
/// the same set — <see cref="ClosedTrades"/> — so the dashboard tiles agree.
/// </summary>
public static class StatsEndpoints
{
    public const int DefaultPerformanceLimit = 1000;

    public const int MaxPerformanceLimit = 10_000;

    /// <summary><c>?window=</c> values, in documented order; <c>all</c> (the default) = no bound.</summary>
    public static readonly string[] WindowKeys = ["all", "24h", "7d", "30d"];

    /// <summary>How far back (by <c>closed_at</c>) each <see cref="WindowKeys">window</see> looks.</summary>
    public static readonly IReadOnlyDictionary<string, TimeSpan?> Windows = new Dictionary<string, TimeSpan?>
    {
        ["all"] = null,
        ["24h"] = TimeSpan.FromHours(24),
        ["7d"] = TimeSpan.FromDays(7),
        ["30d"] = TimeSpan.FromDays(30),
    };

    // The latest points of the curve, each with the running P&L over *every* closed trade up to it — one
    // statement, so the points and the offset of the trades before them come from one snapshot. The WHERE is
    // ClosedTrades.Closed in SQL — keep the two in step.
    private const string LatestEquitySql = """
        SELECT closed_at AS "ClosedAt",
               COALESCE(pnl, 0) AS "Pnl",
               SUM(COALESCE(pnl, 0)) OVER (ORDER BY closed_at, id ROWS UNBOUNDED PRECEDING) AS "Cumulative"
        FROM trades
        WHERE status = 'closed' AND closed_at IS NOT NULL
        ORDER BY closed_at DESC, id DESC
        LIMIT {0}
        """;

    public static RouteGroupBuilder MapStatsEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/stats/winrate", GetWinRateAsync);
        group.MapGet("/stats/performance", GetPerformanceAsync);
        return group;
    }

    /// <summary>
    /// Win rate over the trades closed within <c>window</c> (default <c>all</c>; see
    /// <see cref="TradeStats.WinRate"/>). A window is "closed strictly after now − span", the bot's
    /// <c>IClosedTradeCounter</c> definition of "since".
    /// </summary>
    internal static async Task<WinRateResponse> GetWinRateAsync(
        SoliaDbContext db, TimeProvider clock, string? window, CancellationToken cancellationToken)
    {
        string key = QueryValidation.OneOf(window, "window", WindowKeys) ?? "all";

        IQueryable<Trade> closed = db.Trades.AsNoTracking().Closed();
        if (Windows[key] is { } span)
        {
            DateTimeOffset since = clock.GetUtcNow() - span;
            closed = closed.Where(t => t.ClosedAt > since);
        }

        // One statement (conditional counts), so wins/losses/break-even come from the same snapshot.
        var counts = await closed
            .GroupBy(_ => 1)
            .Select(g => new { Wins = g.Count(t => t.Pnl > 0m), Losses = g.Count(t => t.Pnl < 0m), Total = g.Count() })
            .SingleOrDefaultAsync(cancellationToken);

        int wins = counts?.Wins ?? 0;
        int losses = counts?.Losses ?? 0;
        int breakEven = (counts?.Total ?? 0) - wins - losses;
        return new WinRateResponse(TradeStats.WinRate(wins, losses, breakEven), wins, losses, breakEven, key);
    }

    /// <summary>
    /// The equity curve over the latest <c>limit</c> closed trades (default <see cref="DefaultPerformanceLimit"/>)
    /// in <c>closed_at</c> order, from the bot's configured <see cref="BotOptions.StartingBalance"/> (see
    /// <see cref="TradeStats.EquityCurve"/>). Equity and <c>total_pnl</c> still include every earlier trade.
    /// A closed trade without a recorded P&amp;L counts as 0.
    /// </summary>
    internal static async Task<PerformanceResponse> GetPerformanceAsync(
        SoliaDbContext db, IOptions<BotOptions> bot, int? limit, CancellationToken cancellationToken)
    {
        int take = QueryValidation.Limit(limit, DefaultPerformanceLimit, MaxPerformanceLimit);

        List<EquityRow> latest = await db.Database
            .SqlQueryRaw<EquityRow>(LatestEquitySql, take)
            .ToListAsync(cancellationToken);
        latest.Reverse();

        decimal pnlBefore = latest.Count == 0 ? 0m : latest[0].Cumulative - latest[0].Pnl;
        return TradeStats.EquityCurve(bot.Value.StartingBalance, latest.Select(r => (r.ClosedAt, r.Pnl)), pnlBefore);
    }

    private sealed class EquityRow
    {
        public DateTimeOffset ClosedAt { get; init; }

        public decimal Pnl { get; init; }

        public decimal Cumulative { get; init; }
    }
}
