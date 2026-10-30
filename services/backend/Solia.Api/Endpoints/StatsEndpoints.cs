using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Npgsql;
using NpgsqlTypes;
using Solia.Api.OpenApi;
using Solia.Bot.Configuration;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary>
/// <c>GET /stats/winrate</c> and <c>GET /stats/performance</c>, derived from closed <c>trades</c>. Both read
/// the same set — <see cref="ClosedTrades"/> of one <see cref="SessionScopes">bot session</see> — so the dashboard
/// tiles agree.
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

    // The latest points of the curve, each with the running P&L over *every* closed trade of the scope up to it — one
    // statement, so the points and the offset of the trades before them come from one snapshot. The WHERE is
    // ClosedTrades.Closed + the session scope + since in SQL — keep them in step.
    private const string LatestEquitySql = """
        SELECT closed_at AS "ClosedAt",
               COALESCE(pnl, 0) AS "Pnl",
               SUM(COALESCE(pnl, 0)) OVER (ORDER BY closed_at, id ROWS UNBOUNDED PRECEDING) AS "Cumulative"
        FROM trades
        WHERE status = 'closed' AND closed_at IS NOT NULL
          AND (@session::bigint IS NULL OR session_id = @session)
          AND (@since::timestamptz IS NULL OR closed_at > @since)
        ORDER BY closed_at DESC, id DESC
        LIMIT @limit
        """;

    public static RouteGroupBuilder MapStatsEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/stats/winrate", GetWinRateAsync)
            .WithName("GetWinRate").WithTags("stats")
            .WithSummary("Win rate and exit breakdown over the closed trades of a bot session and window.")
            .ProducesError(StatusCodes.Status400BadRequest)
            .ProducesError(StatusCodes.Status404NotFound);
        group.MapGet("/stats/performance", GetPerformanceAsync)
            .WithName("GetPerformance").WithTags("stats")
            .WithSummary("Total P&L and the equity curve over the latest closed trades of a bot session.")
            .ProducesError(StatusCodes.Status400BadRequest)
            .ProducesError(StatusCodes.Status404NotFound);
        return group;
    }

    /// <summary>
    /// Win rate and exit breakdown over the trades of <c>session</c> (default the current one, see
    /// <see cref="SessionScopes"/>) closed within <c>window</c> (default <c>all</c>; see <see cref="TradeStats.WinRate"/>)
    /// and after <c>since</c>. A window is "closed strictly after now − span", the bot's <c>IClosedTradeCounter</c>
    /// definition of "since"; with both, both bounds apply.
    /// </summary>
    internal static async Task<WinRateResponse> GetWinRateAsync(
        SoliaDbContext db, TimeProvider clock, string? window, string? session, DateTimeOffset? since,
        CancellationToken cancellationToken)
    {
        string key = QueryValidation.OneOf(window, "window", WindowKeys) ?? "all";
        SessionScope scope = await SessionScopes.ResolveAsync(db, session, cancellationToken);

        IQueryable<Trade> closed = db.Trades.AsNoTracking().Closed().In(scope);
        if (Windows[key] is { } span)
        {
            DateTimeOffset after = clock.GetUtcNow() - span;
            closed = closed.Where(t => t.ClosedAt > after);
        }

        if (since is { } from)
        {
            closed = closed.Where(t => t.ClosedAt > from);
        }

        // One statement (conditional counts), so every figure comes from the same snapshot.
        var counts = await closed
            .GroupBy(_ => 1)
            .Select(g => new
            {
                Wins = g.Count(t => t.Pnl > 0m),
                Losses = g.Count(t => t.Pnl < 0m),
                Total = g.Count(),
                Tp = g.Count(t => t.ExitReason == "tp"),
                Sl = g.Count(t => t.ExitReason == "sl"),
                Signal = g.Count(t => t.ExitReason == "signal"),
                Timeout = g.Count(t => t.ExitReason == "timeout"),
            })
            .SingleOrDefaultAsync(cancellationToken);

        int wins = counts?.Wins ?? 0;
        int losses = counts?.Losses ?? 0;
        int breakEven = (counts?.Total ?? 0) - wins - losses;
        var exits = new ExitBreakdown(counts?.Tp ?? 0, counts?.Sl ?? 0, counts?.Signal ?? 0, counts?.Timeout ?? 0);
        return new WinRateResponse(TradeStats.WinRate(wins, losses, breakEven), wins, losses, breakEven, key, scope.Id, exits);
    }

    /// <summary>
    /// The equity curve over the latest <c>limit</c> closed trades (default <see cref="DefaultPerformanceLimit"/>) of
    /// <c>session</c> (default the current one) closed after <c>since</c>, in <c>closed_at</c> order, from the session's
    /// recorded <see cref="BotOptions.StartingBalance"/> — the configured one for every trade (see
    /// <see cref="SessionScope.StartingBalance"/>, <see cref="TradeStats.EquityCurve"/>). Equity and <c>total_pnl</c>
    /// still include every earlier trade of that scope. A closed trade without a recorded P&amp;L counts as 0.
    /// </summary>
    internal static async Task<PerformanceResponse> GetPerformanceAsync(
        SoliaDbContext db, IOptions<BotOptions> bot, int? limit, string? session, DateTimeOffset? since,
        CancellationToken cancellationToken)
    {
        int take = QueryValidation.Limit(limit, DefaultPerformanceLimit, MaxPerformanceLimit);
        SessionScope scope = await SessionScopes.ResolveAsync(db, session, cancellationToken);

        List<EquityRow> latest = await db.Database
            .SqlQueryRaw<EquityRow>(LatestEquitySql,
                new NpgsqlParameter("session", NpgsqlDbType.Bigint) { Value = (object?)scope.Id ?? DBNull.Value },
                new NpgsqlParameter("since", NpgsqlDbType.TimestampTz) { Value = (object?)since ?? DBNull.Value },
                new NpgsqlParameter("limit", NpgsqlDbType.Integer) { Value = take })
            .ToListAsync(cancellationToken);
        latest.Reverse();

        decimal pnlBefore = latest.Count == 0 ? 0m : latest[0].Cumulative - latest[0].Pnl;
        PerformanceResponse curve = TradeStats.EquityCurve(
            scope.StartingBalance(bot.Value.StartingBalance), latest.Select(r => (r.ClosedAt, r.Pnl)), pnlBefore);
        return curve with { Session = scope.Id };
    }

    private sealed class EquityRow
    {
        public DateTimeOffset ClosedAt { get; init; }

        public decimal Pnl { get; init; }

        public decimal Cumulative { get; init; }
    }
}
