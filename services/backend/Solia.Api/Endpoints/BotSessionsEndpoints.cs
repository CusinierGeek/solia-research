using Microsoft.EntityFrameworkCore;
using Solia.Api.OpenApi;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary>
/// <c>GET /bot/sessions</c> — the paper bot's sessions (issue #169), newest first, each with its trade counts. An open
/// trade with an id at or below the current session's <c>resumed_after_trade_id</c> was left by an earlier bot run
/// (a restart empties the bot's book): it is reported as <c>orphaned</c> and left out of <c>open</c> and
/// <c>committed</c>. Before any session is current, every open trade is orphaned.
/// </summary>
public static class BotSessionsEndpoints
{
    public static RouteGroupBuilder MapBotSessionsEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/bot/sessions", GetSessionsAsync)
            .WithName("GetBotSessions").WithTags("bot")
            .WithSummary("The paper bot's sessions, newest first, with their trade counts, P&L and orphaned trades.");
        return group;
    }

    internal static async Task<IReadOnlyList<BotSessionDto>> GetSessionsAsync(SoliaDbContext db, CancellationToken cancellationToken)
    {
        List<BotSessionRecord> sessions = await db.BotSessions.AsNoTracking()
            .OrderByDescending(s => s.Id)
            .ToListAsync(cancellationToken);
        long cutoff = sessions.FirstOrDefault(s => s.EndedAt == null)?.ResumedAfterTradeId ?? long.MaxValue;

        // One statement (conditional aggregates per session), so every figure comes from the same snapshot.
        var counts = await db.Trades.AsNoTracking()
            .Where(t => t.SessionId != null)
            .GroupBy(t => t.SessionId!.Value)
            .Select(g => new
            {
                Session = g.Key,
                Trades = g.LongCount(),
                Closed = g.LongCount(t => t.Status == ClosedTrades.Status && t.ClosedAt != null),
                Open = g.LongCount(t => t.Status == "open" && t.Id > cutoff),
                Orphaned = g.LongCount(t => t.Status == "open" && t.Id <= cutoff),
                Pnl = g.Sum(t => t.Status == ClosedTrades.Status && t.ClosedAt != null ? t.Pnl ?? 0m : 0m),
                Committed = g.Sum(t => t.Status == "open" && t.Id > cutoff ? t.Size ?? 0m : 0m),
            })
            .ToDictionaryAsync(c => c.Session, cancellationToken);

        return sessions.Select(s =>
        {
            var c = counts.GetValueOrDefault(s.Id);
            return new BotSessionDto(s.Id, s.StartedAt, s.EndedAt, s.Label, s.ModelId, s.ModelVersion,
                s.Options is { } options ? new RawJson(options) : null, s.EndedAt == null,
                c?.Trades ?? 0, c?.Closed ?? 0, c?.Open ?? 0, c?.Orphaned ?? 0, c?.Pnl ?? 0m, c?.Committed ?? 0m);
        }).ToList();
    }
}
