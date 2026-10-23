using Microsoft.EntityFrameworkCore;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary><c>GET /trades/history</c> — the bot's paper trades (paper trading only, no real orders).</summary>
public static class TradesEndpoints
{
    /// <summary>The <c>trades.status</c> values (CHECK in SQL).</summary>
    public static readonly string[] Statuses = ["open", "closed"];

    public static RouteGroupBuilder MapTradesEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/trades/history", GetHistoryAsync);
        return group;
    }

    /// <summary>Trades, most recently opened first; <c>from</c>/<c>to</c> bound <c>opened_at</c> (inclusive).</summary>
    internal static async Task<PagedResponse<TradeDto>> GetHistoryAsync(
        SoliaDbContext db, string? status, DateTimeOffset? from, DateTimeOffset? to, int? page, int? pageSize,
        CancellationToken cancellationToken)
    {
        string? state = QueryValidation.OneOf(status, "status", Statuses);
        QueryValidation.Range(from, to);
        PageRequest paging = PageRequest.From(page, pageSize);

        IQueryable<Trade> query = db.Trades.AsNoTracking();
        if (state is not null)
        {
            query = query.Where(t => t.Status == state);
        }

        if (from is { } f)
        {
            query = query.Where(t => t.OpenedAt >= f);
        }

        if (to is { } upper)
        {
            query = query.Where(t => t.OpenedAt <= upper);
        }

        return await query
            .OrderByDescending(t => t.OpenedAt).ThenByDescending(t => t.Id)
            .Select(t => new TradeDto(t.Id, t.Mint, t.PredictionId, t.OpenedAt, t.ClosedAt, t.EntryPrice, t.ExitPrice,
                t.Size, t.DurationSec, t.Pnl, t.PnlPct, t.Strategy, t.Confidence, t.ExitReason, t.Status))
            .ToPageAsync(db, paging, cancellationToken);
    }
}
