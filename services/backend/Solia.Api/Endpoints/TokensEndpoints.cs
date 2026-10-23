using Microsoft.EntityFrameworkCore;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary><c>GET /tokens/live</c> and <c>GET /tokens/history</c>.</summary>
public static class TokensEndpoints
{
    public const int DefaultLiveLimit = 50;

    public const int MaxLiveLimit = 500;

    /// <summary>The <c>final_outcome</c> labels (CHECK in SQL).</summary>
    public static readonly string[] Outcomes = ["pump", "rug", "neutral"];

    /// <summary><c>?sort=</c> keys for the live set; all descending, missing values last.</summary>
    public static readonly string[] LiveSorts = ["updated_at", "market_cap", "liquidity", "holders", "trending_score", "created_at"];

    public static RouteGroupBuilder MapTokensEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/tokens/live", GetLiveAsync);
        group.MapGet("/tokens/history", GetHistoryAsync);
        return group;
    }

    /// <summary>Tokens still tracked (<c>is_active</c>), newest update first unless <c>sort</c> says otherwise.</summary>
    internal static async Task<IReadOnlyList<LiveTokenDto>> GetLiveAsync(
        SoliaDbContext db, int? limit, string? sort, CancellationToken cancellationToken)
    {
        int take = QueryValidation.Limit(limit, DefaultLiveLimit, MaxLiveLimit);
        string sortKey = QueryValidation.OneOf(sort, "sort", LiveSorts) ?? "updated_at";

        IQueryable<TokenLive> query = db.TokensLive.AsNoTracking().Where(t => t.IsActive);
        // Postgres sorts NULLs first under DESC; order on "is null" first to keep them last. No index backs
        // these sorts on purpose: tokens_live is the hot, constantly rewritten set, and an index on a
        // snapshot column would cost every write its HOT update. The active set is small, so the LIMIT
        // runs as a top-N heapsort over it.
        IOrderedQueryable<TokenLive> ordered = sortKey switch
        {
            "market_cap" => query.OrderBy(t => t.LastMarketCap == null).ThenByDescending(t => t.LastMarketCap),
            "liquidity" => query.OrderBy(t => t.LastLiquidity == null).ThenByDescending(t => t.LastLiquidity),
            "holders" => query.OrderBy(t => t.Holders == null).ThenByDescending(t => t.Holders),
            "trending_score" => query.OrderBy(t => t.TrendingScore == null).ThenByDescending(t => t.TrendingScore),
            "created_at" => query.OrderBy(t => t.CreatedAt == null).ThenByDescending(t => t.CreatedAt),
            _ => query.OrderByDescending(t => t.UpdatedAt),
        };

        return await ordered.ThenBy(t => t.Mint)
            .Take(take)
            .Select(t => new LiveTokenDto(t.Mint, t.Symbol, t.Name, t.CreatedAt, t.LastPrice, t.LastMarketCap,
                t.LastLiquidity, t.Holders, t.TrendingScore, t.Venue, t.IsMigrated, t.UpdatedAt))
            .ToListAsync(cancellationToken);
    }

    /// <summary>The registry, newest first; <c>from</c>/<c>to</c> bound <c>first_seen_at</c> (inclusive).</summary>
    internal static async Task<PagedResponse<TokenHistoryDto>> GetHistoryAsync(
        SoliaDbContext db, DateTimeOffset? from, DateTimeOffset? to, string? outcome, int? page, int? pageSize,
        CancellationToken cancellationToken)
    {
        QueryValidation.Range(from, to);
        string? label = QueryValidation.OneOf(outcome, "outcome", Outcomes);
        PageRequest paging = PageRequest.From(page, pageSize);

        IQueryable<TokenHistory> query = db.TokensHistory.AsNoTracking();
        if (from is { } f)
        {
            query = query.Where(x => x.FirstSeenAt >= f);
        }

        if (to is { } upper)
        {
            query = query.Where(x => x.FirstSeenAt <= upper);
        }

        if (label is not null)
        {
            query = query.Where(x => x.FinalOutcome == label);
        }

        return await query
            .OrderByDescending(x => x.FirstSeenAt).ThenBy(x => x.Mint)
            .Select(x => new TokenHistoryDto(x.Mint, x.Symbol, x.Name, x.CreatedAt, x.FirstSeenAt, x.LastSeenAt, x.FinalOutcome))
            .ToPageAsync(db, paging, cancellationToken);
    }
}
