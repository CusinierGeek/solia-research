using Microsoft.EntityFrameworkCore;
using Solia.Api.OpenApi;
using Solia.Bot.Policy;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary><c>GET /predictions/live</c> and <c>GET /predictions/history</c> — the model output the bot recorded.</summary>
public static class PredictionsEndpoints
{
    public const int DefaultLiveLimit = 100;

    public const int MaxLiveLimit = 1000;

    /// <summary>The <c>predictions.strategy</c> values: the codes the bot writes (CHECK in SQL).</summary>
    public static readonly string[] Strategies =
        [.. new[] { TradeSignal.Buy, TradeSignal.Sell, TradeSignal.Hold }.Select(s => s.ToCode())];

    /// <summary><c>?sort=</c> keys for the live set; all descending, missing values last.</summary>
    public static readonly string[] LiveSorts = ["ts", "pump_prob", "rug_prob", "confidence"];

    public static RouteGroupBuilder MapPredictionsEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/predictions/live", GetLiveAsync)
            .WithName("GetLivePredictions").WithTags("predictions")
            .WithSummary("The latest prediction per (tracked token, horizon), newest first (or by ?sort=).")
            .ProducesError(StatusCodes.Status400BadRequest);
        group.MapGet("/predictions/history", GetHistoryAsync)
            .WithName("GetPredictionHistory").WithTags("predictions")
            .WithSummary("Every recorded prediction, newest first, paginated.")
            .ProducesError(StatusCodes.Status400BadRequest);
        return group;
    }

    /// <summary>
    /// One row per (mint, horizon): the latest prediction of every token still tracked (<c>tokens_live.is_active</c>)
    /// at each horizon it has one, newest first unless <c>sort</c> says otherwise.
    /// </summary>
    internal static async Task<IReadOnlyList<PredictionRecordDto>> GetLiveAsync(
        SoliaDbContext db, int? limit, string? horizon, string? sort, CancellationToken cancellationToken)
    {
        int take = QueryValidation.Limit(limit, DefaultLiveLimit, MaxLiveLimit);
        string[] all = [.. PredictionHorizons.All];
        string? only = QueryValidation.OneOf(horizon, "horizon", all);
        string sortKey = QueryValidation.OneOf(sort, "sort", LiveSorts) ?? "ts";
        string[] horizons = only is null ? all : [only];

        // Active tokens × horizons, each joined LATERAL to its single newest prediction: one probe of
        // predictions_mint_horizon_ts_idx (scanned backward, already in (ts, id) order) per pair, a horizon the
        // token was never predicted at included — no per-token round trip and no scan of a token's history. Raw
        // SQL because EF can't correlate a LATERAL with a parameter collection; it stays a composable read, so
        // sorting, paging and the joins below are LINQ.
        IQueryable<Prediction> latest = db.Predictions.FromSql(
            $"""
             SELECT p.* FROM tokens_live t
             CROSS JOIN unnest({horizons}) AS h(horizon)
             CROSS JOIN LATERAL (
                 SELECT * FROM predictions
                 WHERE mint = t.mint AND horizon = h.horizon
                 ORDER BY ts DESC, id DESC
                 LIMIT 1) p
             WHERE t.is_active
             """).AsNoTracking();

        // The symbol comes from tokens_live, like /tokens/live (tokens_history may hold an older one).
        var rows = latest.Join(db.TokensLive, p => p.Mint, t => t.Mint, (p, t) => new { P = p, t.Symbol });

        // Postgres sorts NULLs first under DESC; order on "is null" first to keep them last. Ties: newest, then
        // mint, then horizon shortest first (its position in the list, not the string).
        var ordered = sortKey switch
        {
            "pump_prob" => rows.OrderBy(r => r.P.PumpProb == null).ThenByDescending(r => r.P.PumpProb)
                .ThenByDescending(r => r.P.Ts),
            "rug_prob" => rows.OrderBy(r => r.P.RugProb == null).ThenByDescending(r => r.P.RugProb)
                .ThenByDescending(r => r.P.Ts),
            "confidence" => rows.OrderBy(r => r.P.Confidence == null).ThenByDescending(r => r.P.Confidence)
                .ThenByDescending(r => r.P.Ts),
            _ => rows.OrderByDescending(r => r.P.Ts),
        };

        return await ordered.ThenBy(r => r.P.Mint).ThenBy(r => Array.IndexOf(all, r.P.Horizon))
            .Take(take)
            .Select(r => new PredictionRecordDto(r.P.Id, r.P.Mint, r.Symbol, r.P.Ts, r.P.Horizon, r.P.PumpProb,
                r.P.RugProb, r.P.Strategy, r.P.Confidence, r.P.PositionSize, r.P.TakeProfit, r.P.StopLoss,
                r.P.Model == null ? null : r.P.Model.Version))
            .ToListAsync(cancellationToken);
    }

    /// <summary>Predictions, newest first; <c>from</c>/<c>to</c> bound <c>ts</c> (inclusive).</summary>
    internal static async Task<PagedResponse<PredictionRecordDto>> GetHistoryAsync(
        SoliaDbContext db, string? mint, string? horizon, string? strategy, DateTimeOffset? from, DateTimeOffset? to,
        int? page, int? pageSize, CancellationToken cancellationToken)
    {
        string? h = QueryValidation.OneOf(horizon, "horizon", PredictionHorizons.All);
        string? s = QueryValidation.OneOf(strategy, "strategy", Strategies);
        QueryValidation.Range(from, to);
        PageRequest paging = PageRequest.From(page, pageSize);

        IQueryable<Prediction> query = db.Predictions.AsNoTracking();
        // Mints are case-sensitive base58: matched exactly, only trimmed.
        if (!string.IsNullOrWhiteSpace(mint))
        {
            string m = mint.Trim();
            query = query.Where(p => p.Mint == m);
        }

        if (h is not null)
        {
            query = query.Where(p => p.Horizon == h);
        }

        if (s is not null)
        {
            query = query.Where(p => p.Strategy == s);
        }

        if (from is { } f)
        {
            query = query.Where(p => p.Ts >= f);
        }

        if (to is { } upper)
        {
            query = query.Where(p => p.Ts <= upper);
        }

        return await query
            .OrderByDescending(p => p.Ts).ThenByDescending(p => p.Id)
            .Select(ToDto)
            .ToPageAsync(db, paging, cancellationToken);
    }

    // An expression (not a method) so EF translates it: symbol (the registry's) and model version come from LEFT
    // JOINs in the same statement. Shared with the live hub's prediction_created, so the two shapes can't drift.
    internal static readonly System.Linq.Expressions.Expression<Func<Prediction, PredictionRecordDto>> ToDto = p =>
        new PredictionRecordDto(p.Id, p.Mint, p.Token!.Symbol, p.Ts, p.Horizon, p.PumpProb, p.RugProb, p.Strategy,
            p.Confidence, p.PositionSize, p.TakeProfit, p.StopLoss, p.Model == null ? null : p.Model.Version);
}
