using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Solia.Core.Ai;
using Solia.Core.Configuration;
using Solia.Core.Data;
using Solia.Core.Http;

namespace Solia.Api.Endpoints;

/// <summary>
/// <c>POST /predict</c>: loads the token's latest feature window from the DB and has the ai-service score it.
/// The ai-service is stateless, so the backend owns the window; the ai-service owns its bounds (it keeps the
/// champion's latest <c>seq_len</c> rows and rejects a window under its <c>min_seq_len</c> as
/// <see cref="AiServiceErrorKind.InsufficientHistory"/>). Upstream failures surface as
/// <see cref="AiServiceException"/>s, which the error-shape middleware maps to the envelope.
/// </summary>
public static class PredictEndpoints
{
    /// <summary>Longer than any Solana address (base58, ≤ 44 chars); anything beyond is not a mint.</summary>
    public const int MaxMintLength = 64;

    public static RouteGroupBuilder MapPredictEndpoints(this RouteGroupBuilder group)
    {
        group.MapPost("/predict", PredictAsync);
        return group;
    }

    /// <summary>
    /// Scores <c>mint</c> on its latest <see cref="AiServiceOptions.MaxFeatureWindowSize"/> feature rows, oldest
    /// first. An unknown mint, or one with no feature snapshot yet, is a 404; a window too short for the
    /// champion is a 422 <c>insufficient_history</c>.
    /// </summary>
    internal static async Task<PredictResponse> PredictAsync(
        PredictRequestDto body,
        SoliaDbContext db,
        ISoliaAiClient ai,
        IOptions<AiServiceOptions> options,
        CancellationToken cancellationToken)
    {
        (string mint, IReadOnlyList<string>? horizons) = Validate(body);

        List<IReadOnlyDictionary<string, double?>> window = await LoadWindowAsync(
            db, mint, options.Value.MaxFeatureWindowSize, cancellationToken);
        if (window.Count == 0)
        {
            bool known = await db.TokensHistory.AsNoTracking().AnyAsync(t => t.Mint == mint, cancellationToken);
            throw new NotFoundException(known ? $"no feature snapshot yet for mint {mint}" : $"unknown mint {mint}");
        }

        AiPredictResponse response = await ai.PredictAsync(AiPredictRequest.ForWindow(mint, window, horizons), cancellationToken);
        return new PredictResponse(
            mint,
            response.ModelVersion,
            response.Predictions
                .Select(p => new PredictionDto(p.Horizon, p.PumpProb, p.RugProb, p.ExpectedReturn, p.Confidence))
                .ToList());
    }

    private static (string Mint, IReadOnlyList<string>? Horizons) Validate(PredictRequestDto body)
    {
        string mint = body.Mint?.Trim() ?? "";
        if (mint.Length == 0 || mint.Length > MaxMintLength)
        {
            throw new InvalidRequestException($"mint is required (at most {MaxMintLength} characters).");
        }

        if (body.Horizons is null)
        {
            return (mint, null);
        }

        // Which horizons exist is the model's config, so the ai-service judges the values; only the shape is checked here.
        if (body.Horizons.Count == 0 || body.Horizons.Any(string.IsNullOrWhiteSpace))
        {
            throw new InvalidRequestException("horizons must be omitted (all) or a non-empty list of horizon names.");
        }

        // Deduplicated after trimming (first occurrence wins): "1h" and " 1h" are one horizon, not an upstream 400.
        return (mint, body.Horizons.Select(h => h.Trim()).Distinct(StringComparer.Ordinal).ToList());
    }

    /// <summary>
    /// The latest <paramref name="size"/> rows of <paramref name="mint"/>, oldest first, keyed by the ai-service's
    /// <c>FEATURE_COLUMNS</c> names (the <c>features</c> column names). NULLs stay null — the ai-service imputes them.
    /// </summary>
    internal static async Task<List<IReadOnlyDictionary<string, double?>>> LoadWindowAsync(
        SoliaDbContext db, string mint, int size, CancellationToken cancellationToken)
    {
        var rows = await db.Features.AsNoTracking()
            .Where(f => f.Mint == mint)
            .OrderByDescending(f => f.Ts)
            .Take(size)
            .Select(f => new
            {
                f.Price,
                f.Volume,
                f.MarketCap,
                f.Holders,
                f.Liquidity,
                f.BuyEvents,
                f.SellEvents,
                f.BuySellRatio,
                f.TokenAgeSec,
                f.TrendingScore,
                f.SocialMentions,
                f.HoldersGrowthRate,
                f.VolumeGrowthRate,
                f.TxCreationRate,
                f.Volatility,
                f.Ema,
                f.Rsi,
                f.PriceDerivative,
                f.IsMigrated,
            })
            .ToListAsync(cancellationToken);
        rows.Reverse();

        return rows.Select(f => (IReadOnlyDictionary<string, double?>)new Dictionary<string, double?>
        {
            ["price"] = (double?)f.Price,
            ["volume"] = (double?)f.Volume,
            ["market_cap"] = (double?)f.MarketCap,
            ["holders"] = f.Holders,
            ["liquidity"] = (double?)f.Liquidity,
            ["buy_events"] = f.BuyEvents,
            ["sell_events"] = f.SellEvents,
            ["buy_sell_ratio"] = f.BuySellRatio,
            ["token_age_sec"] = f.TokenAgeSec,
            ["trending_score"] = f.TrendingScore,
            ["social_mentions"] = f.SocialMentions,
            ["holders_growth_rate"] = f.HoldersGrowthRate,
            ["volume_growth_rate"] = f.VolumeGrowthRate,
            ["tx_creation_rate"] = f.TxCreationRate,
            ["volatility"] = f.Volatility,
            ["ema"] = f.Ema,
            ["rsi"] = f.Rsi,
            ["price_derivative"] = f.PriceDerivative,
            ["is_migrated"] = f.IsMigrated ? 1 : 0,
        }).ToList();
    }
}
