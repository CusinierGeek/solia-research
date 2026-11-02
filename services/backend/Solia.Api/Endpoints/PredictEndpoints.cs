using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Solia.Api.OpenApi;
using Solia.Core.Ai;
using Solia.Core.Configuration;
using Solia.Core.Data;
using Solia.Core.Data.Entities;
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
        group.MapPost("/predict", PredictAsync)
            .WithName("Predict").WithTags("model")
            .WithSummary("Score a token's latest feature window with the champion model (proxied to the ai-service).")
            .ProducesError(
                StatusCodes.Status400BadRequest,
                StatusCodes.Status404NotFound,
                StatusCodes.Status415UnsupportedMediaType,
                StatusCodes.Status422UnprocessableEntity,
                StatusCodes.Status502BadGateway,
                StatusCodes.Status503ServiceUnavailable);
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

        List<FeatureRow> window = await LoadWindowAsync(
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
    /// <c>venue</c> travels apart, as the raw string the ai-service one-hot encodes.
    /// With <paramref name="upTo"/>, the window ends at that snapshot time (no row after it is read).
    /// </summary>
    internal static async Task<List<FeatureRow>> LoadWindowAsync(
        SoliaDbContext db, string mint, int size, CancellationToken cancellationToken, DateTimeOffset? upTo = null)
    {
        IQueryable<Feature> query = db.Features.AsNoTracking().Where(f => f.Mint == mint);
        if (upTo is DateTimeOffset end)
        {
            query = query.Where(f => f.Ts <= end);
        }

        var rows = await query
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
                f.NetSolFlow,
                f.BuyVolumeRatio,
                f.AvgBuySol,
                f.MaxBuySol,
                f.LargeBuyCount,
                f.ReturnShort,
                f.ReturnMedium,
                f.ReturnLong,
                f.PriceAcceleration,
                f.EmaSlope,
                f.VolumeAcceleration,
                f.VolumeBurstRatio,
                f.PriceHistorySec,
                f.UniqueTraders,
                f.TopTraderShare,
                f.TraderHhi,
                f.NewTraderRatio,
                f.BurnedLiquidity,
                f.MktCoverageSec,
                f.MktLaunchRate5m,
                f.MktLaunchRate1h,
                f.MktGraduations1h,
                f.MktGraduationRatio1h,
                f.MktBuySol5m,
                f.MktBuySol1h,
                f.MktNetSolFlow5m,
                f.MktNetSolFlow1h,
                f.MktActiveTokens5m,
                f.MktActiveTokens1h,
                f.WalCoverageSec,
                f.WalFresh1hShare,
                f.WalFresh24hShare,
                f.WalMedianTraderAgeSec,
                f.WalCreatorAgeSec,
                f.WalCreatorPrevLaunches,
                f.WalCreatorBuySol,
                f.WalCreatorBuySupplyShare,
                f.WalCreatorSoldShare,
                f.WalCreatorFirstSellSec,
                f.RepCreatorSettledLaunches,
                f.RepCreatorRugShare,
                f.RepCreatorPumpShare,
                f.RepEarlyBuyers,
                f.RepEarlyKnownShare,
                f.RepEarlyWinRate,
                f.RepEarlyPnlSol,
                f.RepSmartBuyShare,
                f.RepSmartNetFlowSol,
                f.RepSmartTraders,
                f.RepBundleCreateBlockShare,
                f.RepBundleTopBlockShare,
                f.MetaCoverageSec,
                f.MetaUriContentAddressed,
                f.MetaHasWebsite,
                f.MetaHasTwitter,
                f.MetaHasTelegram,
                f.MetaNameMatches1h,
                f.MetaNarrativeGrads3h,
                f.Venue,
                f.IsMigrated,
            })
            .ToListAsync(cancellationToken);
        rows.Reverse();

        return rows.Select(f => new FeatureRow(new Dictionary<string, double?>
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
            ["net_sol_flow"] = (double?)f.NetSolFlow,
            ["buy_volume_ratio"] = f.BuyVolumeRatio,
            ["avg_buy_sol"] = (double?)f.AvgBuySol,
            ["max_buy_sol"] = (double?)f.MaxBuySol,
            ["large_buy_count"] = f.LargeBuyCount,
            ["return_short"] = f.ReturnShort,
            ["return_medium"] = f.ReturnMedium,
            ["return_long"] = f.ReturnLong,
            ["price_acceleration"] = f.PriceAcceleration,
            ["ema_slope"] = f.EmaSlope,
            ["volume_acceleration"] = f.VolumeAcceleration,
            ["volume_burst_ratio"] = f.VolumeBurstRatio,
            ["price_history_sec"] = f.PriceHistorySec,
            ["unique_traders"] = f.UniqueTraders,
            ["top_trader_share"] = f.TopTraderShare,
            ["trader_hhi"] = f.TraderHhi,
            ["new_trader_ratio"] = f.NewTraderRatio,
            ["burned_liquidity"] = f.BurnedLiquidity,
            ["mkt_coverage_sec"] = f.MktCoverageSec,
            ["mkt_launch_rate_5m"] = f.MktLaunchRate5m,
            ["mkt_launch_rate_1h"] = f.MktLaunchRate1h,
            ["mkt_graduations_1h"] = f.MktGraduations1h,
            ["mkt_graduation_ratio_1h"] = f.MktGraduationRatio1h,
            ["mkt_buy_sol_5m"] = f.MktBuySol5m,
            ["mkt_buy_sol_1h"] = f.MktBuySol1h,
            ["mkt_net_sol_flow_5m"] = f.MktNetSolFlow5m,
            ["mkt_net_sol_flow_1h"] = f.MktNetSolFlow1h,
            ["mkt_active_tokens_5m"] = f.MktActiveTokens5m,
            ["mkt_active_tokens_1h"] = f.MktActiveTokens1h,
            ["wal_coverage_sec"] = f.WalCoverageSec,
            ["wal_fresh_1h_share"] = f.WalFresh1hShare,
            ["wal_fresh_24h_share"] = f.WalFresh24hShare,
            ["wal_median_trader_age_sec"] = f.WalMedianTraderAgeSec,
            ["wal_creator_age_sec"] = f.WalCreatorAgeSec,
            ["wal_creator_prev_launches"] = f.WalCreatorPrevLaunches,
            ["wal_creator_buy_sol"] = f.WalCreatorBuySol,
            ["wal_creator_buy_supply_share"] = f.WalCreatorBuySupplyShare,
            ["wal_creator_sold_share"] = f.WalCreatorSoldShare,
            ["wal_creator_first_sell_sec"] = f.WalCreatorFirstSellSec,
            ["rep_creator_settled_launches"] = f.RepCreatorSettledLaunches,
            ["rep_creator_rug_share"] = f.RepCreatorRugShare,
            ["rep_creator_pump_share"] = f.RepCreatorPumpShare,
            ["rep_early_buyers"] = f.RepEarlyBuyers,
            ["rep_early_known_share"] = f.RepEarlyKnownShare,
            ["rep_early_win_rate"] = f.RepEarlyWinRate,
            ["rep_early_pnl_sol"] = f.RepEarlyPnlSol,
            ["rep_smart_buy_share"] = f.RepSmartBuyShare,
            ["rep_smart_net_flow_sol"] = f.RepSmartNetFlowSol,
            ["rep_smart_traders"] = f.RepSmartTraders,
            ["rep_bundle_create_block_share"] = f.RepBundleCreateBlockShare,
            ["rep_bundle_top_block_share"] = f.RepBundleTopBlockShare,
            ["meta_coverage_sec"] = f.MetaCoverageSec,
            ["meta_uri_content_addressed"] = f.MetaUriContentAddressed,
            ["meta_has_website"] = f.MetaHasWebsite,
            ["meta_has_twitter"] = f.MetaHasTwitter,
            ["meta_has_telegram"] = f.MetaHasTelegram,
            ["meta_name_matches_1h"] = f.MetaNameMatches1h,
            ["meta_narrative_grads_3h"] = f.MetaNarrativeGrads3h,
            ["is_migrated"] = f.IsMigrated ? 1 : 0,
        }, f.Venue)).ToList();
    }
}
