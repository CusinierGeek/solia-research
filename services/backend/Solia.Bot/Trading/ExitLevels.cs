using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Trading;

/// <summary>
/// The dynamic take-profit / stop-loss of a position, as fractions of the entry price and as absolute
/// price levels. Computed once, at entry, from the entry prediction (docs/ML-MODEL.md "Policy layer":
/// TP/SL from <c>expected_return</c> and recent volatility). Pure and deterministic.
/// <list type="bullet">
///   <item><b>Take-profit</b> = <see cref="PositionSizer.Gain"/>: <c>min(expected_return, TakeProfitPct)</c>
///   — the return the model expects, capped, and at least <see cref="BotOptions.MinTakeProfitPct"/>.
///   The same helper sizes the bet, so TP is exactly the gain <c>b</c> the position was sized on.
///   With <see cref="BotOptions.FixedTakeProfit"/>, TP is <see cref="BotOptions.TakeProfitPct"/> itself, for any
///   prediction the gate accepted.</item>
///   <item><b>Stop-loss</b> = <c>StopLossVolatilityMultiple · volatility</c>, clamped to
///   [<see cref="BotOptions.MinStopLossPct"/>, <see cref="BotOptions.StopLossPct"/>] (the cap wins should
///   the floor exceed it): a calm token gets a
///   tight stop, a noisy one a wider stop so noise doesn't stop it out. Falls back to
///   <see cref="BotOptions.StopLossPct"/> when volatility is unknown (null, non-finite, ≤ 0) or the
///   multiple is 0. Never wider than <see cref="BotOptions.StopLossPct"/>, so the sizer's loss
///   assumption <c>L = StopLossPct</c> stays a conservative bound.</item>
/// </list>
/// </summary>
/// <param name="TakeProfitPct">Distance above entry, &gt; 0.</param>
/// <param name="StopLossPct">Distance below entry, in (0, 1).</param>
/// <param name="TakeProfitPrice">Price at or above which the position takes profit.</param>
/// <param name="StopLossPrice">Price at or below which the position is stopped out.</param>
public readonly record struct ExitLevels(
    double TakeProfitPct,
    double StopLossPct,
    decimal TakeProfitPrice,
    decimal StopLossPrice)
{
    /// <summary>
    /// Levels for a position entered at <paramref name="entryPrice"/> on <paramref name="prediction"/>.
    /// The prediction must be one the entry gate accepted, i.e. with a tradeable gain
    /// (<see cref="PositionSizer.Gain"/> &gt; 0); anything else throws.
    /// </summary>
    public static ExitLevels Compute(ModelPrediction prediction, decimal entryPrice, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(entryPrice);

        double tp = PositionSizer.Gain(prediction, options);
        if (tp <= 0.0)
        {
            throw new ArgumentException(
                "The prediction has no tradeable gain; only an accepted entry has exit levels.", nameof(prediction));
        }

        if (options.FixedTakeProfit)
        {
            tp = options.TakeProfitPct;
        }

        double sl = options.StopLossPct;
        if (options.StopLossVolatilityMultiple > 0.0
            && prediction.Volatility is double vol && double.IsFinite(vol) && vol > 0.0)
        {
            // Floor first, cap last: the cap wins even on unvalidated options (floor above cap).
            sl = Math.Min(Math.Max(options.StopLossVolatilityMultiple * vol, options.MinStopLossPct), options.StopLossPct);
        }

        // tp ∈ (0, 10] and sl ∈ (0, 1), so both casts are safe and the stop stays above zero.
        return new ExitLevels(
            tp,
            sl,
            entryPrice * (1m + (decimal)tp),
            entryPrice * (1m - (decimal)sl));
    }
}
