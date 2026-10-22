using Solia.Bot.Configuration;
using Solia.Bot.Domain;

namespace Solia.Bot.Policy;

/// <summary>
/// Model-driven position sizing (docs/ML-MODEL.md "Policy layer"): fractional Kelly on a two-outcome
/// bet whose payoffs come from the bot's own exit rules, scaled by confidence, capped at a max fraction
/// of balance, and floored to skip dust. Pure and deterministic — no I/O, no clock.
///
/// The bet, per unit staked:
/// <list type="bullet">
///   <item>no rug (probability <c>p = 1 − rug_prob</c>): gain <c>b = min(expected_return, TakeProfitPct)</c>
///   — the take-profit caps what the bot can realize;</item>
///   <item>rug (probability <c>q = rug_prob</c>): lose <c>L = StopLossPct</c> — the stop-loss caps the
///   loss. This assumes the stop fills; a rug that gaps through it loses more, which the hard
///   <see cref="BotOptions.MaxPositionFraction"/> cap and the fractional multiplier absorb.</item>
/// </list>
/// Kelly's optimal fraction is <c>f* = p/L − q/b = (p·b − q·L) / (b·L)</c>. A non-positive gain or edge
/// (<c>p·b − q·L</c>) sizes to zero. The committed fraction is
/// <c>f* · (confidence/100) · RiskFraction</c>, so <see cref="BotOptions.RiskFraction"/> = 1 really is
/// full Kelly; it is then clamped to <see cref="BotOptions.MaxPositionFraction"/>. <c>pump_prob</c> is an
/// entry gate (<see cref="SignalClassifier"/>), not a sizing input: <c>expected_return</c> already
/// prices in the pump likelihood, and counting it twice would double-weight the same head.
/// Size rises with expected return and confidence, falls with rug probability and with the stop-loss
/// distance.
/// </summary>
public static class PositionSizer
{
    /// <summary>
    /// The fraction of balance to commit, in [0, <see cref="BotOptions.MaxPositionFraction"/>]. Zero
    /// means the bet has no positive edge (or the prediction is malformed) — no trade.
    /// </summary>
    public static double Fraction(ModelPrediction prediction, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(options);

        if (!PredictionValidation.IsWellFormed(prediction))
        {
            return 0.0;
        }

        double gain = Math.Min(prediction.ExpectedReturn, options.TakeProfitPct);
        if (gain <= 0.0)
        {
            return 0.0;
        }

        double loss = options.StopLossPct;
        double rug = prediction.RugProb;
        double edge = ((1.0 - rug) * gain) - (rug * loss);
        if (edge <= 0.0)
        {
            return 0.0;
        }

        double kelly = edge / (gain * loss);
        double fraction = kelly * (prediction.Confidence / 100.0) * options.RiskFraction;
        return Math.Min(fraction, options.MaxPositionFraction);
    }

    /// <summary>
    /// The amount of <see cref="PortfolioState.Balance"/> to commit, or 0 to skip the trade (no edge, no
    /// balance, or below <see cref="BotOptions.MinPositionSize"/>). Takes the same
    /// <see cref="PortfolioState"/> snapshot the <see cref="EntryRule"/> gates on, so the gate and the
    /// size can't see different balances.
    /// </summary>
    public static decimal Size(ModelPrediction prediction, PortfolioState portfolio, BotOptions options)
    {
        double fraction = Fraction(prediction, options);
        if (fraction <= 0.0 || portfolio.Balance <= 0m)
        {
            return 0m;
        }

        // fraction is finite and in (0, MaxPositionFraction ≤ 1], so the cast can't overflow.
        decimal size = portfolio.Balance * (decimal)fraction;
        return size < options.MinPositionSize ? 0m : size;
    }
}
