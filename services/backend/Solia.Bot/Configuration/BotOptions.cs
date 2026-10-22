using System.ComponentModel.DataAnnotations;

namespace Solia.Bot.Configuration;

/// <summary>
/// Paper-trading bot knobs — the tunable <em>policy layer</em> (see docs/ML-MODEL.md): entry gates,
/// sizing and exit rules applied to the model's predictions. Bound from the "Bot" config section and
/// validated at startup. Every rule lives on this class — single-field rules as the DataAnnotations
/// below, cross-field rules in <see cref="Validate"/> — the single place a rule is defined.
/// <see cref="StartingBalance"/> and <see cref="MaxConcurrentPositions"/> are also overridable via the flat PAPER_STARTING_BALANCE / PAPER_MAX_CONCURRENT_POSITIONS env keys (mirrors
/// .env.example). Money is <see cref="decimal"/>; probabilities/ratios are <see cref="double"/>.
/// </summary>
public sealed class BotOptions : IValidatableObject
{
    public const string Section = "Bot";

    /// <summary>
    /// Virtual starting balance the bot paper-trades with, (0, 1 000 000]. A bot with no capital can't
    /// trade; the upper bound catches typos (e.g. 1e30) that would make paper PnL meaningless.
    /// Overridable via PAPER_STARTING_BALANCE.
    /// </summary>
    [Range(typeof(decimal), "0", "1000000", MinimumIsExclusive = true,
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public decimal StartingBalance { get; set; } = 10m;

    /// <summary>Most positions open at once. Overridable via PAPER_MAX_CONCURRENT_POSITIONS.</summary>
    [Range(1, 1000)]
    public int MaxConcurrentPositions { get; set; } = 5;

    /// <summary>Minimum model confidence [0, 100] required before the bot acts on a prediction.</summary>
    [Range(0.0, 100.0)]
    public double MinConfidence { get; set; } = 60;

    /// <summary>
    /// Entry gate: enter only when <c>pump_prob</c> is at least this, [0, 1]. Independent of
    /// <see cref="MaxRugProb"/> — the two heads are separate (docs/ML-MODEL.md), so the thresholds
    /// aren't compared with each other.
    /// </summary>
    [Range(0.0, 1.0)]
    public double MinPumpProb { get; set; } = 0.6;

    /// <summary>Entry gate: enter only when <c>rug_prob</c> is at most this, [0, 1].</summary>
    [Range(0.0, 1.0)]
    public double MaxRugProb { get; set; } = 0.2;

    /// <summary>Close an open position after this long regardless of price (seconds).</summary>
    [Range(1, 86_400)]
    public int PositionTimeoutSeconds { get; set; } = 900;

    /// <summary>Cap on the fraction of the current balance committed to a single position, (0, 1] — zero would never open a position.</summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true)]
    public double MaxPositionFraction { get; set; } = 0.2;

    /// <summary>
    /// Fractional-Kelly multiplier applied to the Kelly-optimal fraction when sizing a position, (0, 1]
    /// (see Policy.PositionSizer). 0.5 = half-Kelly (dampened, the usual choice); 1 = full Kelly. Lets global risk appetite scale
    /// without retraining. Separate from <see cref="MaxPositionFraction"/>, which is the hard cap.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true)]
    public double RiskFraction { get; set; } = 0.5;

    /// <summary>
    /// Dust floor: a computed position smaller than this is skipped (size 0) rather than opened, and the
    /// entry gate also requires the balance to be at least this much, [0, 1 000 000]. Zero disables the
    /// floor. Avoids economically meaningless positions. Must not exceed the largest position the bot could
    /// ever open (<see cref="StartingBalance"/> · <see cref="MaxPositionFraction"/>), or every trade would
    /// be dust. Money is <see cref="decimal"/>.
    /// </summary>
    [Range(typeof(decimal), "0", "1000000",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public decimal MinPositionSize { get; set; } = 0.001m;

    /// <summary>Take-profit as a fraction above entry price (e.g. 0.5 = +50%), (0, 10] — zero would close every position immediately.</summary>
    [Range(0.0, 10.0, MinimumIsExclusive = true)]
    public double TakeProfitPct { get; set; } = 0.5;

    /// <summary>
    /// Smallest upside worth trading, [0, 10]: a prediction whose capped expected return is below this
    /// has no tradeable edge (see Policy.PositionSizer.Gain) — otherwise a near-zero expected return would
    /// set the take-profit at the entry price and blow up the Kelly fraction. Zero disables the floor.
    /// Must not exceed <see cref="TakeProfitPct"/>.
    /// </summary>
    [Range(0.0, 10.0)]
    public double MinTakeProfitPct { get; set; } = 0.05;

    /// <summary>
    /// Stop-loss as a fraction below entry price (e.g. 0.3 = -30%), (0, 1). Zero would close every
    /// position immediately; 1 or more puts the stop at or below a zero price, so it could never trigger.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true, MaximumIsExclusive = true)]
    public double StopLossPct { get; set; } = 0.3;

    /// <summary>
    /// Dynamic stop-loss: the stop distance is this many multiples of the prediction's volatility
    /// (per-tick return σ), clamped to [<see cref="MinStopLossPct"/>, <see cref="StopLossPct"/>]
    /// (see Trading.ExitLevels), [0, 100]. Zero disables it — every position then uses the fixed
    /// <see cref="StopLossPct"/>.
    /// </summary>
    [Range(0.0, 100.0)]
    public double StopLossVolatilityMultiple { get; set; } = 3.0;

    /// <summary>
    /// Floor on the dynamic stop distance, (0, 1) — keeps a near-zero volatility from placing the stop at
    /// the entry price, where any down-tick would close the position. Must not exceed <see cref="StopLossPct"/>.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true, MaximumIsExclusive = true)]
    public double MinStopLossPct { get; set; } = 0.05;

    /// <summary>
    /// How old a mint's last price may be and still count as its market, in seconds, [1, 86 400]. An entry
    /// needs a mark at most this old; a signal or timeout exit on an older mark assumes the unobserved
    /// market hit the stop (see Trading.PositionManager), so a silent feed can't book a stale break-even.
    /// </summary>
    [Range(1, 86_400)]
    public int MaxMarkAgeSeconds { get; set; } = 60;

    /// <summary>
    /// How far behind the event clock an event may arrive and still be applied (at the clock's time),
    /// in seconds, [0, 3 600]. A later event is dropped rather than rewinding the clock — the live
    /// stream merges sources that aren't perfectly ordered.
    /// </summary>
    [Range(0, 3_600)]
    public int MaxEventLagSeconds { get; set; } = 5;

    /// <summary>
    /// Cross-field rules. Runs only once every single-field annotation has passed, so the values here
    /// are already in range.
    /// </summary>
    public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
    {
        decimal largestPosition = StartingBalance * (decimal)MaxPositionFraction;
        if (MinPositionSize > largestPosition)
        {
            yield return new ValidationResult(
                $"{nameof(MinPositionSize)} ({MinPositionSize}) exceeds the largest possible position " +
                $"{nameof(StartingBalance)} · {nameof(MaxPositionFraction)} ({largestPosition}); the bot could never trade.",
                [nameof(MinPositionSize), nameof(StartingBalance), nameof(MaxPositionFraction)]);
        }

        if (MinStopLossPct > StopLossPct)
        {
            yield return new ValidationResult(
                $"{nameof(MinStopLossPct)} ({MinStopLossPct}) exceeds {nameof(StopLossPct)} ({StopLossPct}); " +
                "the dynamic stop's floor would sit above its cap.",
                [nameof(MinStopLossPct), nameof(StopLossPct)]);
        }

        if (MinTakeProfitPct > TakeProfitPct)
        {
            yield return new ValidationResult(
                $"{nameof(MinTakeProfitPct)} ({MinTakeProfitPct}) exceeds {nameof(TakeProfitPct)} ({TakeProfitPct}); " +
                "no prediction could ever clear the take-profit floor.",
                [nameof(MinTakeProfitPct), nameof(TakeProfitPct)]);
        }
    }
}
