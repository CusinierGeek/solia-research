using System.ComponentModel.DataAnnotations;

namespace Solia.Bot.Configuration;

/// <summary>
/// Paper-trading bot knobs — the tunable <em>policy layer</em> (see docs/ML-MODEL.md): entry gates,
/// sizing and exit rules applied to the model's predictions. Bound from the "Bot" config section and
/// validated at startup. Every rule lives in the DataAnnotations below — the single place a rule is
/// defined. <see cref="StartingBalance"/> and <see cref="MaxConcurrentPositions"/> are also overridable
/// via the flat PAPER_STARTING_BALANCE / PAPER_MAX_CONCURRENT_POSITIONS env keys (mirrors
/// .env.example). Money is <see cref="decimal"/>; probabilities/ratios are <see cref="double"/>.
/// </summary>
public sealed class BotOptions
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

    /// <summary>Take-profit as a fraction above entry price (e.g. 0.5 = +50%), (0, 10] — zero would close every position immediately.</summary>
    [Range(0.0, 10.0, MinimumIsExclusive = true)]
    public double TakeProfitPct { get; set; } = 0.5;

    /// <summary>
    /// Stop-loss as a fraction below entry price (e.g. 0.3 = -30%), (0, 1). Zero would close every
    /// position immediately; 1 or more puts the stop at or below a zero price, so it could never trigger.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true, MaximumIsExclusive = true)]
    public double StopLossPct { get; set; } = 0.3;
}
