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
    /// Virtual starting balance the bot paper-trades with, in SOL (sizes and P&amp;L are SOL too), (0, 1 000 000]. A bot with no capital can't
    /// trade; the upper bound catches typos (e.g. 1e30) that would make paper PnL meaningless.
    /// Overridable via PAPER_STARTING_BALANCE.
    /// </summary>
    [Range(typeof(decimal), "0", "1000000", MinimumIsExclusive = true,
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public decimal StartingBalance { get; set; } = 10m;

    /// <summary>Most positions open at once. Overridable via PAPER_MAX_CONCURRENT_POSITIONS.</summary>
    [Range(1, 1000)]
    public int MaxConcurrentPositions { get; set; } = 5;

    /// <summary>
    /// Minimum model confidence [0, 100] required before the bot acts on a prediction. Confidence is
    /// derived, 200 · max(|pump − 0.5|, |rug − 0.5|) (docs/ML-MODEL.md "Confidence"), so it is a margin,
    /// not a hit rate: with the rare-event heads mostly near 0 it is mostly high. At 60 it is implied by
    /// every entry while <see cref="MaxRugProb"/> ≤ 0.2 (rug ≤ 0.2 ⇒ confidence ≥ 60); a looser rug gate
    /// would let it block entries whose pump and rug both sit near 0.5 — lower it then.
    /// </summary>
    [Range(0.0, 100.0)]
    public double MinConfidence { get; set; } = 60;

    /// <summary>
    /// Entry gate: enter only when <c>pump_prob</c> (at the bot's horizon) is at least this, (0, 1] —
    /// zero would enter on every eligible token. Independent of <see cref="MaxRugProb"/> — the two heads
    /// are separate (docs/ML-MODEL.md), so the thresholds aren't compared with each other. The defaults
    /// (0.25 / 0.2 at 15m) were chosen by the entry-rule sweep on the champion's held-out windows
    /// (<c>python -m solia_ai.entry_sweep</c>, docs/ML-MODEL.md "Choosing the entry rule"): calibrated
    /// pump probabilities are low, so the pre-calibration 0.6 never fired. Re-run the sweep after a
    /// promotion — a new champion can shift the probability scale — and keep the ai-service's
    /// <c>ReferencePolicy</c> defaults in step.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true)]
    public double MinPumpProb { get; set; } = 0.25;

    /// <summary>
    /// Entry gate: enter only when <c>rug_prob</c> is at most this, [0, 1]. Also the exit signal: a fresh
    /// prediction above it for a held mint closes the position (see <c>SignalClassifier</c>).
    /// </summary>
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
    /// How often the host sweeps position timeouts while the feed is quiet, in seconds, [0, 3 600]. After
    /// this long with no event, the bot advances the event clock by the wall time elapsed since the last
    /// event (see <c>PaperTradingBot</c>), so a silent feed can't hold a position — and its capital —
    /// past its deadline. Zero disables the sweep: a deterministic replay must run on event time alone.
    /// </summary>
    [Range(0, 3_600)]
    public int TimeoutSweepSeconds { get; set; } = 5;

    /// <summary>
    /// Fill against the pool (see Trading.Fills): the liquidity gate, the pool-sized cap, constant-product
    /// slippage and the venue fees. False is the idealized simulator from before #165 — any size fills at the
    /// mark, for free — kept for before/after comparisons and policy-only tests; its P&amp;L can't evaluate a
    /// model (dust pools fake it).
    /// </summary>
    public bool RealisticFills { get; set; } = true;

    /// <summary>
    /// Liquidity gate, SOL per venue: no entry in a pool holding less real SOL (<c>features.liquidity</c>) than its
    /// venue's floor, or whose liquidity is unknown. <c>default</c> applies to every venue not listed (an unknown one
    /// included). Each in [0, 1 000 000]; 0 lets any known liquidity through. The defaults mirror the ai-service's
    /// <c>labeling.liquidity_floor</c> (docs/ML-MODEL.md "Liquidity floor"), so the bot trades the population the
    /// model learned on — keep them in step, and never below the champion's: on an AMM a price read off a pool under
    /// ~1 SOL is an artifact; bonding curves price off virtual reserves, so their real SOL may be ~0 while the
    /// price is valid (floor 0). Keys are case-insensitive.
    /// </summary>
    public Dictionary<string, decimal> LiquidityFloorSol { get; set; } = new(StringComparer.OrdinalIgnoreCase)
    {
        [Trading.Fills.DefaultVenue] = 1m,
        ["pumpfun"] = 0m,
        ["raydium-launchpad"] = 0m,
        ["meteora-launchpad"] = 0m,
        ["bonk"] = 0m,
    };

    /// <summary>
    /// Virtual SOL a venue's bonding curve prices with on top of its real SOL, per venue, each in
    /// [0, 1 000 000]; unlisted venues (the AMMs) have none. The fills' reserve is real + virtual SOL. Pump.fun's
    /// curve starts at 30 virtual SOL (its real SOL is added on top, which <c>features.liquidity</c> already
    /// counts); the other launchpads are approximated alike. Keys are case-insensitive.
    /// </summary>
    public Dictionary<string, decimal> VirtualSol { get; set; } = new(StringComparer.OrdinalIgnoreCase)
    {
        ["pumpfun"] = 30m,
        ["raydium-launchpad"] = 30m,
        ["meteora-launchpad"] = 30m,
        ["bonk"] = 30m,
    };

    /// <summary>
    /// Cap on a position as a fraction of its pool's SOL reserve, (0, 1] — on top of the balance caps. At 0.02 the
    /// price impact stays ~2 % per side; a pool too small for <see cref="MinPositionSize"/> is skipped.
    /// </summary>
    [Range(0.0, 1.0, MinimumIsExclusive = true)]
    public double MaxPoolFraction { get; set; } = 0.02;

    /// <summary>
    /// Per-side trading fee by venue, as a fraction of the SOL traded (0.01 = 1 %), each in [0, 0.5]; <c>default</c>
    /// for every venue not listed. Charged on entry and exit, so it is in <c>pnl</c>. Defaults from the venues'
    /// published schedules (2026, approximate — check before relying on them): Pump.fun curve 1.25 % (0.95 %
    /// protocol + 0.30 % creator), PumpSwap 0.30 %, the other launchpads ~1 %, the AMMs (Raydium, Meteora, Orca…)
    /// ~0.25 %. Keys are case-insensitive.
    /// </summary>
    public Dictionary<string, double> FeePct { get; set; } = new(StringComparer.OrdinalIgnoreCase)
    {
        [Trading.Fills.DefaultVenue] = 0.0025,
        ["pumpfun"] = 0.0125,
        ["pumpswap"] = 0.003,
        ["raydium-launchpad"] = 0.01,
        ["meteora-launchpad"] = 0.01,
        ["bonk"] = 0.01,
    };

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

        foreach (ValidationResult problem in CheckVenues(nameof(LiquidityFloorSol), LiquidityFloorSol, v => v is >= 0m and <= 1_000_000m, "[0, 1 000 000] SOL")
            .Concat(CheckVenues(nameof(VirtualSol), VirtualSol, v => v is >= 0m and <= 1_000_000m, "[0, 1 000 000] SOL"))
            .Concat(CheckVenues(nameof(FeePct), FeePct, v => v is >= 0.0 and <= 0.5, "[0, 0.5]")))
        {
            yield return problem;
        }
    }

    private static IEnumerable<ValidationResult> CheckVenues<T>(
        string name, IReadOnlyDictionary<string, T>? table, Func<T, bool> inRange, string range)
    {
        if (table is null)
        {
            yield return new ValidationResult($"{name} is required (a per-venue table, may be empty).", [name]);
            yield break;
        }

        foreach ((string venue, T value) in table)
        {
            if (string.IsNullOrWhiteSpace(venue))
            {
                yield return new ValidationResult($"{name} has an empty venue key.", [name]);
            }
            else if (!inRange(value))
            {
                yield return new ValidationResult($"{name}:{venue} ({value}) must be in {range}.", [name]);
            }
        }
    }
}
