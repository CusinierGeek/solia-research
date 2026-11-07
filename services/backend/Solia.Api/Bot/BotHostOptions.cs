using System.ComponentModel.DataAnnotations;
using System.Globalization;
using Solia.Bot.Configuration;
using Solia.Core.Data;

namespace Solia.Api.Bot;

/// <summary>
/// How the backend hosts the paper-trading bot: the on/off switch and the live feed's polling knobs. Bound
/// from the same "Bot" section as the policy layer's <see cref="BotOptions"/> (each class reads its own keys)
/// and validated at startup. <see cref="Enabled"/> is also settable via the flat PAPER_BOT_ENABLED key.
/// </summary>
public sealed class BotHostOptions
{
    public const string Section = BotOptions.Section;

    /// <summary>Runs the bot inside the API host. Off by default: the bot needs a live feed and an ai-service.</summary>
    public bool Enabled { get; set; }

    /// <summary>Pause between two feed polls when the last one drained the backlog, [100 ms, 1 h].</summary>
    [Range(typeof(TimeSpan), "00:00:00.1", "01:00:00",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromSeconds(2);

    /// <summary>
    /// The single horizon the bot trades on; must be one the <c>predictions</c> table accepts. 15m is the entry-rule
    /// sweep's choice (see <see cref="BotOptions.MinPumpProb"/>) and the span of the default position timeout (900 s).
    /// </summary>
    [Required]
    public string Horizon { get; set; } = "15m";

    /// <summary>Most <c>features</c> rows read (as price ticks) per poll, [1, 100 000].</summary>
    [Range(1, 100_000)]
    public int MaxTicksPerPoll { get; set; } = 1_000;

    /// <summary>
    /// How long the feed holds its events to put the ones written out of time order back in order before the
    /// engine sees them, [0, 1 min]; 0 emits each poll as read. It delays the bot's reaction by up to that
    /// much (fills still happen at the snapshot's price).
    /// </summary>
    [Range(typeof(TimeSpan), "00:00:00", "00:01:00",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public TimeSpan ReorderWindow { get; set; } = TimeSpan.FromSeconds(2);

    /// <summary>
    /// How long the feed waits for a <c>features</c> id that was skipped (its insert not committed yet) before
    /// giving up on it as rolled back, [1 s, 10 min] — longer than the data-pipeline's longest write transaction.
    /// </summary>
    [Range(typeof(TimeSpan), "00:00:01", "00:10:00",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public TimeSpan MaxCommitDelay { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>
    /// Scoring floor: a row under this many <c>unique_traders</c> isn't queued for scoring, 0 = off. The queue scores
    /// one token at a time, so every inactive token delays the active ones. A capacity setting of the live bot, not
    /// the champion's <see cref="Solia.Bot.Policy.EligibilityRule"/>: training reads <c>features</c>, which this doesn't touch.
    /// </summary>
    [Range(0, int.MaxValue)]
    public int MinScoreTraders { get; set; }

    /// <summary>Scoring floor: a row under this much pool liquidity (SOL) isn't queued for scoring, 0 = off.</summary>
    [Range(typeof(decimal), "0", "1000000000",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public decimal MinScoreLiquiditySol { get; set; }

    public bool HasKnownHorizon() => PredictionHorizons.IsKnown(Horizon);

    /// <summary>Whether a row clears the scoring floor; an unknown value counts as under a floor that is on.</summary>
    public bool MeetsScoringFloor(int? uniqueTraders, decimal? liquiditySol) =>
        (uniqueTraders ?? 0) >= MinScoreTraders && (liquiditySol ?? 0m) >= MinScoreLiquiditySol;

    /// <summary>The floor as a regime value; null when it is off, so that it leaves a session's fingerprint alone.</summary>
    public string? ScoringFloorRegime() => MinScoreTraders == 0 && MinScoreLiquiditySol == 0m
        ? null
        : string.Create(CultureInfo.InvariantCulture, $"traders>={MinScoreTraders},liquidity>={MinScoreLiquiditySol:G29}");
}
