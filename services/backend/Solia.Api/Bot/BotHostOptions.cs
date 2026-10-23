using System.ComponentModel.DataAnnotations;
using Solia.Bot.Configuration;

namespace Solia.Api.Bot;

/// <summary>
/// How the backend hosts the paper-trading bot: the on/off switch and the live feed's polling knobs. Bound
/// from the same "Bot" section as the policy layer's <see cref="BotOptions"/> (each class reads its own keys)
/// and validated at startup. <see cref="Enabled"/> is also settable via the flat PAPER_BOT_ENABLED key.
/// </summary>
public sealed class BotHostOptions
{
    public const string Section = BotOptions.Section;

    /// <summary>
    /// The horizons the <c>predictions</c> table accepts (its CHECK constraint in services/db/migrations, the
    /// source of truth; a test fails when the two drift).
    /// </summary>
    public static readonly IReadOnlySet<string> Horizons = new HashSet<string>(StringComparer.Ordinal)
    {
        "30s", "1m", "5m", "15m", "1h",
    };

    /// <summary>Runs the bot inside the API host. Off by default: the bot needs a live feed and an ai-service.</summary>
    public bool Enabled { get; set; }

    /// <summary>Pause between two feed polls when the last one drained the backlog, [100 ms, 1 h].</summary>
    [Range(typeof(TimeSpan), "00:00:00.1", "01:00:00",
        ParseLimitsInInvariantCulture = true, ConvertValueInInvariantCulture = true)]
    public TimeSpan PollInterval { get; set; } = TimeSpan.FromSeconds(2);

    /// <summary>The single horizon the bot trades on; must be one the <c>predictions</c> table accepts.</summary>
    [Required]
    public string Horizon { get; set; } = "1m";

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

    public bool HasKnownHorizon() => Horizon is not null && Horizons.Contains(Horizon);
}
