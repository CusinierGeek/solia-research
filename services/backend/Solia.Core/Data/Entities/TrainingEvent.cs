namespace Solia.Core.Data.Entities;

/// <summary>
/// Audit trail of training / retrain runs — table <c>training_events</c>.
/// Read-oriented here: written by the ai-service continuous-training loop.
/// </summary>
public sealed class TrainingEvent
{
    public long Id { get; set; }

    /// <summary>DB default <c>now()</c>.</summary>
    public DateTimeOffset StartedAt { get; set; }

    public DateTimeOffset? FinishedAt { get; set; }

    /// <summary>One of <see cref="TrainingTriggers.All"/> (CHECK lives in SQL).</summary>
    public string? Trigger { get; set; }

    public int? NSamples { get; set; }

    /// <summary>FK → <c>models</c>.</summary>
    public long? CandidateModelId { get; set; }

    /// <summary>FK → <c>models</c> (nullable).</summary>
    public long? ChampionModelId { get; set; }

    /// <summary>Loss, AUC, precision/recall, etc. (JSONB).</summary>
    public string? Metrics { get; set; }

    /// <summary>One of <see cref="TrainingDecisions.All"/>, null while the run is in flight (CHECK lives in SQL).</summary>
    public string? Decision { get; set; }

    public string? Notes { get; set; }

    public ModelRecord? CandidateModel { get; set; }

    public ModelRecord? ChampionModel { get; set; }
}

/// <summary>The <c>training_events.trigger</c> values (mirrors the SQL CHECK, migration 0010).</summary>
public static class TrainingTriggers
{
    public const string Initial = "initial";
    public const string Scheduled = "scheduled";
    public const string NTrades = "n_trades";
    public const string Manual = "manual";

    public static readonly IReadOnlyList<string> All = [Initial, Scheduled, NTrades, Manual];
}

/// <summary>The <c>training_events.decision</c> values (mirrors the SQL CHECK, migration 0010).</summary>
public static class TrainingDecisions
{
    public const string Promoted = "promoted";
    public const string Rejected = "rejected";
    public const string Rollback = "rollback";

    /// <summary>The run crashed or errored: it trained no usable model.</summary>
    public const string Failed = "failed";

    public static readonly IReadOnlyList<string> All = [Promoted, Rejected, Rollback, Failed];

    /// <summary>
    /// The outcomes of a run that trained no model on the trades since the last training: a <see cref="Failed"/>
    /// run produced nothing usable, and a <see cref="Rollback"/> only restores the previous champion.
    /// </summary>
    public static readonly IReadOnlyList<string> NoModelTrained = [Rollback, Failed];

    /// <summary>
    /// The outcomes of a run that consumed the trades it trained on (a rejected candidate still did), so they reset
    /// the trades-since-last-train count: every decision but <see cref="NoModelTrained"/>.
    /// </summary>
    public static readonly IReadOnlyList<string> ResetTradeCount = [.. All.Except(NoModelTrained)];
}
