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

    /// <summary><c>initial</c> / <c>scheduled</c> / <c>n_trades</c> (CHECK lives in SQL).</summary>
    public string? Trigger { get; set; }

    public int? NSamples { get; set; }

    /// <summary>FK → <c>models</c>.</summary>
    public long? CandidateModelId { get; set; }

    /// <summary>FK → <c>models</c> (nullable).</summary>
    public long? ChampionModelId { get; set; }

    /// <summary>Loss, AUC, precision/recall, etc. (JSONB).</summary>
    public string? Metrics { get; set; }

    /// <summary><c>promoted</c> / <c>rejected</c> / <c>rollback</c> (CHECK lives in SQL).</summary>
    public string? Decision { get; set; }

    public string? Notes { get; set; }

    public ModelRecord? CandidateModel { get; set; }

    public ModelRecord? ChampionModel { get; set; }
}
