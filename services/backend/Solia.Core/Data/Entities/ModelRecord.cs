namespace Solia.Core.Data.Entities;

/// <summary>
/// Model registry / versioning — table <c>models</c>. Only one row has <c>is_champion = true</c>
/// (enforced by a partial unique index in SQL). Read-oriented here: written by the ai-service.
/// </summary>
public sealed class ModelRecord
{
    public long Id { get; set; }

    /// <summary>Semantic or incremental version (unique).</summary>
    public string Version { get; set; } = null!;

    /// <summary>DB default <c>now()</c>.</summary>
    public DateTimeOffset CreatedAt { get; set; }

    /// <summary>Checkpoint location (<c>MODEL_DIR/...</c>).</summary>
    public string Path { get; set; } = null!;

    /// <summary>The currently served model.</summary>
    public bool IsChampion { get; set; }

    /// <summary>Validation metrics at promotion (JSONB).</summary>
    public string? Metrics { get; set; }

    /// <summary>Config snapshot (JSONB).</summary>
    public string? Hyperparams { get; set; }

    public long? NParams { get; set; }
}
