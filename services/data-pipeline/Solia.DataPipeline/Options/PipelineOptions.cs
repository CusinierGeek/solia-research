using System.ComponentModel.DataAnnotations;

namespace Solia.DataPipeline.Options;

/// <summary>
/// Core pipeline knobs. Bound from the "Pipeline" config section; <see cref="EnrichHolders"/>
/// is also overridable via the flat ENRICH_HOLDERS env var (mirrors .env.example).
/// Validated at startup.
/// </summary>
public sealed class PipelineOptions
{
    public const string Section = "Pipeline";

    /// <summary>How often a feature snapshot is emitted per tracked token (seconds).</summary>
    [Range(1, 3600)]
    public int SnapshotCadenceSeconds { get; set; } = 10;

    /// <summary>Rolling window length used for feature computation (seconds).</summary>
    [Range(1, 86_400)]
    public int WindowSeconds { get; set; } = 300;

    /// <summary>Maximum number of tokens held in memory at once (eviction bound).</summary>
    [Range(1, 1_000_000)]
    public int MaxTrackedTokens { get; set; } = 5000;

    /// <summary>
    /// When false (default), holders are approximated from unique traders seen in the
    /// stream (free). When true, a paid Get Token Info enrichment may be used.
    /// </summary>
    public bool EnrichHolders { get; set; } = false;

    /// <summary>
    /// Opt-in (off by default): run the live pumpapi source and log every event, for
    /// observability / recovery verification. Overridable via the flat STREAM_PROBE env var.
    /// Continuous ingestion into the DB is wired separately (T7), not by this flag.
    /// </summary>
    public bool StreamProbe { get; set; } = false;

    /// <summary>A window shorter than the snapshot cadence would compute features over no data.</summary>
    public bool HasConsistentWindow() => WindowSeconds >= SnapshotCadenceSeconds;
}
