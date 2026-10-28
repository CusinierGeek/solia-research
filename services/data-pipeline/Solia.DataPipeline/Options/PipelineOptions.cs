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

    /// <summary>Span of the price EMA feature (as in a span-N exponential moving average).</summary>
    [Range(2, 1000)]
    public int EmaSpan { get; set; } = 20;

    /// <summary>Look-back period (in price updates) for the RSI feature.</summary>
    [Range(2, 1000)]
    public int RsiPeriod { get; set; } = 14;

    /// <summary>A buy of at least this many SOL counts towards the large-buy count feature.</summary>
    [Range(0.000001, 1_000_000)]
    public decimal LargeBuySol { get; set; } = DefaultLargeBuySol;

    public const decimal DefaultLargeBuySol = 1m;

    /// <summary>
    /// When false (default), holders are approximated from unique traders seen in the
    /// stream (free). When true, a paid Get Token Info enrichment may be used.
    /// </summary>
    public bool EnrichHolders { get; set; } = false;

    /// <summary>
    /// Opt-in (off by default): run the live pumpapi source and log every event, for
    /// observability / recovery verification. Overridable via the flat STREAM_PROBE env var.
    /// Mutually exclusive with <see cref="Ingest"/> (both would open the single live connection);
    /// enabling both fails startup validation.
    /// </summary>
    public bool StreamProbe { get; set; } = false;

    /// <summary>
    /// Opt-in (off by default): run the full ingestion engine (live source → features →
    /// PostgreSQL). Overridable via the flat INGEST env var. Requires a reachable database.
    /// </summary>
    public bool Ingest { get; set; } = false;

    /// <summary>Ingest and the probe each own the single live connection, so at most one may be on.</summary>
    public bool HasExclusiveLiveConsumer() => !(Ingest && StreamProbe);

    /// <summary>A window shorter than the snapshot cadence would compute features over no data.</summary>
    public bool HasConsistentWindow() => WindowSeconds >= SnapshotCadenceSeconds;
}
