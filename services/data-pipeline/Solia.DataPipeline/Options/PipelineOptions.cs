using System.ComponentModel.DataAnnotations;
using Solia.DataPipeline.Features;

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

    /// <summary>Short momentum window (seconds): short return, burst numerator, acceleration step.</summary>
    [Range(1, 86_400)]
    public int MomentumShortSeconds { get; set; } = MomentumWindows.Default.ShortSeconds;

    /// <summary>Medium momentum window (seconds): medium return.</summary>
    [Range(1, 86_400)]
    public int MomentumMediumSeconds { get; set; } = MomentumWindows.Default.MediumSeconds;

    /// <summary>Long momentum window (seconds): long return, burst denominator.</summary>
    [Range(1, 86_400)]
    public int MomentumLongSeconds { get; set; } = MomentumWindows.Default.LongSeconds;

    public MomentumWindows Momentum => new(MomentumShortSeconds, MomentumMediumSeconds, MomentumLongSeconds);

    /// <summary>
    /// Cross-token wallet memory (FE-T9, <see cref="WalletStore"/>): a wallet inactive for longer than
    /// this is forgotten (it counts as fresh again). Default 14 days, aligned with the <c>features</c> retention.
    /// </summary>
    [Range(1, 8_760)]
    public int WalletTtlHours { get; set; } = 336;

    /// <summary>
    /// Horizon of the wallet features: ages are capped at it and a creator's previous launches are counted
    /// within it, so any history at least this long gives the same features. Also the replay warm-up the
    /// wallet features need (they stay NULL until this much history has been observed). At least 24 h
    /// (the longest freshness bucket), at most <see cref="WalletTtlHours"/>.
    /// </summary>
    [Range(24, 8_760)]
    public int WalletAgeCapHours { get; set; } = 24;

    /// <summary>How often (event time, seconds) live ingestion persists the changed wallets to <c>wallet_state</c>.</summary>
    [Range(10, 86_400)]
    public int WalletPersistSeconds { get; set; } = 300;

    /// <summary>The wallet age cap can't outlive the wallets it measures.</summary>
    public bool HasConsistentWalletHorizon() => WalletAgeCapHours <= WalletTtlHours;

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

    /// <summary>
    /// Momentum windows must be strictly increasing and fit the rolling window, including the two
    /// short windows the accelerations compare. A look-back of exactly the window is fine: pruning
    /// keeps the price in force at the window's start.
    /// </summary>
    public bool HasConsistentMomentumWindows() =>
        MomentumShortSeconds < MomentumMediumSeconds
        && MomentumMediumSeconds < MomentumLongSeconds
        && MomentumLongSeconds <= WindowSeconds
        && 2 * MomentumShortSeconds <= WindowSeconds;
}
