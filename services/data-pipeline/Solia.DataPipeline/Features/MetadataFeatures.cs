namespace Solia.DataPipeline.Features;

/// <summary>
/// Token metadata features (FE-T12, issue #196): what a token says about itself at launch, before any trading
/// history. Two sources, both point-in-time:
/// <list type="bullet">
/// <item>stream-derived (free, deterministic): the kind of metadata host (<see cref="UriContentAddressed"/>) and the
/// narrative heat of the token's normalized name / ticker (<see cref="NarrativeTracker"/>);</item>
/// <item>fetched once per token by live ingestion (<see cref="Metadata.MetadataFetcher"/>): whether the metadata JSON
/// lists a website / X / Telegram link, visible only to snapshots taken at or after the fetch time. Null when not
/// known by the snapshot's ts (not fetched yet, fetch failed, or a replay of a token the cache doesn't hold).</item>
/// </list>
/// </summary>
public sealed record MetadataFeatures
{
    /// <summary>Seconds of stream history behind the narrative counts, capped at <see cref="NarrativeTracker.FullCoverage"/>.</summary>
    public int CoverageSec { get; init; }

    /// <summary>
    /// True when the metadata URI is content-addressed (IPFS / Arweave: the pump.fun UI default, immutable), false for a
    /// launch tool's own host; null when the token's create (or its URI) wasn't seen.
    /// </summary>
    public bool? UriContentAddressed { get; init; }

    public bool? HasWebsite { get; init; }
    public bool? HasTwitter { get; init; }
    public bool? HasTelegram { get; init; }

    /// <summary>
    /// Other launches in the last hour whose normalized name or ticker equals this token's; null before an hour of
    /// stream history, or without a usable name / ticker.
    /// </summary>
    public int? NameMatches1h { get; init; }

    /// <summary>
    /// Other tokens sharing a normalized name or ticker with this one that graduated in the last
    /// <see cref="NarrativeTracker.GraduationWindow"/>; null before <see cref="NarrativeTracker.FullCoverage"/> of
    /// history, or without a usable name / ticker.
    /// </summary>
    public int? NarrativeGraduations { get; init; }
}
