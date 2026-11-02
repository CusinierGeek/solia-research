namespace Solia.DataPipeline.Metadata;

/// <summary>Outcome of a token's single metadata fetch (stored as <c>token_metadata.status</c>).</summary>
public enum MetadataFetchStatus : short
{
    /// <summary>The JSON was read; the link flags are set.</summary>
    Ok = 1,

    /// <summary>The request failed (unreachable, timeout, non-2xx, too large, blocked address) or the JSON was malformed.</summary>
    Failed = 2,

    /// <summary>The URI can't be fetched (missing, not HTTPS, a non-default port...): nothing was requested.</summary>
    Unsupported = 3,
}

/// <summary>
/// What the pipeline keeps of a token's metadata (FE-T12): the outcome of its single fetch, when it completed, and
/// whether a website / X / Telegram link was listed. Nothing else — not the URI, the links themselves or the
/// description — is stored or logged.
/// </summary>
public sealed record TokenMetadataRecord(
    string Mint,
    DateTimeOffset FetchedAt,
    MetadataFetchStatus Status,
    bool HasWebsite = false,
    bool HasTwitter = false,
    bool HasTelegram = false)
{
    /// <summary>
    /// The links, as seen by a snapshot at <paramref name="ts"/>: only a successful fetch completed by then, and less
    /// than <paramref name="horizon"/> earlier, counts (point-in-time: a link fetched later is invisible before).
    /// </summary>
    public MetadataLinks? VisibleAt(DateTimeOffset ts, TimeSpan horizon) =>
        Status == MetadataFetchStatus.Ok && FetchedAt <= ts && ts - FetchedAt < horizon
            ? new MetadataLinks(HasWebsite, HasTwitter, HasTelegram)
            : null;
}
