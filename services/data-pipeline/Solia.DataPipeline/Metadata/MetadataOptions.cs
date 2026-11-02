using System.ComponentModel.DataAnnotations;

namespace Solia.DataPipeline.Metadata;

/// <summary>
/// Token metadata fetch (FE-T12, <see cref="MetadataFetcher"/>). Bound from the "Metadata" config section. The fetch
/// runs only with live ingestion (<c>Pipeline:Ingest</c>); a replay only ever reads the cache.
/// </summary>
public sealed class MetadataOptions
{
    public const string Section = "Metadata";

    /// <summary>Fetch each new token's metadata JSON once (live ingestion only). Off: the link features stay NULL.</summary>
    public bool FetchEnabled { get; set; } = true;

    /// <summary>Gateway IPFS CIDs are fetched from (ipfs.io, which most URIs name, refuses scripted reads).</summary>
    public string IpfsGateway { get; set; } = "https://4everland.io/ipfs/";

    /// <summary>Gateway Arweave ids are fetched from.</summary>
    public string ArweaveGateway { get; set; } = "https://arweave.net/";

    /// <summary>Requests started per second, across every host.</summary>
    [Range(0.1, 100)]
    public double MaxRequestsPerSecond { get; set; } = 4;

    /// <summary>Requests in flight at once.</summary>
    [Range(1, 64)]
    public int MaxConcurrency { get; set; } = 4;

    /// <summary>Whole-request timeout (connect + headers + body), seconds.</summary>
    [Range(1, 60)]
    public int TimeoutSeconds { get; set; } = 8;

    /// <summary>A response body larger than this is a failure (metadata JSON is a few hundred bytes).</summary>
    [Range(1_024, 1_048_576)]
    public int MaxResponseBytes { get; set; } = 64 * 1024;

    /// <summary>Tokens waiting for a fetch; a new token arriving when it is full is skipped (counted, never blocks ingestion).</summary>
    [Range(1, 1_000_000)]
    public int QueueCapacity { get; set; } = 10_000;

    /// <summary>
    /// A fetch result stays visible to a token's snapshots for this many hours after the fetch, then is forgotten
    /// (bounds the in-memory cache). Fixed horizon, so live (pruning) and replay (loading the window's range) agree.
    /// </summary>
    [Range(1, 8_760)]
    public int LinkHorizonHours { get; set; } = 48;

    public TimeSpan LinkHorizon => TimeSpan.FromHours(LinkHorizonHours);

    /// <summary>Both gateways are absolute HTTPS URLs.</summary>
    public bool HasValidGateways() => IsHttps(IpfsGateway) && IsHttps(ArweaveGateway);

    private static bool IsHttps(string url) =>
        Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) && uri.Scheme == Uri.UriSchemeHttps;
}
