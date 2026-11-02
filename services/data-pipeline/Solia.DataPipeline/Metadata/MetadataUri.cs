using System.Net;

namespace Solia.DataPipeline.Metadata;

/// <summary>
/// Reads a token's metadata URI (the <c>uri</c> of its create event, FE-T12). Two jobs: classify the host for the
/// stream-derived <c>meta_uri_content_addressed</c> feature, and turn the URI into the URL the fetcher may request.
/// <para>
/// Content-addressed URIs (an IPFS CID behind any gateway, <c>ipfs://</c>, Arweave) are immutable, so they are
/// rewritten to the configured gateways: the public <c>ipfs.io</c> gateway most launches point at now refuses
/// scripted reads. Anything else is a launch tool's own host: fetched as given, HTTPS on the default port only.
/// </para>
/// </summary>
public static class MetadataUri
{
    private static readonly string[] ArweaveHosts = ["arweave.net", "www.arweave.net"];

    /// <summary>
    /// True for a content-addressed URI (IPFS / Arweave), false for any other parseable absolute URI, null when there
    /// is no usable URI.
    /// </summary>
    public static bool? IsContentAddressed(string? uri)
    {
        if (SchemePath(uri) is not null)
        {
            return true;
        }

        if (!TryParse(uri, out Uri? parsed))
        {
            return null;
        }

        return IpfsPath(parsed) is not null || IsArweave(parsed);
    }

    /// <summary>
    /// The URL to fetch for <paramref name="uri"/>, or null when it must not be fetched (not HTTPS after rewriting,
    /// a non-default port, user info, or unparseable). Redirects are vetted hop by hop with <see cref="ResolveRedirect"/>.
    /// </summary>
    public static Uri? ResolveFetchUrl(string? uri, MetadataOptions options)
    {
        string? target;
        if (SchemePath(uri) is ({ } scheme, { } rest))
        {
            target = Combine(scheme == "ipfs" ? options.IpfsGateway : options.ArweaveGateway, rest);
        }
        else if (!TryParse(uri, out Uri? parsed))
        {
            return null;
        }
        else
        {
            target = parsed.Scheme switch
            {
                "https" when IpfsPath(parsed) is { } cidPath => Combine(options.IpfsGateway, cidPath),
                "https" when IsArweave(parsed) => Combine(options.ArweaveGateway, parsed.PathAndQuery.TrimStart('/')),
                "https" => parsed.AbsoluteUri,
                _ => null,
            };
        }

        return target is not null
               && Uri.TryCreate(target, UriKind.Absolute, out Uri? url)
               && url.Scheme == Uri.UriSchemeHttps
               && url.IsDefaultPort
               && string.IsNullOrEmpty(url.UserInfo)
            ? url
            : null;
    }

    /// <summary>301, 302, 303, 307 or 308: a redirect the fetcher may follow (see <see cref="ResolveRedirect"/>).</summary>
    public static bool IsRedirect(HttpStatusCode status) =>
        status is HttpStatusCode.MovedPermanently or HttpStatusCode.Found or HttpStatusCode.SeeOther
            or HttpStatusCode.TemporaryRedirect or HttpStatusCode.PermanentRedirect;

    /// <summary>
    /// The URL a redirect from <paramref name="from"/> to <paramref name="location"/> (absolute or relative) leads to,
    /// or null when it must not be followed: no location, not HTTPS, a non-default port, user info, or an IP literal
    /// that isn't public. A hostname is resolved at connect time by <see cref="PublicAddress.ConnectAsync"/>, which
    /// applies the same public-address check to every hop.
    /// </summary>
    public static Uri? ResolveRedirect(Uri from, Uri? location)
    {
        if (location is null || !Uri.TryCreate(from, location, out Uri? next))
        {
            return null;
        }

        bool ipLiteral = next.HostNameType is UriHostNameType.IPv4 or UriHostNameType.IPv6;
        return next.Scheme == Uri.UriSchemeHttps
               && next.IsDefaultPort
               && string.IsNullOrEmpty(next.UserInfo)
               && !string.IsNullOrEmpty(next.Host)
               && (!ipLiteral || (IPAddress.TryParse(next.DnsSafeHost, out IPAddress? address) && PublicAddress.IsPublic(address)))
            ? next
            : null;
    }

    /// <summary>
    /// ("ipfs" | "ar", "&lt;id&gt;/rest") for an <c>ipfs://</c> / <c>ar://</c> URI, read as a string: a CID is
    /// case-sensitive and <see cref="Uri"/> would lower-case it as a host.
    /// </summary>
    private static (string Scheme, string Path)? SchemePath(string? uri)
    {
        string trimmed = uri?.Trim() ?? "";
        foreach (string scheme in (string[])["ipfs", "ar"])
        {
            string prefix = scheme + "://";
            if (trimmed.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                string rest = trimmed[prefix.Length..].TrimStart('/');
                if (scheme == "ipfs" && rest.StartsWith("ipfs/", StringComparison.OrdinalIgnoreCase))
                {
                    rest = rest["ipfs/".Length..]; // ipfs://ipfs/<cid>
                }

                return rest.Length > 0 ? (scheme, rest) : null;
            }
        }

        return null;
    }

    private static bool TryParse(string? uri, out Uri parsed)
    {
        parsed = null!;
        return !string.IsNullOrWhiteSpace(uri) && Uri.TryCreate(uri.Trim(), UriKind.Absolute, out parsed!)
               && !string.IsNullOrEmpty(parsed.Host);
    }

    /// <summary>"&lt;cid&gt;/rest?query" when the path is <c>/ipfs/&lt;cid&gt;...</c> on any host.</summary>
    private static string? IpfsPath(Uri uri)
    {
        const string marker = "/ipfs/";
        string path = uri.AbsolutePath;
        int at = path.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
        if (at < 0 || at + marker.Length >= path.Length)
        {
            return null;
        }

        return path[(at + marker.Length)..] + uri.Query;
    }

    private static bool IsArweave(Uri uri) =>
        ArweaveHosts.Contains(uri.Host, StringComparer.OrdinalIgnoreCase) && uri.AbsolutePath.Length > 1;

    private static string Combine(string gateway, string rest) => gateway.TrimEnd('/') + "/" + rest;
}
