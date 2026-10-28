namespace Solia.Shared.Hosting;

/// <summary>
/// Container healthcheck probe. The .NET runtime images ship no <c>curl</c>/<c>wget</c>, so the
/// service binary probes itself: <c>dotnet Service.dll --healthcheck [url]</c> GETs its own
/// <c>/health</c> (or the path the service passes, e.g. the backend's <c>/health/live</c>; on the port Kestrel was configured with, see <see cref="ResolveDefaultUrl"/>) and exits 0 on a 2xx (Healthy or Degraded), 1 otherwise (503 Unhealthy,
/// connection refused, timeout). Must run before the host is built so it starts nothing else.
/// </summary>
public static class HealthProbe
{
    public const string Flag = "--healthcheck";

    /// <summary>The in-container port of the .NET images (their default <c>ASPNETCORE_HTTP_PORTS=8080</c>).</summary>
    public const int DefaultPort = 8080;

    /// <summary>The probed path when the service passes none.</summary>
    public const string DefaultPath = "/health";

    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(5);

    /// <summary>
    /// True when the process was started as a probe: exactly <c>--healthcheck [url]</c>. Any other
    /// argument shape (e.g. host arguments after the flag) is not a probe, so the service starts normally.
    /// </summary>
    public static bool IsRequested(string[] args) =>
        args.Length is 1 or 2 && string.Equals(args[0], Flag, StringComparison.Ordinal)
        && (args.Length == 1 || !args[1].StartsWith('-'));

    /// <summary>
    /// Probes the URL given after the flag (or <see cref="ResolveDefaultUrl"/>) and returns the process
    /// exit code. Never throws: every failure is reported on stderr and maps to 1.
    /// </summary>
    /// <param name="environment">Reads an environment variable; defaults to the process environment.</param>
    /// <param name="path">Path probed on the default URL (ignored when a URL follows the flag).</param>
    public static async Task<int> RunAsync(
        string[] args, HttpMessageHandler? handler = null, TimeSpan? timeout = null, TextWriter? error = null,
        Func<string, string?>? environment = null, string path = DefaultPath)
    {
        error ??= Console.Error;

        Uri? url;
        if (args.Length > 1)
        {
            if (!(Uri.TryCreate(args[1], UriKind.Absolute, out url)
                  && (url.Scheme == Uri.UriSchemeHttp || url.Scheme == Uri.UriSchemeHttps)))
            {
                await error.WriteLineAsync($"healthcheck: invalid url '{args[1]}' (absolute http(s) URL expected)");
                return 1;
            }
        }
        else
        {
            // Resolved only when no URL is given, so an explicit URL never depends on the service path.
            try
            {
                url = ResolveDefaultUrl(environment ?? Environment.GetEnvironmentVariable, path);
            }
            catch (ArgumentException ex)
            {
                await error.WriteLineAsync($"healthcheck: invalid path '{path}' ({ex.Message})");
                return 1;
            }
        }

        using var client = handler is null ? new HttpClient() : new HttpClient(handler, disposeHandler: false);
        client.Timeout = timeout ?? DefaultTimeout;
        try
        {
            using HttpResponseMessage response = await client.GetAsync(url);
            if (response.IsSuccessStatusCode)
            {
                return 0;
            }

            await error.WriteLineAsync($"healthcheck: {url} returned {(int)response.StatusCode}");
            return 1;
        }
        catch (Exception ex)
        {
            // Any failure is "unhealthy": a probe that crashes would report a stack trace, not the cause.
            await error.WriteLineAsync($"healthcheck: {url} unreachable ({ex.Message})");
            return 1;
        }
    }

    /// <summary>
    /// <c>http://localhost:{port}{path}</c> (<see cref="DefaultPath"/> unless given), the port being the one Kestrel listens on: the first
    /// <c>http://</c> entry of <c>ASPNETCORE_URLS</c> (which wins in Kestrel), else the first of
    /// <c>ASPNETCORE_HTTP_PORTS</c>, else <see cref="DefaultPort"/>. Unparseable values are skipped.
    /// </summary>
    public static Uri ResolveDefaultUrl(Func<string, string?> environment, string path = DefaultPath)
    {
        ArgumentNullException.ThrowIfNull(environment);
        if (string.IsNullOrEmpty(path) || path[0] != '/')
        {
            throw new ArgumentException("An absolute path ('/...') is expected.", nameof(path));
        }
        int port = PortFromUrls(environment("ASPNETCORE_URLS"))
                   ?? FirstPort(environment("ASPNETCORE_HTTP_PORTS"))
                   ?? DefaultPort;
        return new Uri($"http://localhost:{port}{path}");
    }

    private static readonly char[] ListSeparators = [';', ','];

    private static int? FirstPort(string? ports) =>
        ports?.Split(ListSeparators, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(ParsePort).FirstOrDefault(p => p is not null);

    private static int? PortFromUrls(string? urls)
    {
        const string http = "http://";
        foreach (string entry in urls?.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) ?? [])
        {
            if (!entry.StartsWith(http, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            // The host may be a wildcard (+, *) that Uri rejects: read the port after the authority's last ':'.
            string authority = entry[http.Length..].Split('/')[0];
            int colon = authority.LastIndexOf(':');
            if (colon < 0 || authority.EndsWith(']'))
            {
                return 80;
            }
            if (ParsePort(authority[(colon + 1)..]) is int port)
            {
                return port;
            }
        }
        return null;
    }

    private static int? ParsePort(string value) =>
        int.TryParse(value, out int port) && port is > 0 and <= 65535 ? port : null;
}
