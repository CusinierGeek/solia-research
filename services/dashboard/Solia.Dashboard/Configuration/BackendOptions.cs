namespace Solia.Dashboard.Configuration;

/// <summary>
/// Where the dashboard reaches the backend API (REST + the <c>/hubs/live</c> SignalR hub), bound from the
/// "Backend" config section and validated at startup (fail fast). <see cref="BaseUrl"/> is also overridable
/// via the flat BACKEND_URL env key (mirrors .env.example). There is deliberately no code default: a
/// deployment that forgets it must not silently point at localhost. Development supplies one in
/// appsettings.Development.json.
/// </summary>
public sealed class BackendOptions
{
    public const string Section = "Backend";

    /// <summary>Absolute http(s) URL of the backend, e.g. <c>http://localhost:8080</c> or <c>http://backend:8080</c>.</summary>
    public string BaseUrl { get; set; } = string.Empty;

    /// <summary>Bound on one HTTP attempt (connection + full response body).</summary>
    public TimeSpan AttemptTimeout { get; set; } = TimeSpan.FromSeconds(5);

    /// <summary>Bound on a whole call, the retry included; must leave room for two attempts and the retry delay.</summary>
    public TimeSpan TotalTimeout { get; set; } = TimeSpan.FromSeconds(12);

    /// <summary>True when <see cref="BaseUrl"/> is an absolute http(s) URL.</summary>
    public bool HasValidBaseUrl() =>
        Uri.TryCreate(BaseUrl, UriKind.Absolute, out Uri? uri)
        && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);

    /// <summary>
    /// <paramref name="relativePath"/> (no leading slash) under <paramref name="baseUrl"/>, which may or may not end
    /// with a slash — the one place REST and the live hub resolve their endpoints.
    /// </summary>
    internal static Uri Resolve(string baseUrl, string relativePath) =>
        new(new Uri(baseUrl.EndsWith('/') ? baseUrl : baseUrl + "/", UriKind.Absolute), relativePath);

    /// <summary>The nominal delay before the one retry.</summary>
    public static readonly TimeSpan RetryDelay = TimeSpan.FromMilliseconds(200);

    /// <summary>The longest the retry waits: <see cref="RetryDelay"/> plus Polly's ±25 % constant-backoff jitter.</summary>
    public static readonly TimeSpan MaxRetryDelay = RetryDelay * 1.25;

    /// <summary>The range Polly's timeout strategy accepts; anything outside it fails when the pipeline is built.</summary>
    public static readonly TimeSpan MinTimeout = TimeSpan.FromMilliseconds(10);

    /// <inheritdoc cref="MinTimeout"/>
    public static readonly TimeSpan MaxTimeout = TimeSpan.FromHours(24);

    /// <summary>
    /// True when both timeouts are within [<see cref="MinTimeout"/>, <see cref="MaxTimeout"/>] and
    /// <see cref="TotalTimeout"/> fits two full attempts plus the jittered retry delay between them.
    /// </summary>
    public bool HasConsistentTimeouts() =>
        IsInPollyRange(AttemptTimeout) && IsInPollyRange(TotalTimeout)
        && TotalTimeout >= 2 * AttemptTimeout + MaxRetryDelay;

    private static bool IsInPollyRange(TimeSpan timeout) => timeout >= MinTimeout && timeout <= MaxTimeout;
}
