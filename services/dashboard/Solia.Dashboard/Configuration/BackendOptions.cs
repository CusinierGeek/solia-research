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

    /// <summary>True when <see cref="BaseUrl"/> is an absolute http(s) URL.</summary>
    public bool HasValidBaseUrl() =>
        Uri.TryCreate(BaseUrl, UriKind.Absolute, out Uri? uri)
        && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);
}
