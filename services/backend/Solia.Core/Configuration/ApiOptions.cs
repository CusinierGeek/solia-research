using System.ComponentModel.DataAnnotations;

namespace Solia.Core.Configuration;

/// <summary>
/// Cross-cutting host knobs for the backend API, bound from the "Api" config section and validated at
/// startup (fail fast on bad values). <see cref="BaseUrl"/> is also overridable via the flat BACKEND_URL
/// env key (mirrors .env.example). Endpoint-, AI-client- and bot-specific options land with their tasks;
/// this is only the scaffold seam.
/// </summary>
public sealed class ApiOptions
{
    public const string Section = "Api";

    /// <summary>
    /// Absolute http(s) URL the backend is reachable at by other services (mirrors BACKEND_URL). Advertised
    /// as a secondary server in the OpenAPI document for generated clients; the Swagger UI targets the
    /// host that served the document first, since this address (a compose hostname, a fixed port) is often
    /// not reachable from a browser. Defaults to the local-dev address so the host boots without a .env.
    /// </summary>
    [Required]
    public string BaseUrl { get; set; } = "http://localhost:8080";

    /// <summary>True when <see cref="BaseUrl"/> is an absolute http(s) URL.</summary>
    public bool HasValidBaseUrl() =>
        Uri.TryCreate(BaseUrl, UriKind.Absolute, out Uri? uri)
        && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);
}
