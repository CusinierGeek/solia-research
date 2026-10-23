using Solia.Core.Configuration;
using Solia.Shared.Data;

namespace Solia.Api.Hosting;

/// <summary>
/// Host wiring for the backend API: validated <see cref="ApiOptions"/> and the Postgres log sink
/// (service = 'backend'). The EF <c>SoliaDbContext</c> is wired in <c>Program.cs</c> via
/// <c>AddSoliaDbContext</c> and the AI client via <c>AddSoliaAiClient</c>; the secrets provider and the secrets-aware data source via <c>AddSoliaSecrets</c>.
/// </summary>
public static class ApiServiceCollectionExtensions
{
    /// <summary>The `logs.service` value the backend's rows carry.</summary>
    public const string BackendService = "backend";

    /// <summary>
    /// Binds <see cref="ApiOptions"/> from the "Api" section, overlays the flat .env keys (BACKEND_URL, …)
    /// and validates it at startup (fail fast). Both are read when the options are resolved, so every
    /// configuration source — including ones added after this call — is honoured.
    /// </summary>
    public static IServiceCollection AddApiOptions(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<ApiOptions>()
            .Bind(configuration.GetSection(ApiOptions.Section))
            .Configure(options => ApiFlatEnvAliases.Apply(configuration, options))
            .ValidateDataAnnotations()
            .Validate(o => o.HasValidBaseUrl(), "Api:BaseUrl (BACKEND_URL) must be an absolute http(s) URL.")
            .ValidateOnStart();
        return services;
    }

    /// <summary>
    /// Registers the shared pooled data source and the best-effort <c>PostgresLogSink</c> for service
    /// 'backend' (see <see cref="PostgresServiceCollectionExtensions"/>), which the <c>Postgres</c> logger
    /// provider drains into. A malformed connection string disables the mirror instead of failing startup.
    /// </summary>
    public static IServiceCollection AddBackendLogSink(this IServiceCollection services, IConfiguration configuration) =>
        services.AddPostgresLogSink(configuration, BackendService);
}
