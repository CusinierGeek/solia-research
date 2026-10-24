using Solia.Shared.Configuration;

namespace Solia.Dashboard.Configuration;

/// <summary>Host wiring for the dashboard's options.</summary>
public static class DashboardServiceCollectionExtensions
{
    /// <summary>The validation failure message for a missing or malformed backend URL.</summary>
    public const string InvalidBaseUrlMessage = "Backend:BaseUrl (BACKEND_URL) must be an absolute http(s) URL.";

    // The dashboard's flat keys from .env.example and the option paths they override. Add a row to support a new one.
    private static readonly FlatAlias[] Aliases =
    [
        new("BACKEND_URL", $"{BackendOptions.Section}:{nameof(BackendOptions.BaseUrl)}"),
    ];

    /// <summary>
    /// Binds <see cref="BackendOptions"/> from the "Backend" section, overlays the flat BACKEND_URL key (a flat
    /// key wins) and validates it at startup (fail fast). Both are read when the options are resolved, so
    /// every configuration source — including ones added after this call (e.g. a test host's) — is honoured.
    /// </summary>
    public static IServiceCollection AddDashboardOptions(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<BackendOptions>()
            .Bind(configuration.GetSection(BackendOptions.Section))
            .Configure(options => ApplyFlatAliases(configuration, options))
            .Validate(o => o.HasValidBaseUrl(), InvalidBaseUrlMessage)
            .ValidateOnStart();
        return services;
    }

    private static void ApplyFlatAliases(IConfiguration configuration, BackendOptions options) =>
        new ConfigurationBuilder()
            .AddInMemoryCollection(FlatAliases.Resolve(key => configuration[key], Aliases))
            .Build()
            .GetSection(BackendOptions.Section)
            .Bind(options);
}
