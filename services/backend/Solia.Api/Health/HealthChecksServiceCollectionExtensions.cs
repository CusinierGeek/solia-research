using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Solia.Api.Logging;

namespace Solia.Api.Health;

/// <summary>Registers the backend's dependency checks.</summary>
public static class HealthChecksServiceCollectionExtensions
{
    /// <summary>Tag of the checks <c>/health/ready</c> runs.</summary>
    public const string ReadyTag = "ready";

    /// <summary>
    /// Bound on each check, so a probe answers inside an orchestrator's probe timeout even when a dependency
    /// hangs: a check that outlives it reports Unhealthy.
    /// </summary>
    public static readonly TimeSpan CheckTimeout = TimeSpan.FromSeconds(5);

    /// <summary>
    /// Log category of the health-check service, which logs every Unhealthy probe at Error. Kept out of the
    /// `logs` table (console only): <see cref="HealthTransitionLog"/> records failures once per transition there.
    /// </summary>
    internal const string HealthCheckServiceCategory = "Microsoft.Extensions.Diagnostics.HealthChecks";

    /// <summary>
    /// The database (<see cref="DatabaseHealthCheck"/>) and ai-service (<see cref="AiServiceHealthCheck"/>)
    /// readiness checks, both tagged <see cref="ReadyTag"/> and Unhealthy on failure.
    /// </summary>
    public static IServiceCollection AddSoliaHealthChecks(this IServiceCollection services)
    {
        services.TryAddSingleton<HealthTransitionLog>();
        services.Configure<LoggerFilterOptions>(options =>
            options.AddFilter<PostgresLoggerProvider>(HealthCheckServiceCategory, LogLevel.None));
        services.AddHealthChecks()
            .AddCheck<DatabaseHealthCheck>(DatabaseHealthCheck.Name, HealthStatus.Unhealthy, [ReadyTag], CheckTimeout)
            .AddCheck<AiServiceHealthCheck>(AiServiceHealthCheck.Name, HealthStatus.Unhealthy, [ReadyTag], CheckTimeout);
        return services;
    }
}
