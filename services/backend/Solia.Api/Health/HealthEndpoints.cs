using Microsoft.Extensions.Diagnostics.HealthChecks;
using Solia.Core.Http;

namespace Solia.Api.Health;

/// <summary>
/// The probes: <c>/health/live</c> (process up — runs no check, so it never depends on the DB or the
/// ai-service, and is always <c>200</c>), <c>/health/ready</c> (the
/// <see cref="HealthChecksServiceCollectionExtensions.ReadyTag">ready</see> checks) and <c>/health</c> (every
/// check). An Unhealthy report is a <c>503</c> carrying the same health JSON; Healthy and Degraded are <c>200</c>.
/// </summary>
public static class HealthEndpoints
{
    public static IEndpointRouteBuilder MapHealthEndpoints(this IEndpointRouteBuilder app)
    {
        RouteGroupBuilder group = app.MapGroup(string.Empty)
            .WithTags("health")
            .WithMetadata(new ProducesResponseTypeMetadata(StatusCodes.Status200OK, typeof(HealthResponse), ["application/json"]));

        group.MapGet(ApiRoutes.Health, (HealthCheckService health, HttpContext http, CancellationToken ct) =>
                ReportAsync(health, http, _ => true, ct))
            .WithName("GetHealth")
            .WithSummary("Every health check (database, ai-service).")
            .WithMetadata(Unavailable);
        group.MapGet(ApiRoutes.HealthReady, (HealthCheckService health, HttpContext http, CancellationToken ct) =>
                ReportAsync(health, http, IsReadinessCheck, ct))
            .WithName("GetReadiness")
            .WithSummary("Readiness: the backend can reach its database and the ai-service.")
            .WithMetadata(Unavailable);
        group.MapGet(ApiRoutes.HealthLive, (HealthCheckService health, HttpContext http, CancellationToken ct) =>
                ReportAsync(health, http, _ => false, ct))
            .WithName("GetLiveness")
            .WithSummary("Liveness: the process is up (no dependency is checked).");

        return app;
    }

    private static readonly ProducesResponseTypeMetadata Unavailable =
        new(StatusCodes.Status503ServiceUnavailable, typeof(HealthResponse), ["application/json"]);

    private static bool IsReadinessCheck(HealthCheckRegistration registration) =>
        registration.Tags.Contains(HealthChecksServiceCollectionExtensions.ReadyTag);

    private static async Task<IResult> ReportAsync(
        HealthCheckService health, HttpContext http, Func<HealthCheckRegistration, bool> predicate, CancellationToken cancellationToken)
    {
        HealthReport report = await health.CheckHealthAsync(predicate, cancellationToken);
        http.Response.Headers.CacheControl = "no-store, no-cache";
        return TypedResults.Json(ToResponse(report),
            statusCode: report.Status == HealthStatus.Unhealthy ? StatusCodes.Status503ServiceUnavailable : StatusCodes.Status200OK);
    }

    internal static HealthResponse ToResponse(HealthReport report) =>
        new(report.Status.ToString(), report.TotalDuration,
            report.Entries.ToDictionary(
                e => e.Key,
                e => new HealthEntryDto(
                    e.Value.Status.ToString(), HealthDescriptions.Sanitize(e.Value.Description), e.Value.Duration, e.Value.Tags.ToList())));
}
