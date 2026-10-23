using Microsoft.Extensions.Diagnostics.HealthChecks;
using Solia.Core.Ai;

namespace Solia.Api.Health;

/// <summary>
/// Readiness: does the ai-service answer its <c>/health</c> (via <see cref="ISoliaAiClient"/>, a single attempt the
/// circuit breaker ignores)? Only liveness is probed — a missing champion is not a readiness failure, the AI-proxy
/// endpoints report it per call.
/// </summary>
internal sealed class AiServiceHealthCheck(ISoliaAiClient ai, HealthTransitionLog log) : IHealthCheck
{
    public const string Name = "ai_service";

    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            AiHealthResponse health = await ai.CheckHealthAsync(cancellationToken);
            if (!health.IsOk)
            {
                return Fail(context, HealthDescriptions.AiServiceNotHealthy, cause: null);
            }

            log.Healthy(Name);
            return HealthCheckResult.Healthy(HealthDescriptions.AiServiceReachable);
        }
        catch (Exception ex)
        {
            // Any failure — including the check's timeout cancelling the call — is "unreachable". The cause is
            // logged on the transition only, never returned to the caller.
            return Fail(context, HealthDescriptions.AiServiceUnreachable, ex);
        }
    }

    private HealthCheckResult Fail(HealthCheckContext context, string description, Exception? cause)
    {
        log.Unhealthy(Name, description, cause);
        return new HealthCheckResult(context.Registration.FailureStatus, description);
    }
}
