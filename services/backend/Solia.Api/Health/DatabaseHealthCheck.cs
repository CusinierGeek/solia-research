using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Solia.Core.Data;

namespace Solia.Api.Health;

/// <summary>
/// Readiness: can the backend query Postgres through <see cref="SoliaDbContext"/>? A <c>SELECT 1</c> round trip
/// on the shared data source — not <c>CanConnectAsync</c>, which Npgsql runs on a clone of the connection
/// string, without the password the secrets provider injected. The context is resolved inside the check, not
/// injected: a rejected connection string fails while the shared data source is built, and that must report
/// Unhealthy instead of failing the health endpoint itself.
/// </summary>
internal sealed class DatabaseHealthCheck(IServiceProvider services, HealthTransitionLog log) : IHealthCheck
{
    public const string Name = "database";

    public async Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            SoliaDbContext db = services.GetRequiredService<SoliaDbContext>();
            await db.Database.SqlQuery<int>($"SELECT 1 AS \"Value\"").SingleAsync(cancellationToken);
            log.Healthy(Name);
            return HealthCheckResult.Healthy(HealthDescriptions.DatabaseReachable);
        }
        catch (Exception ex)
        {
            // Any failure — including the check's timeout cancelling a hanging connect — is "unreachable". The
            // cause is logged on the transition only (through the redacting logger), never returned to the caller.
            log.Unhealthy(Name, HealthDescriptions.DatabaseUnreachable, ex);
            return new HealthCheckResult(context.Registration.FailureStatus, HealthDescriptions.DatabaseUnreachable);
        }
    }
}
