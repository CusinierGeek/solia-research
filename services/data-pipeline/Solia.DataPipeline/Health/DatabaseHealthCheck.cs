using Microsoft.Extensions.Diagnostics.HealthChecks;
using Npgsql;

namespace Solia.DataPipeline.Health;

/// <summary>
/// Reports healthy when the database answers a trivial query. Bounded by the timeout given at
/// registration (Program.cs). Cancellation is rethrown so the health service reports a timeout
/// (or drops an aborted probe) instead of "database unreachable".
/// </summary>
public sealed class DatabaseHealthCheck(NpgsqlDataSource dataSource) : IHealthCheck
{
    public async Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            await using NpgsqlCommand cmd = dataSource.CreateCommand("SELECT 1");
            await cmd.ExecuteScalarAsync(cancellationToken);
            return HealthCheckResult.Healthy("database reachable");
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            return HealthCheckResult.Unhealthy("database unreachable", ex);
        }
    }
}
