using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Health;

/// <summary>
/// Reflects the live stream's connection state. When the probe is off the stream isn't running,
/// so it reports Healthy ("idle"). When on, it's Healthy while connected and Degraded (not
/// Unhealthy — reconnecting is expected) while down, so a transient drop doesn't fail /health.
/// </summary>
public sealed class StreamHealthCheck(StreamConnectionState state, IOptions<PipelineOptions> options) : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        if (!options.Value.StreamProbe)
        {
            return Task.FromResult(HealthCheckResult.Healthy("stream idle (probe disabled)"));
        }

        StreamStatus status = state.Current;
        return Task.FromResult(status.IsConnected
            ? HealthCheckResult.Healthy("stream connected")
            : HealthCheckResult.Degraded($"stream disconnected ({status.LastError ?? "connecting"})"));
    }
}
