using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Health;

/// <summary>
/// Reflects the live stream's connection state. The stream runs in ingest mode or under the probe
/// (mutually exclusive); with neither it isn't running, so it reports Healthy ("idle"). Otherwise it's
/// Healthy while connected and Degraded (not Unhealthy — reconnecting is expected) while down, so a
/// transient drop doesn't fail /health.
/// </summary>
public sealed class StreamHealthCheck(StreamConnectionState state, IOptions<PipelineOptions> options) : IHealthCheck
{
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        if (!options.Value.Ingest && !options.Value.StreamProbe)
        {
            return Task.FromResult(HealthCheckResult.Healthy("stream idle (ingest and probe disabled)"));
        }

        StreamStatus status = state.Current;
        return Task.FromResult(status.IsConnected
            ? HealthCheckResult.Healthy("stream connected")
            : HealthCheckResult.Degraded($"stream disconnected ({status.LastError ?? "connecting"})"));
    }
}
