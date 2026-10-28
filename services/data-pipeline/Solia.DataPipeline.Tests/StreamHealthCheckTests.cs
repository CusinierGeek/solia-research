using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Health;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests;

public class StreamHealthCheckTests
{
    private static Task<HealthCheckResult> Check(StreamConnectionState state, bool probe, bool ingest = false) =>
        new StreamHealthCheck(state, Microsoft.Extensions.Options.Options.Create(
                new PipelineOptions { StreamProbe = probe, Ingest = ingest }))
            .CheckHealthAsync(new HealthCheckContext());

    [Fact]
    public async Task Ingest_mode_reports_a_stream_drop_as_degraded()
    {
        var state = new StreamConnectionState();
        state.MarkConnected();
        state.MarkDisconnected("closed by server");

        HealthCheckResult result = await Check(state, probe: false, ingest: true);

        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("closed by server", result.Description);
    }

    [Fact]
    public async Task Ingest_mode_is_healthy_while_connected()
    {
        var state = new StreamConnectionState();
        state.MarkConnected();

        HealthCheckResult result = await Check(state, probe: false, ingest: true);

        Assert.Equal(HealthStatus.Healthy, result.Status);
    }

    [Fact]
    public async Task Idle_when_ingest_and_probe_disabled_even_if_disconnected()
    {
        HealthCheckResult result = await Check(new StreamConnectionState(), probe: false);

        Assert.Equal(HealthStatus.Healthy, result.Status);
        Assert.Contains("idle", result.Description);
    }

    [Fact]
    public async Task Healthy_while_connected()
    {
        var state = new StreamConnectionState();
        state.MarkConnected();

        HealthCheckResult result = await Check(state, probe: true);

        Assert.Equal(HealthStatus.Healthy, result.Status);
    }

    [Fact]
    public async Task Degraded_with_the_last_error_while_disconnected()
    {
        var state = new StreamConnectionState();
        state.MarkConnected();
        state.MarkDisconnected("closed by server");

        HealthCheckResult result = await Check(state, probe: true);

        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("closed by server", result.Description);
    }

    [Fact]
    public async Task Degraded_as_connecting_before_the_first_connect()
    {
        HealthCheckResult result = await Check(new StreamConnectionState(), probe: true);

        Assert.Equal(HealthStatus.Degraded, result.Status);
        Assert.Contains("connecting", result.Description);
    }

    [Fact]
    public void State_snapshot_changes_as_a_whole()
    {
        var state = new StreamConnectionState();
        state.MarkDisconnected("boom");
        StreamStatus down = state.Current;

        state.MarkConnected();
        StreamStatus up = state.Current;

        Assert.Equal((false, "boom"), (down.IsConnected, down.LastError)); // earlier snapshot untouched
        Assert.Equal((true, (string?)null), (up.IsConnected, up.LastError));
        Assert.NotNull(up.LastChangeUtc);
    }
}
