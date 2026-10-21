using Microsoft.Extensions.Options;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Hosting;

/// <summary>
/// Opt-in observability: when <see cref="PipelineOptions.StreamProbe"/> is set, consumes the
/// live pumpapi source and logs every event. Lets us watch real Pump.fun events scroll by and
/// verify reconnect recovery. A no-op otherwise. This is NOT the production ingestion path
/// (that's the T7 engine) — it just exercises and demonstrates the live source.
/// </summary>
public sealed class StreamProbeService(
    PumpApiStreamSource source,
    IOptions<PipelineOptions> options,
    ILogger<StreamProbeService> logger) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!options.Value.StreamProbe)
        {
            return;
        }

        logger.LogInformation("stream probe enabled — logging live pumpapi events (not persisting)");
        long count = 0;
        try
        {
            await foreach (RawEvent evt in source.StreamAsync(stoppingToken))
            {
                count++;
                logger.LogInformation(
                    "event #{Count} {Kind} mint={Mint} venue={Venue} migrated={Migrated} price={Price}",
                    count, evt.Kind, evt.Mint, evt.Venue, evt.IsMigrated, evt.Price);
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown.
        }

        logger.LogInformation("stream probe stopped after {Count} events", count);
    }
}
