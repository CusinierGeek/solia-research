using System.Diagnostics;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Hosting;

/// <summary>
/// Opt-in observability: when <see cref="PipelineOptions.StreamProbe"/> is set, consumes the
/// live pumpapi source, logs each event at Debug and a running count at Information every
/// <see cref="SummaryInterval"/> (the firehose is far too busy to log every event at
/// Information). Lets us watch real Pump.fun events and verify reconnect recovery. A no-op
/// otherwise. This is NOT the production ingestion path (that's the T7 engine) — it just
/// exercises and demonstrates the live source.
/// </summary>
public sealed class StreamProbeService(
    PumpApiStreamSource source,
    IOptions<PipelineOptions> options,
    ILogger<StreamProbeService> logger) : BackgroundService
{
    internal TimeSpan SummaryInterval { get; init; } = TimeSpan.FromSeconds(30);

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!options.Value.StreamProbe)
        {
            return;
        }

        logger.LogInformation("stream probe enabled — logging live pumpapi events (not persisting)");
        long count = 0;
        var sinceSummary = Stopwatch.StartNew();
        try
        {
            await foreach (RawEvent evt in source.StreamAsync(stoppingToken))
            {
                count++;
                logger.LogDebug(
                    "event #{Count} {Kind} mint={Mint} venue={Venue} migrated={Migrated} price={Price}",
                    count, evt.Kind, evt.Mint, evt.Venue, evt.IsMigrated, evt.Price);

                if (sinceSummary.Elapsed >= SummaryInterval)
                {
                    logger.LogInformation("stream probe: {Count} events so far", count);
                    sinceSummary.Restart();
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown.
        }

        logger.LogInformation("stream probe stopped after {Count} events", count);
    }
}
