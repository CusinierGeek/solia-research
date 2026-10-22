using Microsoft.Extensions.Options;
using Solia.DataPipeline.Logging;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline;

/// <summary>
/// The pipeline host. In <see cref="PipelineOptions.Ingest"/> mode it runs the full engine
/// (live source → features → PostgreSQL). Otherwise — including <see cref="PipelineOptions.StreamProbe"/>
/// watch mode, which owns the single live connection itself — it just heartbeats. Enabling both
/// modes is rejected by options validation at startup, so only one component ever opens the
/// (single-per-client) live connection.
/// </summary>
public sealed class Worker(
    ILogger<Worker> logger,
    ILogSink logSink,
    PipelineEngine engine,
    IOptions<PipelineOptions> options) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        PipelineOptions opt = options.Value;
        logger.LogInformation(
            "pipeline started (ingest={Ingest}, streamProbe={Probe}, cadence={Cadence}s, window={Window}s)",
            opt.Ingest, opt.StreamProbe, opt.SnapshotCadenceSeconds, opt.WindowSeconds);

        // Best-effort: don't crash the worker if the DB isn't reachable yet at startup.
        try
        {
            await logSink.WriteAsync(LogSeverity.Info, "pipeline started", new
            {
                opt.Ingest,
                opt.StreamProbe,
                opt.SnapshotCadenceSeconds,
                opt.WindowSeconds,
            }, stoppingToken);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            logger.LogInformation("pipeline stopping");
            return;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "could not write startup log to database");
        }

        if (opt.Ingest)
        {
            try
            {
                // Live ingestion starts tracking from scratch, so last run's "active" rows are stale.
                await engine.ResetActiveTokensAsync(stoppingToken);
                await engine.RunAsync(stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                // Stopped during the startup reset.
            }
            catch (Exception ex) when (!stoppingToken.IsCancellationRequested)
            {
                // Fail loudly: rethrowing stops the host instead of leaving it up but ingesting nothing.
                logger.LogCritical(ex, "pipeline engine failed");
                throw;
            }

            logger.LogInformation("pipeline stopping");
            return;
        }

        // Idle heartbeat: default run, or while the opt-in stream probe owns the live connection.
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(opt.SnapshotCadenceSeconds));
        try
        {
            while (await timer.WaitForNextTickAsync(stoppingToken))
            {
                logger.LogDebug("heartbeat");
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown.
        }

        logger.LogInformation("pipeline stopping");
    }
}
