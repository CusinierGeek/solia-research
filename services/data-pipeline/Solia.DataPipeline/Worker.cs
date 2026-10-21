using Microsoft.Extensions.Options;
using Solia.DataPipeline.Logging;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline;

/// <summary>
/// Skeleton pipeline host (Sprint 1 / T2). For now it only logs startup, heartbeats on the
/// configured cadence, and shuts down gracefully. The ingestion engine (DataSource → parser
/// → windows → features → sink) is wired here in T7.
/// </summary>
public sealed class Worker(
    ILogger<Worker> logger,
    ILogSink logSink,
    IOptions<PipelineOptions> options) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        PipelineOptions opt = options.Value;
        logger.LogInformation(
            "pipeline started (cadence={Cadence}s, window={Window}s, maxTokens={Max}, enrichHolders={Enrich})",
            opt.SnapshotCadenceSeconds, opt.WindowSeconds, opt.MaxTrackedTokens, opt.EnrichHolders);

        // Best-effort: don't crash the worker if the DB isn't reachable yet at startup.
        try
        {
            await logSink.WriteAsync(LogSeverity.Info, "pipeline started", new
            {
                opt.SnapshotCadenceSeconds,
                opt.WindowSeconds,
                opt.MaxTrackedTokens,
                opt.EnrichHolders,
            }, stoppingToken);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Stopped while the write was in flight: a normal shutdown, not a DB problem.
            logger.LogInformation("pipeline stopping");
            return;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "could not write startup log to database");
        }

        // PipelineOptions is validated at startup, so the cadence is at least 1s here.
        TimeSpan period = TimeSpan.FromSeconds(opt.SnapshotCadenceSeconds);
        using var timer = new PeriodicTimer(period);
        try
        {
            while (await timer.WaitForNextTickAsync(stoppingToken))
            {
                logger.LogDebug("heartbeat");
            }
        }
        catch (OperationCanceledException)
        {
            // Normal shutdown path.
        }

        logger.LogInformation("pipeline stopping");
    }
}
