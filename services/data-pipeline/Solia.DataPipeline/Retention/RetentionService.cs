using Microsoft.Extensions.Options;

namespace Solia.DataPipeline.Retention;

/// <summary>
/// Runs <see cref="RetentionJob"/> at startup and then every <see cref="RetentionOptions.Interval"/>, when
/// <see cref="RetentionOptions.Enabled"/>. A failed run is logged and retried on the next tick; it never stops the
/// host. Independent of ingestion: it uses its own short-lived connections from the pool.
/// </summary>
public sealed class RetentionService(
    RetentionJob job,
    IOptions<RetentionOptions> options,
    TimeProvider time,
    ILogger<RetentionService> logger) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        RetentionOptions opt = options.Value;
        if (!opt.Enabled)
        {
            logger.LogInformation("retention disabled (Retention:Enabled / RETENTION_ENABLED): nothing is ever deleted");
            return;
        }

        logger.LogInformation(
            "retention enabled: every {Interval} min, features {Features} d, logs {Logs} d, predictions {Predictions} d, batches of {Batch}",
            opt.IntervalMinutes, opt.FeaturesTtlDays, opt.LogsTtlDays, opt.PredictionsTtlDays, opt.BatchSize);

        using var timer = new PeriodicTimer(opt.Interval, time);
        try
        {
            do
            {
                try
                {
                    await job.RunAsync(dryRun: false, stoppingToken);
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "retention run failed; retrying in {Interval} min", opt.IntervalMinutes);
                }
            }
            while (await timer.WaitForNextTickAsync(stoppingToken));
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Normal shutdown; an interrupted batch was its own transaction, the rest waits for the next start.
        }
    }
}
