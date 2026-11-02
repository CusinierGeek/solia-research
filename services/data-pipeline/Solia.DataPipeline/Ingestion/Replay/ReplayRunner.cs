using Microsoft.Extensions.Logging;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>Outcome of one replay (or mirror-only) run over a window.</summary>
public sealed record ReplaySummary(
    int Hours,
    IReadOnlyList<DateTimeOffset> MissingHours,
    IReadOnlyList<ReplayHourFailure> FailedHours,
    long FeatureRowsWritten,
    long FailedBatches,
    int MirrorHits,
    int MirrorDownloads)
{
    /// <summary>Hours read to the end (neither missing nor failed).</summary>
    public int ProcessedHours => Hours - MissingHours.Count - FailedHours.Count;

    /// <summary>False when an hour or a write batch was given up on: re-run the window (idempotent) to retry.</summary>
    public bool IsComplete => FailedHours.Count == 0 && FailedBatches == 0;
}

/// <summary>
/// Runs a historical replay of a window: <see cref="PumpApiReplaySource"/> → the same
/// <see cref="PipelineEngine"/> (features, sink) as live ingestion → PostgreSQL. The archive is
/// either the HTTP source directly (default: nothing kept on disk) or an <see cref="ArchiveMirror"/>.
/// Unlike live ingestion it never resets active tokens: the sink's newer-wins guards keep a replay
/// from rolling live rows back. It only marks inactive, at the end, the tokens it still tracks (as of
/// the last replayed event, so rows live ingestion has updated since are untouched).
/// Also runs mirror-only downloads (no engine, no database).
/// </summary>
public sealed class ReplayRunner
{
    private readonly ILoggerFactory _loggerFactory;
    private readonly ILogger<ReplayRunner> _logger;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;

    public ReplayRunner(ILoggerFactory loggerFactory)
        : this(loggerFactory, static (d, ct) => Task.Delay(d, ct))
    {
    }

    // Test seam: inject the retry delay so unit tests run without real waits.
    internal ReplayRunner(ILoggerFactory loggerFactory, Func<TimeSpan, CancellationToken, Task> delay)
    {
        _loggerFactory = loggerFactory;
        _logger = loggerFactory.CreateLogger<ReplayRunner>();
        _delay = delay;
    }

    /// <summary>
    /// Replays <paramref name="window"/> from <paramref name="archive"/> into <paramref name="sink"/>.
    /// <paramref name="featureRowsWritten"/> reads the sink's committed-row counter for the summary; an
    /// <see cref="ArchiveMirror"/> archive also reports its hit/download counts.
    /// Fatal source errors propagate; cancellation throws <see cref="OperationCanceledException"/>.
    /// <para>
    /// A <paramref name="warmup"/> replays that much history before the window through the same engine but writes
    /// nothing dated before the window's start (<see cref="WarmupSink"/>): the wallet memory and the market context
    /// enter the window warm. Its hours count in the summary like the window's.
    /// </para>
    /// </summary>
    public async Task<ReplaySummary> ReplayAsync(
        ReplayWindow window,
        IReplayArchive archive,
        ITokenSink sink,
        Func<long> featureRowsWritten,
        PipelineOptions options,
        CancellationToken cancellationToken,
        TimeSpan warmup = default)
    {
        ReplayWindow sourceWindow = new(window.StartUtc - warmup, window.EndUtc);
        ITokenSink target = warmup > TimeSpan.Zero ? new WarmupSink(sink, window.StartUtc) : sink;
        var source = new PumpApiReplaySource(sourceWindow, archive.OpenAsync, _loggerFactory.CreateLogger<PumpApiReplaySource>(),
            PumpApiReplaySource.DefaultMaxAttemptsPerHour, _delay);
        var engine = new PipelineEngine(source, new TokenWindowStore(options), target,
            Microsoft.Extensions.Options.Options.Create(options), _loggerFactory.CreateLogger<PipelineEngine>())
        {
            ReleaseTrackedOnStop = true,
        };

        await engine.RunAsync(cancellationToken);
        cancellationToken.ThrowIfCancellationRequested(); // the engine treats cancellation as a graceful stop

        ArchiveMirror? mirror = archive as ArchiveMirror;
        return new ReplaySummary(sourceWindow.Hours().Count(), source.MissingHours.ToList(), source.FailedHours.ToList(),
            featureRowsWritten(), engine.FailedBatches, mirror?.Hits ?? 0, mirror?.Downloads ?? 0);
    }

    /// <summary>
    /// Downloads every absent hour of <paramref name="window"/> into <paramref name="mirror"/> without
    /// replaying it. A failed hour is retried with backoff, then recorded and skipped.
    /// </summary>
    public async Task<ReplaySummary> MirrorOnlyAsync(
        ReplayWindow window,
        ArchiveMirror mirror,
        CancellationToken cancellationToken,
        int maxAttemptsPerHour = PumpApiReplaySource.DefaultMaxAttemptsPerHour)
    {
        window.EnsureValid();
        var retry = new RetryPolicy(maxAttemptsPerHour);
        var missing = new List<DateTimeOffset>();
        var failed = new List<ReplayHourFailure>();
        int hours = 0;

        foreach (DateTimeOffset hour in window.Hours())
        {
            hours++;
            Exception? last = null;
            bool done = await retry.RunAsync(async ct =>
            {
                if (await mirror.EnsureAsync(hour, ct) == MirrorOutcome.Missing)
                {
                    missing.Add(hour);
                    _logger.LogInformation("mirror: no archive for {Hour:yyyy-MM-dd HH}:00 UTC", hour);
                }
            }, _delay, (ex, attempt, wait) =>
            {
                last = ex;
                if (wait is not null)
                {
                    _logger.LogWarning(ex, "mirror: {Hour:yyyy-MM-dd HH}:00 UTC failed (attempt {Attempt}/{Max}); retrying in {Delay}",
                        hour, attempt, maxAttemptsPerHour, wait);
                }
                else
                {
                    _logger.LogWarning(ex, "mirror: skipping {Hour:yyyy-MM-dd HH}:00 UTC after {Attempts} attempts", hour, attempt);
                }
            }, cancellationToken);

            if (!done)
            {
                failed.Add(new ReplayHourFailure(hour, last!.Message));
            }
        }

        return new ReplaySummary(hours, missing, failed, 0, 0, mirror.Hits, mirror.Downloads);
    }
}
