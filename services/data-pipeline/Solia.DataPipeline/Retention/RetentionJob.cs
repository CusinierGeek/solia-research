using System.Diagnostics;
using Microsoft.Extensions.Options;
using Solia.Shared.Logging;

namespace Solia.DataPipeline.Retention;

/// <summary>One table's outcome: what was removed (or would be, in a dry run), how long it took, and the failure if any.</summary>
/// <remarks>
/// For <c>features</c> with established tokens kept, <see cref="EstablishedCutoff"/> is the older cutoff before which
/// every row goes; between it and <see cref="Cutoff"/> only the rows <see cref="EstablishedRule"/> doesn't keep go.
/// </remarks>
public sealed record TablePurge(
    RetainedTable Table, DateTimeOffset Cutoff, PurgeCount Removed, TimeSpan Duration, string? Error = null,
    DateTimeOffset? EstablishedCutoff = null)
{
    public bool Failed => Error is not null;
}

public sealed record RetentionReport(bool DryRun, IReadOnlyList<TablePurge> Tables)
{
    public bool Failed => Tables.Any(t => t.Failed);
}

/// <summary>
/// One purge of every <see cref="RetainedTable"/>: deletes the rows older than each table's TTL in bounded
/// batches, then vacuums the tables it shrank so the space is reused. A table that fails (e.g. database
/// unreachable) is reported and skipped — never fatal; what it didn't delete is picked up by the next run.
/// The outcome is logged per table and mirrored to the <c>logs</c> table (best-effort).
/// <para>
/// <c>features</c> of established tokens (<see cref="RetentionOptions.KeepsEstablished"/>): every row older than
/// <see cref="RetentionOptions.EstablishedTtlDays"/> goes; the band between that and the general TTL is scanned in
/// <c>(ts, id)</c> order and loses the rows <see cref="EstablishedRule"/> doesn't keep. Once a run has scanned the
/// band up to its cutoff, everything before is kept rows only, so the next run resumes there (in memory) instead of
/// re-reading weeks of kept rows; a fresh process starts <see cref="BandStartupLookback"/> before the cutoff.
/// </para>
/// </summary>
public sealed class RetentionJob(
    IRetentionStore store,
    ILogSink logSink,
    IOptions<RetentionOptions> options,
    TimeProvider time,
    ILogger<RetentionJob> logger)
{
    /// <summary>
    /// How far before the general features cutoff a fresh process starts scanning the established band. Rows left there
    /// by an outage longer than this (or written by a replay older than the cutoff, see <c>--allow-expired</c>) wait
    /// for the established TTL.
    /// </summary>
    public static readonly TimeSpan BandStartupLookback = TimeSpan.FromDays(3);

    /// <summary>Where the band is known clean up to (the general cutoff of the last complete, non-dry run).</summary>
    private DateTimeOffset? _bandScannedUntil;

    public async Task<RetentionReport> RunAsync(bool dryRun, CancellationToken ct)
    {
        RetentionOptions opt = options.Value;
        DateTimeOffset now = time.GetUtcNow();
        var tables = new List<TablePurge>();
        foreach (RetainedTable table in RetainedTables.All)
        {
            TablePurge purge = table == RetainedTable.Features && opt.KeepsEstablished
                ? await PurgeFeaturesAsync(opt, now, dryRun, ct)
                : await PurgeAsync(table, now - opt.TtlOf(table), opt.BatchSize, dryRun, ct);
            tables.Add(purge);
            Log(purge, dryRun);
        }

        var report = new RetentionReport(dryRun, tables);
        await WriteSummaryAsync(report, ct);
        return report;
    }

    private async Task<TablePurge> PurgeAsync(
        RetainedTable table, DateTimeOffset cutoff, int batchSize, bool dryRun, CancellationToken ct)
    {
        var clock = Stopwatch.StartNew();
        PurgeCount removed = PurgeCount.None;
        try
        {
            if (dryRun)
            {
                return new TablePurge(table, cutoff, await store.CountExpiredAsync(table, cutoff, ct), clock.Elapsed);
            }

            PurgeCount batch;
            do
            {
                batch = await store.DeleteExpiredBatchAsync(table, cutoff, batchSize, ct);
                removed = removed.Add(batch);
            }
            while (batch.Rows >= batchSize);

            if (removed.Rows > 0)
            {
                await store.VacuumAsync(table, ct);
            }

            return new TablePurge(table, cutoff, removed, clock.Elapsed);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "retention: purging {Table} failed after removing {Rows} rows; the next run retries",
                table.SqlName(), removed.Rows);
            return new TablePurge(table, cutoff, removed, clock.Elapsed, ex.Message);
        }
    }

    private async Task<TablePurge> PurgeFeaturesAsync(RetentionOptions opt, DateTimeOffset now, bool dryRun, CancellationToken ct)
    {
        const RetainedTable table = RetainedTable.Features;
        DateTimeOffset cutoff = now - opt.TtlOf(table);
        DateTimeOffset establishedCutoff = now - opt.EstablishedTtl;
        var clock = Stopwatch.StartNew();
        PurgeCount removed = PurgeCount.None;
        try
        {
            if (dryRun)
            {
                removed = await store.CountExpiredAsync(table, establishedCutoff, ct);
            }
            else
            {
                PurgeCount batch;
                do
                {
                    batch = await store.DeleteExpiredBatchAsync(table, establishedCutoff, opt.BatchSize, ct);
                    removed = removed.Add(batch);
                }
                while (batch.Rows >= opt.BatchSize);
            }

            DateTimeOffset from = Max(establishedCutoff, _bandScannedUntil ?? cutoff - BandStartupLookback);
            var after = FeaturesCursor.At(from);
            BandBatch band;
            do
            {
                band = await store.PurgeEstablishedBandBatchAsync(after, cutoff, opt.Established, opt.BatchSize, dryRun, ct);
                removed = removed.Add(band.Removed);
                after = band.Last ?? after;
            }
            while (band.Scanned >= opt.BatchSize);

            if (!dryRun)
            {
                _bandScannedUntil = cutoff;
                if (removed.Rows > 0)
                {
                    await store.VacuumAsync(table, ct);
                }
            }

            return new TablePurge(table, cutoff, removed, clock.Elapsed, EstablishedCutoff: establishedCutoff);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "retention: purging {Table} failed after removing {Rows} rows; the next run retries",
                table.SqlName(), removed.Rows);
            return new TablePurge(table, cutoff, removed, clock.Elapsed, ex.Message, establishedCutoff);
        }
    }

    private static DateTimeOffset Max(DateTimeOffset a, DateTimeOffset b) => a > b ? a : b;

    private void Log(TablePurge p, bool dryRun) =>
        logger.Log(p.Failed ? LogLevel.Warning : LogLevel.Information,
            "retention{DryRun}: {Table} {Verb} {Rows} rows older than {Cutoff:O}{Established} (span {Oldest:O} .. {Newest:O}) in {Duration} ms{Error}",
            dryRun ? " (dry run)" : "", p.Table.SqlName(), dryRun ? "would remove" : "removed", p.Removed.Rows, p.Cutoff,
            p.EstablishedCutoff is { } e ? $" (established tokens' kept rows: older than {e:O})" : "",
            p.Removed.Oldest, p.Removed.Newest, (long)p.Duration.TotalMilliseconds, p.Failed ? $" — failed: {p.Error}" : "");

    private async Task WriteSummaryAsync(RetentionReport report, CancellationToken ct)
    {
        try
        {
            await logSink.WriteAsync(report.Failed ? LogSeverity.Warn : LogSeverity.Info,
                report.DryRun ? "retention dry run" : report.Failed ? "retention run failed; retrying next run" : "retention run",
                new
                {
                    report.DryRun,
                    Tables = report.Tables.Select(t => new
                    {
                        Table = t.Table.SqlName(),
                        t.Cutoff,
                        t.EstablishedCutoff,
                        t.Removed.Rows,
                        t.Removed.Oldest,
                        t.Removed.Newest,
                        DurationMs = (long)t.Duration.TotalMilliseconds,
                        t.Error,
                    }),
                }, ct);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "could not write the retention summary to the database");
        }
    }
}
