using Solia.DataPipeline.Retention;
using Solia.Shared.Logging;

namespace Solia.DataPipeline.Tests.Retention;

/// <summary>An in-memory store: each table holds a number of expired rows, deleted oldest first in batches.</summary>
internal sealed class FakeRetentionStore : IRetentionStore
{
    private static readonly DateTimeOffset Epoch = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly Dictionary<RetainedTable, long> _expired = new();

    public Dictionary<RetainedTable, DateTimeOffset> Cutoffs { get; } = new();

    public List<(RetainedTable Table, int BatchSize)> Deletes { get; } = [];

    public List<RetainedTable> Vacuums { get; } = [];

    public HashSet<RetainedTable> Failing { get; } = [];

    public bool Unreachable { get; set; }

    /// <summary>Rows each table still has expired (<c>ts</c> = Epoch + i minutes for the i-th row).</summary>
    public long Remaining(RetainedTable table) => _expired.GetValueOrDefault(table);

    public FakeRetentionStore With(RetainedTable table, long expiredRows)
    {
        _expired[table] = expiredRows;
        return this;
    }

    public Task<PurgeCount> CountExpiredAsync(RetainedTable table, DateTimeOffset cutoff, CancellationToken ct)
    {
        Touch(table, cutoff);
        long n = Remaining(table);
        return Task.FromResult(n == 0 ? PurgeCount.None : new PurgeCount(n, Epoch, Epoch.AddMinutes(n - 1)));
    }

    public Task<PurgeCount> DeleteExpiredBatchAsync(RetainedTable table, DateTimeOffset cutoff, int batchSize, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Touch(table, cutoff);
        Deletes.Add((table, batchSize));
        long before = Remaining(table);
        long n = Math.Min(batchSize, before);
        _expired[table] = before - n;
        return Task.FromResult(n == 0
            ? PurgeCount.None
            : new PurgeCount(n, Epoch, Epoch.AddMinutes(n - 1)));
    }

    /// <summary>Rows of the established band, by (ts, id); <c>Doomed</c> = not kept by the rule.</summary>
    public List<(DateTimeOffset Ts, long Id, bool Doomed)> Band { get; } = [];

    public List<(FeaturesCursor After, DateTimeOffset Cutoff, EstablishedRule Rule, int BatchSize, bool DryRun)> BandCalls { get; } = [];

    public FakeRetentionStore WithBand(DateTimeOffset start, int rows, Func<int, bool> doomed)
    {
        for (int i = 0; i < rows; i++)
        {
            Band.Add((start.AddMinutes(i), i + 1, doomed(i)));
        }

        return this;
    }

    public Task<BandBatch> PurgeEstablishedBandBatchAsync(
        FeaturesCursor after, DateTimeOffset cutoff, EstablishedRule rule, int batchSize, bool dryRun, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Touch(RetainedTable.Features, cutoff);
        BandCalls.Add((after, cutoff, rule, batchSize, dryRun));
        var scanned = Band
            .Where(r => r.Ts >= after.Ts && r.Ts < cutoff && (r.Ts > after.Ts || r.Id > after.Id))
            .OrderBy(r => r.Ts).ThenBy(r => r.Id)
            .Take(batchSize)
            .ToList();
        var doomed = scanned.Where(r => r.Doomed).ToList();
        if (!dryRun)
        {
            Band.RemoveAll(doomed.Contains);
        }

        PurgeCount removed = doomed.Count == 0 ? PurgeCount.None : new PurgeCount(doomed.Count, doomed[0].Ts, doomed[^1].Ts);
        FeaturesCursor? last = scanned.Count == 0 ? null : new FeaturesCursor(scanned[^1].Ts, scanned[^1].Id);
        return Task.FromResult(new BandBatch(scanned.Count, last, removed));
    }

    public Task VacuumAsync(RetainedTable table, CancellationToken ct)
    {
        Vacuums.Add(table);
        return Task.CompletedTask;
    }

    private void Touch(RetainedTable table, DateTimeOffset cutoff)
    {
        if (Unreachable)
        {
            throw new InvalidOperationException("database unreachable");
        }

        if (Failing.Contains(table))
        {
            throw new InvalidOperationException($"{table} failed");
        }

        Cutoffs[table] = cutoff;
    }
}

internal sealed class RecordingLogSink : ILogSink
{
    public List<(LogSeverity Level, string Message, object? Context)> Entries { get; } = [];

    public bool Throws { get; set; }

    public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
    {
        if (Throws)
        {
            throw new InvalidOperationException("logs table unreachable");
        }

        Entries.Add((level, message, context));
        return Task.CompletedTask;
    }
}
