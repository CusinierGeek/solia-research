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
