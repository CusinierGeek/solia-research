namespace Solia.DataPipeline.Retention;

/// <summary>Rows matched or removed, and the time span (<c>ts</c>) they covered; null bounds when there were none.</summary>
public readonly record struct PurgeCount(long Rows, DateTimeOffset? Oldest, DateTimeOffset? Newest)
{
    public static PurgeCount None => default;

    public PurgeCount Add(PurgeCount other) => new(
        Rows + other.Rows,
        Min(Oldest, other.Oldest),
        Max(Newest, other.Newest));

    private static DateTimeOffset? Min(DateTimeOffset? a, DateTimeOffset? b) => a is null ? b : b is null ? a : a < b ? a : b;

    private static DateTimeOffset? Max(DateTimeOffset? a, DateTimeOffset? b) => a is null ? b : b is null ? a : a > b ? a : b;
}

/// <summary>The database side of the purge, one table at a time.</summary>
public interface IRetentionStore
{
    /// <summary>What a purge would remove: rows with <c>ts</c> strictly before <paramref name="cutoff"/>.</summary>
    Task<PurgeCount> CountExpiredAsync(RetainedTable table, DateTimeOffset cutoff, CancellationToken ct);

    /// <summary>Deletes at most <paramref name="batchSize"/> of the oldest expired rows in one short transaction.</summary>
    Task<PurgeCount> DeleteExpiredBatchAsync(RetainedTable table, DateTimeOffset cutoff, int batchSize, CancellationToken ct);

    /// <summary>Makes the deleted rows' space reusable (plain VACUUM: no exclusive lock, readers and writers carry on).</summary>
    Task VacuumAsync(RetainedTable table, CancellationToken ct);
}
