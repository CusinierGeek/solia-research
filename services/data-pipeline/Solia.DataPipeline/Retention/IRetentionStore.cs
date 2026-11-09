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

/// <summary>
/// Which <c>features</c> rows past the general TTL are kept: those of a token that has a row aged at least
/// <paramref name="Age"/> (<c>token_age_sec</c>) between the row's time and <paramref name="Age"/> later, i.e. whose
/// activity reached that age. Of those, rows older than <paramref name="FullResolution"/> keep only the first row of
/// each <paramref name="Sample"/> bucket (<see cref="TimeSpan.Zero"/>: every row).
/// </summary>
public sealed record EstablishedRule(TimeSpan Age, TimeSpan FullResolution, TimeSpan Sample);

/// <summary>A position in <c>features</c> by <c>(ts, id)</c>: the next scan starts strictly after it.</summary>
public readonly record struct FeaturesCursor(DateTimeOffset Ts, long Id)
{
    /// <summary>Before every row at or after <paramref name="ts"/> (ids are positive).</summary>
    public static FeaturesCursor At(DateTimeOffset ts) => new(ts, 0);
}

/// <summary>One batch of the established band: rows scanned, the last one scanned (null if none), rows removed.</summary>
public readonly record struct BandBatch(long Scanned, FeaturesCursor? Last, PurgeCount Removed);

/// <summary>The database side of the purge, one table at a time.</summary>
public interface IRetentionStore
{
    /// <summary>
    /// Scans the next <paramref name="batchSize"/> <c>features</c> rows after <paramref name="after"/> with
    /// <c>ts</c> before <paramref name="cutoff"/>, in <c>(ts, id)</c> order, and removes those
    /// <paramref name="rule"/> doesn't keep (only counts them when <paramref name="dryRun"/>). One short transaction.
    /// </summary>
    Task<BandBatch> PurgeEstablishedBandBatchAsync(
        FeaturesCursor after, DateTimeOffset cutoff, EstablishedRule rule, int batchSize, bool dryRun, CancellationToken ct);

    /// <summary>What a purge would remove: rows with <c>ts</c> strictly before <paramref name="cutoff"/>.</summary>
    Task<PurgeCount> CountExpiredAsync(RetainedTable table, DateTimeOffset cutoff, CancellationToken ct);

    /// <summary>Deletes at most <paramref name="batchSize"/> of the oldest expired rows in one short transaction.</summary>
    Task<PurgeCount> DeleteExpiredBatchAsync(RetainedTable table, DateTimeOffset cutoff, int batchSize, CancellationToken ct);

    /// <summary>Makes the deleted rows' space reusable (plain VACUUM: no exclusive lock, readers and writers carry on).</summary>
    Task VacuumAsync(RetainedTable table, CancellationToken ct);
}
