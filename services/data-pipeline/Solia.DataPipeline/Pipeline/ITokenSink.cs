using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// One flush: the feature snapshots taken at <see cref="Ts"/> and the mints that stopped being
/// tracked since the previous flush (pruned or evicted), so the hot table can mark them inactive.
/// </summary>
public sealed record FlushBatch(DateTimeOffset Ts, IReadOnlyList<FeatureSnapshot> Snapshots, IReadOnlyList<string> Deactivated)
{
    public bool IsEmpty => Snapshots.Count == 0 && Deactivated.Count == 0;
}

/// <summary>
/// Persists a flush (feature snapshots, the token rows they imply, and deactivations).
/// Abstracted so the engine can be unit-tested with an in-memory sink and integration-tested
/// against PostgreSQL.
/// </summary>
public interface ITokenSink
{
    Task WriteAsync(FlushBatch batch, CancellationToken cancellationToken);

    /// <summary>
    /// Marks every token in the hot table inactive. Called once when live ingestion starts: a fresh
    /// process tracks nothing yet, so rows left active by a previous run are stale. Tokens become
    /// active again as soon as their next snapshot is written.
    /// </summary>
    Task DeactivateAllAsync(CancellationToken cancellationToken);
}
