using System.Collections.Concurrent;

namespace Solia.DataPipeline.Metadata;

/// <summary>
/// The per-token metadata fetch results (FE-T12), in memory: loaded from <c>token_metadata</c> at startup (live and
/// replay), filled by <see cref="MetadataFetcher"/> (live only) and read by the snapshots. Thread-safe: the fetcher
/// writes from its own tasks while the pipeline loop reads. First result wins: a token is fetched at most once.
/// </summary>
public sealed class MetadataCache(TimeSpan horizon)
{
    private readonly ConcurrentDictionary<string, TokenMetadataRecord> _records = new(StringComparer.Ordinal);

    public MetadataCache()
        : this(new MetadataOptions().LinkHorizon)
    {
    }

    /// <summary>How long after its fetch a result stays visible (<see cref="MetadataOptions.LinkHorizonHours"/>).</summary>
    public TimeSpan Horizon { get; } = horizon;

    public int Count => _records.Count;

    public bool Contains(string mint) => _records.ContainsKey(mint);

    /// <summary>Adds a result unless the token already has one; false when it had.</summary>
    public bool TryAdd(TokenMetadataRecord record) => _records.TryAdd(record.Mint, record);

    public void Load(IEnumerable<TokenMetadataRecord> records)
    {
        foreach (TokenMetadataRecord record in records)
        {
            TryAdd(record);
        }
    }

    /// <summary>The links of <paramref name="mint"/> visible to a snapshot at <paramref name="ts"/> (see <see cref="TokenMetadataRecord.VisibleAt"/>).</summary>
    public MetadataLinks? Lookup(string mint, DateTimeOffset ts) =>
        _records.TryGetValue(mint, out TokenMetadataRecord? record) ? record.VisibleAt(ts, Horizon) : null;

    /// <summary>Forgets results fetched at least <see cref="Horizon"/> before <paramref name="now"/>: no later snapshot can see them.</summary>
    public int Prune(DateTimeOffset now)
    {
        int removed = 0;
        foreach ((string mint, TokenMetadataRecord record) in _records)
        {
            if (now - record.FetchedAt >= Horizon && _records.TryRemove(mint, out _))
            {
                removed++;
            }
        }

        return removed;
    }
}

/// <summary>Where live ingestion hands new tokens to be fetched. Must never block the pipeline loop.</summary>
public interface IMetadataFetchQueue
{
    /// <summary>Queues <paramref name="mint"/>'s metadata for a single fetch; false when skipped (disabled, known, queued or full).</summary>
    bool TryEnqueue(string mint, string? uri);
}
