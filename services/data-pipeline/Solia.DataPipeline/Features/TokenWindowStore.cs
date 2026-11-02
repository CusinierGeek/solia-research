using Microsoft.Extensions.Options;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Metadata;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Holds the rolling <see cref="TokenState"/> for every tracked token. Collapses the firehose
/// into a bounded working set: a token starts being tracked only on a market event (create /
/// trade / pool / liquidity / migrate) — a <see cref="EventKind.Transfer"/> or
/// <see cref="EventKind.Other"/> only updates a token that is already tracked, so transfer noise
/// never creates entries. Enforces <see cref="PipelineOptions.MaxTrackedTokens"/> by first
/// dropping idle tokens, then evicting the least recently updated (O(1), via a recency list), and
/// prunes tokens idle beyond the window. A token whose newest event is dated more than
/// <see cref="MaxFutureSkew"/> past the snapshot time is pruned too, so a bad timestamp can't pin it
/// in memory (it is re-tracked cleanly on its next real event).
/// Produces feature snapshots on demand (driven by T7).
/// Not thread-safe: the pipeline applies events and snapshots from a single loop.
/// </summary>
public sealed class TokenWindowStore
{
    internal static readonly TimeSpan MaxFutureSkew = TimeSpan.FromSeconds(60);

    private static readonly HashSet<EventKind> TrackingKinds =
    [
        EventKind.Create, EventKind.Buy, EventKind.Sell, EventKind.Migrate,
        EventKind.CreatePool, EventKind.AddLiquidity, EventKind.RemoveLiquidity,
    ];

    private readonly PipelineOptions _options;
    private readonly Dictionary<string, LinkedListNode<TokenState>> _tokens = new(StringComparer.Ordinal);
    private readonly LinkedList<TokenState> _byRecency = new(); // least recently updated first
    private readonly HashSet<string> _removed = new(StringComparer.Ordinal);
    private readonly MarketContext _market = new();
    private readonly WalletStore _wallets;
    private readonly NarrativeTracker _narrative = new();
    private readonly MetadataCache _metadata;
    private readonly IMetadataFetchQueue? _fetchQueue;
    private DateTimeOffset? _lastMetadataPrune;

    /// <summary>Live ingestion: wallet changes are persisted, new tokens are handed to the metadata fetcher.</summary>
    public TokenWindowStore(IOptions<PipelineOptions> options, MetadataCache metadata, IMetadataFetchQueue fetchQueue)
        : this(options.Value, trackWalletChanges: true, metadata, fetchQueue)
    {
    }

    /// <param name="options">Pipeline options.</param>
    /// <param name="trackWalletChanges">
    /// Collect wallet changes for persistence (live ingestion, see <see cref="WalletStore"/>); a replay keeps its
    /// wallet state in memory only.
    /// </param>
    /// <param name="metadata">The metadata fetch results (FE-T12); a fresh empty cache when null.</param>
    /// <param name="fetchQueue">Live ingestion only: where new tokens go to be fetched. A replay never fetches.</param>
    internal TokenWindowStore(
        PipelineOptions options,
        bool trackWalletChanges = false,
        MetadataCache? metadata = null,
        IMetadataFetchQueue? fetchQueue = null)
    {
        _options = options;
        _wallets = new WalletStore(options, trackWalletChanges);
        _metadata = metadata ?? new MetadataCache();
        _fetchQueue = fetchQueue;
    }

    public int TrackedCount => _tokens.Count;

    /// <summary>Mints currently tracked.</summary>
    public IReadOnlyList<string> TrackedMints() => _tokens.Keys.ToList();

    /// <summary>
    /// Mints pruned or evicted since the last call that aren't tracked again, so the hot table
    /// can mark them inactive. Each mint is reported once.
    /// </summary>
    public IReadOnlyList<string> DrainRemoved()
    {
        List<string> removed = _removed.Where(mint => !_tokens.ContainsKey(mint)).ToList();
        _removed.Clear();
        return removed;
    }

    /// <summary>Market-wide aggregates (fed every event, tracked token or not).</summary>
    internal MarketContext Market => _market;

    /// <summary>Cross-token wallet memory (fed every event, tracked token or not).</summary>
    public WalletStore Wallets => _wallets;

    /// <summary>The per-token metadata fetch results the snapshots read (FE-T12).</summary>
    public MetadataCache Metadata => _metadata;

    /// <summary>Narrative heat over normalized names / tickers (fed every event, tracked token or not).</summary>
    internal NarrativeTracker Narrative => _narrative;

    public void Apply(RawEvent e)
    {
        _market.Apply(e);
        _wallets.Apply(e);
        _narrative.Apply(e);
        if (e.Kind == EventKind.Create)
        {
            _fetchQueue?.TryEnqueue(e.Mint, e.MetadataUri); // non-blocking; the fetch runs off the loop
        }

        if (_tokens.TryGetValue(e.Mint, out LinkedListNode<TokenState>? node))
        {
            node.Value.Apply(e);
            _byRecency.Remove(node);
            _byRecency.AddLast(node);
            return;
        }

        if (!TrackingKinds.Contains(e.Kind))
        {
            return;
        }

        MakeRoom(e.Timestamp);
        var state = new TokenState(e.Mint, _options.WindowSeconds, _options.EmaSpan, _options.RsiPeriod, e.Timestamp, _options.LargeBuySol, _options.Momentum, _options.ReputationEarlyBuyers,
            _options.HolderLargeShare, _options.HolderMaxWallets);
        state.Apply(e);
        _tokens[e.Mint] = _byRecency.AddLast(state);
    }

    /// <summary>
    /// Evicts idle tokens, then returns one snapshot per token still active in the window. Eager:
    /// each call snapshots exactly once (snapshots advance each token's growth-rate baseline). The
    /// market-wide aggregates are computed once for the tick and shared by every snapshot; the wallet memory is
    /// advanced to <paramref name="now"/> (events strictly before it) before any lookup.
    /// </summary>
    public IReadOnlyList<FeatureSnapshot> SnapshotActive(DateTimeOffset now)
    {
        Prune(now);
        MarketAggregates market = _market.Aggregate(now);
        _wallets.Advance(now);
        _narrative.Prune(now);
        PruneMetadata(now);

        var snapshots = new List<FeatureSnapshot>();
        foreach (TokenState state in _byRecency)
        {
            if (state.HasActivityInWindow(now))
            {
                snapshots.Add(state.Snapshot(now, _wallets) with { Market = market, Metadata = MetadataOf(state, now) });
            }
        }

        return snapshots;
    }

    private MetadataFeatures MetadataOf(TokenState state, DateTimeOffset now)
    {
        (int coverage, int? matches, int? graduations) = _narrative.Count(state.NameKeys, state.Mint, now);
        MetadataLinks? links = _metadata.Lookup(state.Mint, now);
        return new MetadataFeatures
        {
            CoverageSec = coverage,
            UriContentAddressed = state.UriContentAddressed,
            HasWebsite = links?.HasWebsite,
            HasTwitter = links?.HasTwitter,
            HasTelegram = links?.HasTelegram,
            NameMatches1h = matches,
            NarrativeGraduations = graduations,
        };
    }

    /// <summary>Forgets fetch results past their horizon, at most hourly (no snapshot can see them any more).</summary>
    private void PruneMetadata(DateTimeOffset now)
    {
        if (_lastMetadataPrune is { } last && now - last < TimeSpan.FromHours(1))
        {
            return;
        }

        _lastMetadataPrune = now;
        _metadata.Prune(now);
    }

    /// <summary>Removes tokens with no activity within the window, or whose activity is implausibly far in the future.</summary>
    public void Prune(DateTimeOffset now)
    {
        DateTimeOffset cutoff = now - TimeSpan.FromSeconds(_options.WindowSeconds);
        DateTimeOffset future = now + MaxFutureSkew;

        LinkedListNode<TokenState>? node = _byRecency.First;
        while (node is not null)
        {
            LinkedListNode<TokenState>? next = node.Next;
            DateTimeOffset last = node.Value.LastActivity;
            if (last < cutoff || last > future)
            {
                Remove(node);
            }

            node = next;
        }
    }

    /// <summary>At capacity, drop idle tokens from the cold end first; evict the coldest live one only if still full.</summary>
    private void MakeRoom(DateTimeOffset now)
    {
        if (_tokens.Count < _options.MaxTrackedTokens)
        {
            return;
        }

        DateTimeOffset cutoff = now - TimeSpan.FromSeconds(_options.WindowSeconds);
        while (_byRecency.First is { } coldest && coldest.Value.LastActivity < cutoff)
        {
            Remove(coldest);
        }

        if (_tokens.Count >= _options.MaxTrackedTokens && _byRecency.First is { } victim)
        {
            Remove(victim);
        }
    }

    private void Remove(LinkedListNode<TokenState> node)
    {
        _tokens.Remove(node.Value.Mint);
        _byRecency.Remove(node);
        _removed.Add(node.Value.Mint);
    }
}
