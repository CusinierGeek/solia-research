using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Metadata;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Metadata.MetadataFetcherTests;
using static Solia.DataPipeline.Tests.Pipeline.PipelineEngineTests;

namespace Solia.DataPipeline.Tests.Metadata;

/// <summary>FE-T12 (#196): metadata features end to end through the token store and the engine.</summary>
public class MetadataPipelineTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private static readonly PipelineOptions Options = new() { SnapshotCadenceSeconds = 10, WindowSeconds = 300 };
    private const string Uri = "https://ipfs.io/ipfs/QmCid";

    private static RawEvent Create(string mint, DateTimeOffset ts, string? uri = Uri) => new()
    {
        Mint = mint,
        Kind = EventKind.Create,
        Timestamp = ts,
        Name = "Pepe",
        Symbol = "PEPE",
        MetadataUri = uri,
        QuoteMint = RawEvent.WrappedSolMint,
        Price = 0.000001m,
    };

    private static RawEvent Buy(string mint, DateTimeOffset ts) => new()
    {
        Mint = mint,
        Kind = EventKind.Buy,
        Timestamp = ts,
        Trader = "w",
        QuoteMint = RawEvent.WrappedSolMint,
        QuoteAmount = 0.5m,
        TokenAmount = 1000m,
        Price = 0.000001m,
    };

    private sealed class RecordingQueue : IMetadataFetchQueue
    {
        public List<(string Mint, string? Uri)> Queued { get; } = [];

        public bool TryEnqueue(string mint, string? uri)
        {
            Queued.Add((mint, uri));
            return true;
        }
    }

    [Fact]
    public void Live_ingestion_hands_each_create_to_the_fetcher_and_a_replay_never_does()
    {
        var queue = new RecordingQueue();
        var live = new TokenWindowStore(Options, trackWalletChanges: true, new MetadataCache(), queue);

        live.Apply(Create("A", T0));
        live.Apply(new RawEvent { Mint = "A", Kind = EventKind.Buy, Timestamp = T0.AddSeconds(1), Trader = "w" });

        Assert.Equal([("A", Uri)], queue.Queued);
        new TokenWindowStore(Options).Apply(Create("B", T0)); // replay store: no queue at all, nothing to call
    }

    [Fact]
    public void Snapshots_carry_the_uri_kind_and_see_links_only_once_fetched()
    {
        var store = new TokenWindowStore(Options);
        store.Apply(Create("A", T0));
        store.Apply(Create("B", T0, "https://meta.uxento.io/data/1"));
        store.Apply(Buy("A", T0.AddSeconds(1)));
        store.Apply(Buy("B", T0.AddSeconds(1)));
        store.Metadata.TryAdd(new TokenMetadataRecord("A", T0.AddSeconds(15), MetadataFetchStatus.Ok, HasTwitter: true));

        MetadataFeatures before = store.SnapshotActive(T0.AddSeconds(10)).Single(s => s.Mint == "A").Metadata!;
        IReadOnlyList<FeatureSnapshot> after = store.SnapshotActive(T0.AddSeconds(20));

        Assert.True(before.UriContentAddressed);
        Assert.Null(before.HasTwitter); // fetched at +15 s: unknown to the +10 s snapshot
        MetadataFeatures a = after.Single(s => s.Mint == "A").Metadata!;
        Assert.Equal((false, true, false), (a.HasWebsite, a.HasTwitter, a.HasTelegram));
        MetadataFeatures b = after.Single(s => s.Mint == "B").Metadata!;
        Assert.False(b.UriContentAddressed);
        Assert.Null(b.HasTwitter);
        Assert.Null(b.NameMatches1h); // under an hour of history
        Assert.Equal(20, b.CoverageSec);
    }

    [Fact]
    public async Task A_replay_loads_the_results_fetched_over_its_span_and_never_fetches()
    {
        var stored = new MemoryMetadataStore();
        stored.Stored.Add(new TokenMetadataRecord("A", T0.AddSeconds(5), MetadataFetchStatus.Ok, HasWebsite: true));
        var tokens = new TokenWindowStore(Options);
        var sink = new RecordingSink();
        var engine = new PipelineEngine(new ListSource(
                [Create("A", T0), .. Enumerable.Range(1, 30).Select(i => Buy("A", T0.AddSeconds(i)))]),
            tokens, sink, Options, NullLogger<PipelineEngine>.Instance, (_, _) => Task.CompletedTask, TimeSpan.FromSeconds(5),
            metadataStore: stored)
        {
            MetadataRange = (T0.AddHours(-48), T0.AddHours(1)),
        };

        await engine.RunAsync(default);

        Assert.Equal([(T0.AddHours(-48), (DateTimeOffset?)T0.AddHours(1))], stored.Loads);
        Assert.Empty(stored.Saved);
        List<FeatureSnapshot> rows = sink.Batches.SelectMany(b => b.Snapshots).Where(s => s.Mint == "A").ToList();
        Assert.All(rows.Where(r => r.Ts < T0.AddSeconds(5)), r => Assert.Null(r.Metadata!.HasWebsite));
        Assert.All(rows.Where(r => r.Ts >= T0.AddSeconds(5)), r => Assert.True(r.Metadata!.HasWebsite));
        Assert.Contains(rows, r => r.Ts >= T0.AddSeconds(5));
    }

    [Fact]
    public async Task An_unreachable_store_at_startup_only_leaves_the_links_unknown()
    {
        var sink = new RecordingSink();
        var engine = new PipelineEngine(new ListSource(
                [Create("A", T0), .. Enumerable.Range(1, 15).Select(i => Buy("A", T0.AddSeconds(i)))]),
            new TokenWindowStore(Options), sink, Options, NullLogger<PipelineEngine>.Instance, (_, _) => Task.CompletedTask,
            TimeSpan.FromSeconds(5), metadataStore: new ThrowingStore());

        await engine.RunAsync(default);

        Assert.NotEmpty(sink.Batches);
        Assert.All(sink.Batches.SelectMany(b => b.Snapshots), s => Assert.Null(s.Metadata!.HasTwitter));
    }

    private sealed class ThrowingStore : IMetadataStore
    {
        public Task<IReadOnlyList<TokenMetadataRecord>> LoadAsync(DateTimeOffset since, DateTimeOffset? until, CancellationToken cancellationToken) =>
            throw new InvalidOperationException("database down");

        public Task SaveAsync(TokenMetadataRecord record, CancellationToken cancellationToken) => Task.CompletedTask;
    }
}
