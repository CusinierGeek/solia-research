using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Features.WalletStoreTests;
using static Solia.DataPipeline.Tests.Pipeline.PipelineEngineTests;

namespace Solia.DataPipeline.Tests.Pipeline;

/// <summary>FE-T9 (#189): the engine loads and saves the wallet memory (live), never per trade.</summary>
public class WalletPersistenceTests
{
    private static readonly PipelineOptions Options = new()
    {
        SnapshotCadenceSeconds = 10,
        WindowSeconds = 300,
        WalletPersistSeconds = 60,
        WalletTtlHours = 48,
    };

    private static IEnumerable<RawEvent> Trades(string mint, int fromSec, int toSec, string trader = "w") =>
        Enumerable.Range(fromSec, toSec - fromSec + 1).Select(i => Trade(mint, $"{trader}{i % 3}", T0.AddSeconds(i)));

    private static PipelineEngine Engine(IEnumerable<RawEvent> events, IWalletStateStore? store, TokenWindowStore? tokens = null,
        ITokenSink? sink = null) =>
        new(new ListSource(events), tokens ?? new TokenWindowStore(Options, trackWalletChanges: true), sink ?? new RecordingSink(),
            Options, NullLogger<PipelineEngine>.Instance, (_, _) => Task.CompletedTask, TimeSpan.FromSeconds(5), store);

    [Fact]
    public async Task Saves_in_batches_on_the_persist_cadence_and_at_shutdown()
    {
        var store = new MemoryWalletStateStore();

        await Engine(Trades("A", 0, 150), store).RunAsync(default);

        // Flushes every 10 s; a save once 60 s of event time have passed since the first flush (at 70 and 130),
        // plus the final one: never one per trade.
        Assert.Equal(3, store.Saves.Count);
        Assert.Equal(1, store.Loads);
        Assert.Equal(3, store.Rows.Count); // w0, w1, w2
        Assert.Equal(T0, store.HistoryStart);
    }

    [Fact]
    public async Task A_restart_restores_the_wallets_instead_of_resetting_them_to_fresh()
    {
        var store = new MemoryWalletStateStore();
        await Engine([Trade("A", "veteran", T0), .. Trades("A", 1, 30, "x")], store).RunAsync(default);

        // Second run: same durable store, a fresh process (new token store).
        var tokens = new TokenWindowStore(Options, trackWalletChanges: true);
        await Engine(Trades("B", 40, 50, "y"), store, tokens).RunAsync(default);

        WalletRecord veteran = tokens.Wallets.Get("veteran", T0.AddSeconds(60))!;
        Assert.Equal(T0.ToUnixTimeSeconds(), veteran.FirstSeen);
        Assert.Equal(T0, store.HistoryStart); // the coverage survived the restart
    }

    [Fact]
    public async Task A_failed_save_is_retried_with_the_next_one()
    {
        var store = new MemoryWalletStateStore { FailSaves = 3 }; // the first save exhausts its 3 attempts

        await Engine(Trades("A", 0, 150), store).RunAsync(default);

        Assert.Equal(3, store.Rows.Count); // nothing lost: the failed batch went out with a later save
    }

    [Fact]
    public async Task An_unreachable_store_at_startup_starts_from_an_empty_memory()
    {
        var store = new MemoryWalletStateStore { FailLoads = true };
        var sink = new RecordingSink();

        await Engine(Trades("A", 0, 30), store, sink: sink).RunAsync(default);

        Assert.NotEmpty(sink.Batches); // ingestion went on
    }

    [Fact]
    public async Task Without_change_tracking_nothing_is_loaded_or_saved()
    {
        var store = new MemoryWalletStateStore();

        // A replay's token store doesn't track wallet changes: the engine ignores the durable store.
        await Engine(Trades("A", 0, 150), store, new TokenWindowStore(Options)).RunAsync(default);

        Assert.Equal(0, store.Loads);
        Assert.Empty(store.Saves);
    }

    [Fact]
    public void Live_ingestion_tracks_wallet_changes_and_replay_does_not()
    {
        Assert.True(new TokenWindowStore(Microsoft.Extensions.Options.Options.Create(Options)).Wallets.TracksChanges);
        Assert.False(new TokenWindowStore(Options).Wallets.TracksChanges);
    }

    internal sealed class MemoryWalletStateStore : IWalletStateStore
    {
        public Dictionary<string, WalletRecord> Rows { get; } = new(StringComparer.Ordinal);
        public List<WalletStateChanges> Saves { get; } = [];
        public DateTimeOffset? HistoryStart { get; private set; }
        public DateTimeOffset? LastEvent { get; private set; }
        public int Loads { get; private set; }
        public int FailSaves { get; set; }
        public bool FailLoads { get; init; }

        public Task<WalletStateSnapshot?> LoadAsync(CancellationToken cancellationToken)
        {
            Loads++;
            if (FailLoads)
            {
                throw new InvalidOperationException("db down");
            }

            return Task.FromResult(HistoryStart is { } start
                ? new WalletStateSnapshot(start, LastEvent!.Value, Rows.Values.ToList())
                : null);
        }

        public Task SaveAsync(WalletStateChanges changes, CancellationToken cancellationToken)
        {
            if (FailSaves > 0)
            {
                FailSaves--;
                throw new InvalidOperationException("db down");
            }

            Saves.Add(changes);
            foreach (WalletRecord r in changes.Upserts)
            {
                Rows[r.Wallet] = r;
            }

            foreach (string w in changes.Deletes)
            {
                Rows.Remove(w);
            }

            HistoryStart = changes.HistoryStart;
            LastEvent = changes.LastEvent;
            return Task.CompletedTask;
        }
    }
}
