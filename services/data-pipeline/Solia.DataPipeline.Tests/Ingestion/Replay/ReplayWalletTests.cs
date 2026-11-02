using System.Globalization;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives;
using static Solia.DataPipeline.Tests.Pipeline.PipelineEngineTests;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

/// <summary>FE-T9 (#189): the wallet features are the same live and in replay, and a replay warm-up feeds them.</summary>
public class ReplayWalletTests
{
    private static readonly DateTimeOffset H0 = new(2026, 4, 20, 0, 0, 0, TimeSpan.Zero);
    private static readonly PipelineOptions Options = new() { SnapshotCadenceSeconds = 10, WindowSeconds = 300 };

    /// <summary>Two hours of trades every 7 s over rotating tokens and wallets, with a create and a creator sell per token.</summary>
    private static Dictionary<DateTimeOffset, List<string>> Fixture()
    {
        var hours = new Dictionary<DateTimeOffset, List<string>> { [H0] = [], [H0.AddHours(1)] = [] };
        for (int i = 0; i * 7 < 7200; i++)
        {
            DateTimeOffset at = H0.AddSeconds(i * 7);
            string mint = $"T{i / 60}"; // a new token every 7 minutes
            string line = (i % 60) switch
            {
                0 => Line("create", mint, $"dev{i / 120}", at, initialBuy: 50_000_000), // devs launch twice
                30 => Line("sell", mint, $"dev{i / 120}", at),
                _ => Line(i % 4 == 0 ? "sell" : "buy", mint, i % 5 == 0 ? $"new{i}" : $"w{i % 9}", at),
            };
            hours[ReplayWindow.FloorToHour(at)].Add(line);
        }

        return hours;
    }

    private static string Line(string type, string mint, string signer, DateTimeOffset at, long initialBuy = 0) =>
        string.Create(CultureInfo.InvariantCulture,
            $$"""{"txType":"{{type}}","mint":"{{mint}}","pool":"pump","price":0.001,"solAmount":0.5,"tokenAmount":{{(type == "create" ? "null" : "1000")}},"initialBuy":{{initialBuy}},"supply":1000000000,"txSigner":"{{signer}}","timestamp":{{at.ToUnixTimeMilliseconds()}}}""");

    private static Dictionary<DateTimeOffset, byte[]?> Archives(Dictionary<DateTimeOffset, List<string>> hours) =>
        hours.ToDictionary(h => h.Key, h => (byte[]?)Zstd(string.Join("\n", h.Value)));

    private static async Task<List<FeatureSnapshot>> ReplayAsync(ReplayWindow window, TimeSpan warmup = default)
    {
        var sink = new RecordingSink();
        await new ReplayRunner(NullLoggerFactory.Instance, (_, _) => Task.CompletedTask).ReplayAsync(window,
            new DelegateArchive(Opener(Archives(Fixture()))), sink, () => 0, Options, default, warmup);
        return sink.Batches.SelectMany(b => b.Snapshots).ToList();
    }

    [Fact]
    public async Task The_same_events_give_the_same_wallet_features_live_and_in_replay()
    {
        // Live: the stream's frames parsed and fed to the engine as they come.
        List<RawEvent> stream = Fixture().OrderBy(h => h.Key).SelectMany(h => h.Value)
            .Select(line => PumpApiEventParser.Parse(line)!).ToList();
        var live = new RecordingSink();
        await new PipelineEngine(new ListSource(stream), new TokenWindowStore(Options), live, Options,
            NullLogger<PipelineEngine>.Instance, (_, _) => Task.CompletedTask, TimeSpan.FromSeconds(5)).RunAsync(default);

        // Replay: the same frames from the hourly archives, hour after hour.
        List<FeatureSnapshot> replayed = await ReplayAsync(new ReplayWindow(H0, H0.AddHours(2)));

        List<FeatureSnapshot> liveSnapshots = live.Batches.SelectMany(b => b.Snapshots).ToList();
        Assert.True(liveSnapshots.Count > 500);
        Assert.Equal(liveSnapshots.Select(s => (s.Mint, s.Ts, s.Wallet)), replayed.Select(s => (s.Mint, s.Ts, s.Wallet)));
        Assert.Contains(replayed, s => s.Wallet!.FreshShare1h is not null);       // past the first hour
        Assert.Contains(replayed, s => s.Wallet!.CreatorFirstSellSec is not null); // creators sold
        Assert.Contains(replayed, s => s.Wallet!.CreatorBuySupplyShare == 0.05);   // the create's dev buy
    }

    [Fact]
    public async Task A_warm_up_feeds_the_wallet_memory_without_writing_its_hours()
    {
        var window = new ReplayWindow(H0.AddHours(1), H0.AddHours(2));

        List<FeatureSnapshot> cold = await ReplayAsync(window);
        List<FeatureSnapshot> warm = await ReplayAsync(window, TimeSpan.FromHours(1));

        Assert.All(warm, s => Assert.True(s.Ts >= window.StartUtc)); // nothing of the warm-up is written
        FeatureSnapshot firstWarm = warm.MinBy(s => s.Ts)!;
        FeatureSnapshot firstCold = cold.MinBy(s => s.Ts)!;
        Assert.True(firstWarm.Wallet!.CoverageSec >= 3600);
        Assert.Null(firstCold.Wallet!.FreshShare1h);     // no history yet
        Assert.NotNull(firstWarm.Wallet!.FreshShare1h);  // the warm-up hour covers it
    }

    [Fact]
    public async Task The_warm_up_sink_forwards_only_the_window()
    {
        var inner = new RecordingSink();
        var sink = new WarmupSink(inner, H0.AddHours(1));
        FeatureSnapshot before = new() { Mint = "A", Ts = H0.AddMinutes(59) };
        FeatureSnapshot after = new() { Mint = "A", Ts = H0.AddHours(1) };

        await sink.WriteAsync(new FlushBatch(H0.AddMinutes(59), [before], ["GONE"]), default);
        await sink.WriteAsync(new FlushBatch(H0.AddHours(1), [before, after], ["X"]), default);
        await sink.DeactivateAllAsync(default);

        FlushBatch written = Assert.Single(inner.Batches);
        Assert.Equal([after], written.Snapshots);
        Assert.Equal(["X"], written.Deactivated);
        Assert.Equal(1, inner.ResetCalls);
    }
}
