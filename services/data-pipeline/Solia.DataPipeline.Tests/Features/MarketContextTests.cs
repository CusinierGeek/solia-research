using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Tests.Pipeline;

namespace Solia.DataPipeline.Tests.Features;

public class MarketContextTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private const int Hour = 3600;

    private static RawEvent Ev(string mint, EventKind kind, double offsetSec, decimal? sol = null, string? quote = RawEvent.WrappedSolMint) =>
        new()
        {
            Mint = mint,
            Kind = kind,
            Timestamp = T0.AddSeconds(offsetSec),
            QuoteMint = quote,
            QuoteAmount = sol,
        };

    private static RawEvent Buy(string mint, double offsetSec, decimal sol = 1m) => Ev(mint, EventKind.Buy, offsetSec, sol);

    private static RawEvent Sell(string mint, double offsetSec, decimal sol = 1m) => Ev(mint, EventKind.Sell, offsetSec, sol);

    /// <summary>
    /// A context whose history starts at T0 (so both windows are covered from T0 + 1 h), with its event
    /// clock walked to <paramref name="clockAt"/> by quiet heartbeat events, then fed <paramref name="events"/>.
    /// </summary>
    private static MarketContext Warm(double clockAt, params RawEvent[] events)
    {
        var market = new MarketContext();
        Walk(market, 0, clockAt);
        foreach (RawEvent e in events)
        {
            market.Apply(e);
        }

        return market;
    }

    /// <summary>Heartbeats (no market content) every 30 s, so the event clock follows time like a real stream.</summary>
    private static void Walk(MarketContext market, double from, double to)
    {
        for (double t = from; t < to; t += 30)
        {
            market.Apply(Ev("TICK", EventKind.Other, t));
        }

        market.Apply(Ev("TICK", EventKind.Other, to));
    }

    private static DateTimeOffset At(double offsetSec) => T0.AddSeconds(offsetSec);

    [Fact]
    public void Aggregates_launches_graduations_flow_and_active_tokens_per_window()
    {
        MarketContext market = Warm(
            Hour,
            Ev("A", EventKind.Create, 100), Ev("B", EventKind.Create, 200), Ev("C", EventKind.Create, 3_500),
            Ev("A", EventKind.Migrate, 3_000),
            Buy("A", 150, 4m), Sell("A", 160, 1m), // long window only
            Buy("C", 3_550, 2m), Sell("C", 3_560, 0.5m), Buy("D", 3_590, 3m)); // both windows

        MarketAggregates m = market.Aggregate(At(Hour + 10));

        Assert.Equal(Hour, m.CoverageSec);
        Assert.Equal(1 / 5.0, m.LaunchRateShort);
        Assert.Equal(3 / 60.0, m.LaunchRateLong);
        Assert.Equal(1, m.GraduationsLong);
        Assert.Equal(1 / 3.0, m.GraduationRatioLong);
        Assert.Equal(5.0, m.BuySolShort);
        Assert.Equal(9.0, m.BuySolLong);
        Assert.Equal(4.5, m.NetSolFlowShort);
        Assert.Equal(7.5, m.NetSolFlowLong);
        Assert.Equal(2, m.ActiveTokensShort); // C, D
        Assert.Equal(3, m.ActiveTokensLong); // A, C, D
    }

    [Fact]
    public void Windows_are_null_until_fully_observed()
    {
        MarketContext market = Warm(0, Ev("A", EventKind.Create, 10), Buy("A", 20));

        MarketAggregates early = market.Aggregate(At(299));
        Assert.Equal(299, early.CoverageSec);
        Assert.Null(early.LaunchRateShort);
        Assert.Null(early.ActiveTokensShort);
        Assert.Null(early.BuySolShort);

        MarketAggregates mid = market.Aggregate(At(300));
        Assert.Equal(1 / 5.0, mid.LaunchRateShort);
        Assert.Equal(1, mid.ActiveTokensShort);
        Assert.Null(mid.LaunchRateLong);
        Assert.Null(mid.GraduationsLong);
        Assert.Null(mid.GraduationRatioLong);
        Assert.Null(mid.NetSolFlowLong);
        Assert.Null(mid.ActiveTokensLong);

        Assert.Equal(Hour, market.Aggregate(At(2 * Hour)).CoverageSec); // capped at the long window
        Assert.Equal(0, new MarketContext().Aggregate(T0).CoverageSec); // nothing seen yet
    }

    [Fact]
    public void Only_events_strictly_before_the_snapshot_second_count()
    {
        DateTimeOffset ts = At(Hour + 100.5);
        MarketContext market = Warm(
            Hour + 99,
            Buy("BEFORE", Hour + 99.9, 1m), // the previous whole second: counted
            Buy("SAME", Hour + 100.2, 10m), // the snapshot's own second, even before ts: not counted
            Ev("SAME", EventKind.Create, Hour + 100.0));

        MarketAggregates m = market.Aggregate(ts);

        Assert.Equal(1.0, m.BuySolShort);
        Assert.Equal(1, m.ActiveTokensShort);
        Assert.Equal(0.0, m.LaunchRateShort);

        // The same events are counted once the clock moves past their second.
        MarketAggregates next = market.Aggregate(At(Hour + 101));
        Assert.Equal(11.0, next.BuySolShort);
        Assert.Equal(2, next.ActiveTokensShort);
        Assert.Equal(1 / 5.0, next.LaunchRateShort);
    }

    [Fact]
    public void A_trade_in_the_current_second_does_not_hide_the_previous_one()
    {
        MarketContext market = Warm(Hour + 50, Buy("A", Hour + 50), Buy("A", Hour + 100));

        Assert.Equal(1, market.Aggregate(At(Hour + 100)).ActiveTokensShort);
    }

    [Fact]
    public void Old_events_leave_the_windows_and_old_tokens_are_forgotten()
    {
        MarketContext market = Warm(10, Ev("A", EventKind.Create, 10), Buy("A", 10, 2m));
        Walk(market, 10, Hour + 10);
        market.Apply(Buy("B", Hour + 10, 1m));

        MarketAggregates beforeExpiry = market.Aggregate(At(Hour + 10));
        Assert.Equal(1 / 60.0, beforeExpiry.LaunchRateLong);
        Assert.Equal(2.0, beforeExpiry.BuySolLong);
        Assert.Equal(1, beforeExpiry.ActiveTokensLong);

        MarketAggregates afterExpiry = market.Aggregate(At(Hour + 11));
        Assert.Equal(0.0, afterExpiry.LaunchRateLong);
        Assert.Null(afterExpiry.GraduationRatioLong); // no launch left in the window
        Assert.Equal(1.0, afterExpiry.BuySolLong);
        Assert.Equal(1, afterExpiry.ActiveTokensLong); // B only

        market.Aggregate(At(3 * Hour));
        Assert.Equal(0, market.TrackedTokens); // A and B evicted: memory is bounded by time
    }

    [Fact]
    public void Buckets_are_reused_across_hours_without_leaking_old_counts()
    {
        var market = new MarketContext();
        for (int h = 0; h < 4; h++)
        {
            Walk(market, h == 0 ? 0 : ((h - 1) * Hour) + 5, (h * Hour) + 5);
            market.Apply(Buy("M", (h * Hour) + 5, 1m)); // four hours: the ring wraps several times
        }

        MarketAggregates m = market.Aggregate(At((3 * Hour) + 6));
        Assert.Equal(1.0, m.BuySolLong);
        Assert.Equal(1.0, m.BuySolShort);
    }

    [Fact]
    public void Non_sol_quotes_count_as_activity_but_not_as_sol_volume()
    {
        MarketContext market = Warm(
            Hour + 10,
            Ev("U", EventKind.Buy, Hour + 10, 500m, quote: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"),
            Ev("N", EventKind.Buy, Hour + 11, null)); // amount not reported

        MarketAggregates m = market.Aggregate(At(Hour + 20));

        Assert.Equal(0.0, m.BuySolShort);
        Assert.Equal(0.0, m.NetSolFlowShort);
        Assert.Equal(2, m.ActiveTokensShort);
    }

    [Fact]
    public void A_single_far_future_timestamp_is_dropped()
    {
        MarketContext market = Warm(Hour + 10, Buy("A", Hour + 10, 1m), Buy("BOGUS", Hour + 1_000, 100m), Buy("A", Hour + 20, 1m));

        MarketAggregates m = market.Aggregate(At(Hour + 1_001));

        Assert.Equal(2.0, m.BuySolLong);
        Assert.Equal(1, m.ActiveTokensLong);
    }

    [Fact]
    public void A_late_event_older_than_the_long_window_is_dropped()
    {
        MarketContext market = Warm(2 * Hour, Buy("A", 2 * Hour, 1m), Ev("OLD", EventKind.Create, 2 * Hour - Hour - 120));

        Assert.Equal(0.0, market.Aggregate(At((2 * Hour) + 1)).LaunchRateLong);
    }

    [Fact]
    public void Arrival_order_does_not_change_the_aggregates()
    {
        var rng = new Random(195);
        List<RawEvent> events = Enumerable.Range(0, 2_000)
            .Select(i => (i % 7) switch
            {
                0 => Ev($"T{i}", EventKind.Create, i * 2.3),
                1 => Ev($"T{i - 1}", EventKind.Migrate, i * 2.3),
                2 or 3 or 4 => Buy($"T{i % 97}", i * 2.3, 0.1m * (i % 13)),
                _ => Sell($"T{i % 89}", i * 2.3, 0.05m * (i % 11)),
            })
            .ToList();

        // Live delivery jitters within a few seconds (well inside the clock skew).
        List<RawEvent> jittered = events
            .Select(e => (e, key: e.Timestamp.AddSeconds(rng.NextDouble() * 5)))
            .OrderBy(x => x.key)
            .Select(x => x.e)
            .ToList();

        Assert.Equal(Warm(0, [.. events]).Aggregate(At(4_500)), Warm(0, [.. jittered]).Aggregate(At(4_500)));
    }

    // ---- Through the store and the engine (what live ingestion and replay both run) ----

    [Fact]
    public void Every_snapshot_of_a_tick_shares_one_aggregate()
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300 });
        store.Apply(Ev("A", EventKind.Create, 0));
        store.Apply(Buy("A", 1));
        store.Apply(Buy("B", 2));
        store.Apply(Ev("NOISE", EventKind.Transfer, 2)); // untracked, but still market time

        IReadOnlyList<FeatureSnapshot> snapshots = store.SnapshotActive(At(10));

        Assert.Equal(2, snapshots.Count);
        Assert.Same(snapshots[0].Market, snapshots[1].Market);
        Assert.Equal(10, snapshots[0].Market!.CoverageSec);
    }

    [Fact]
    public async Task Live_and_replay_attach_the_same_point_in_time_aggregates()
    {
        List<RawEvent> events = [];
        for (int s = 0; s < Hour + 400; s++)
        {
            if (s % 30 == 0)
            {
                events.Add(Ev($"L{s}", EventKind.Create, s));
            }

            if (s % 500 == 0)
            {
                events.Add(Ev($"L{s - 30}", EventKind.Migrate, s + 0.4));
            }

            events.Add(Buy($"L{s / 30 * 30}", s + 0.25, 0.5m));
            if (s % 3 == 0)
            {
                events.Add(Sell($"L{s / 30 * 30}", s + 0.75, 0.2m));
            }
        }

        // Replay: a finite source. Live: the same stream held open until shutdown (arrival-order
        // independence within a second is covered above).
        var replaySink = new PipelineEngineTests.RecordingSink();
        await PipelineEngineTests.Engine(new PipelineEngineTests.ListSource(events), replaySink).RunAsync(default);

        var liveSink = new PipelineEngineTests.RecordingSink();
        using var cts = new CancellationTokenSource();
        Task live = PipelineEngineTests.Engine(new PipelineEngineTests.ListSource(events) { WaitAtEnd = true }, liveSink).RunAsync(cts.Token);
        while (liveSink.Batches.Count < replaySink.Batches.Count - 1)
        {
            await Task.Delay(10);
        }

        await cts.CancelAsync();
        await live;

        Dictionary<DateTimeOffset, MarketAggregates?> replay = Ticks(replaySink);
        Dictionary<DateTimeOffset, MarketAggregates?> liveTicks = Ticks(liveSink);
        Assert.True(replay.Count > 300);
        Assert.True(liveTicks.Count >= replay.Count - 1);
        foreach ((DateTimeOffset ts, MarketAggregates? atTick) in liveTicks)
        {
            Assert.Equal(replay[ts], atTick);
            Assert.Equal(Reference(events, ts), atTick); // only events strictly before ts's second
        }

        Assert.NotNull(replay.Values.Last()!.ActiveTokensLong); // past the warm-up, the long window is reported
    }

    private static Dictionary<DateTimeOffset, MarketAggregates?> Ticks(PipelineEngineTests.RecordingSink sink) =>
        sink.Batches.Where(b => b.Snapshots.Count > 0).ToDictionary(b => b.Ts, b => b.Snapshots[0].Market);

    /// <summary>Recomputed offline, knowing the whole stream: a fresh context fed only what precedes ts's second.</summary>
    private static MarketAggregates Reference(IEnumerable<RawEvent> events, DateTimeOffset ts)
    {
        var market = new MarketContext();
        long cutoff = ts.ToUnixTimeSeconds();
        foreach (RawEvent e in events.Where(e => e.Timestamp.ToUnixTimeSeconds() < cutoff))
        {
            market.Apply(e);
        }

        return market.Aggregate(ts);
    }
}
