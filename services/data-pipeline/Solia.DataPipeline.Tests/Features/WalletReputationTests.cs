using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Features.WalletStoreTests;

namespace Solia.DataPipeline.Tests.Features;

/// <summary>FE-T10 (#190): wallet reputation — round trips, creator track record, early buyers, smart flow, bundling.</summary>
public class WalletReputationTests
{
    private const int Day = 86_400;
    private const int Hour = 3_600;

    // Age cap 24 h, positions live 6 h, so trips count when opened within the last 18 h; outcomes after 1 h + 1 h.
    private static readonly PipelineOptions Options = new() { WalletTtlHours = 48, WalletAgeCapHours = 24 };

    private static WalletStore Store(bool track = false) => new(Options, track);

    /// <summary>A store whose history covers the age cap before <see cref="WalletStoreTests.T0"/>.</summary>
    private static WalletStore Covered(bool track = false)
    {
        WalletStore store = Store(track);
        store.Apply(Trade("FILL", "filler", T0.AddSeconds(-Day - 1)));
        return store;
    }

    private static TokenWindowStore Window(PipelineOptions? options = null)
    {
        var store = new TokenWindowStore(options ?? new PipelineOptions { WindowSeconds = 300, WalletTtlHours = 48, ReputationEarlyBuyers = 3 });
        store.Apply(Trade("FILL", "filler", T0.AddSeconds(-Day - 1)));
        return store;
    }

    /// <summary><paramref name="trips"/> round trips of 1 SOL on distinct tokens, ending before T0; the first <paramref name="wins"/> double their SOL.</summary>
    private static void RoundTrips(Action<RawEvent> apply, string wallet, int trips, int wins, DateTimeOffset? start = null)
    {
        DateTimeOffset at = start ?? T0.AddHours(-2);
        for (int i = 0; i < trips; i++)
        {
            string mint = $"RT-{wallet}-{i}";
            apply(Trade(mint, wallet, at.AddSeconds(i * 20)));
            apply(Trade(mint, wallet, at.AddSeconds((i * 20) + 10), buy: false, sol: i < wins ? 2m : 0.5m));
        }
    }

    private static RawEvent Priced(RawEvent e, decimal price) => e with { Price = price };

    // --- round trips ---

    [Fact]
    public void A_round_trip_realizes_its_pnl_at_the_average_cost_once_fully_sold()
    {
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0, sol: 1m, tokens: 1_000m));
        store.Apply(Trade("A", "w", T0.AddSeconds(5), sol: 3m, tokens: 1_000m)); // 2 000 tokens for 4 SOL
        store.Apply(Trade("A", "w", T0.AddSeconds(10), buy: false, sol: 1.5m, tokens: 1_000m)); // half: +(1.5 - 2)
        DateTimeOffset mid = T0.AddSeconds(11);
        store.Advance(mid);
        Assert.Equal((0, 0, 0.0), store.TripStats("w", mid)); // still open: never a loss
        Assert.Equal(1, store.PositionCount);

        store.Apply(Trade("A", "w", T0.AddSeconds(20), buy: false, sol: 3.5m, tokens: 1_000m)); // the rest: +(3.5 - 2)
        DateTimeOffset now = T0.AddSeconds(21);
        store.Advance(now);

        (int trips, int wins, double pnl) = store.TripStats("w", now);
        Assert.Equal((1, 1), (trips, wins));
        Assert.Equal(1.0, pnl, 9);
        Assert.Equal(0, store.PositionCount);
        Assert.Equal([new RoundTrip(T0.ToUnixTimeSeconds(), pnl)], store.Get("w", now)!.Trips!);
    }

    [Fact]
    public void A_losing_round_trip_is_not_a_win_and_dust_left_by_rounding_still_closes()
    {
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0, sol: 2m, tokens: 1_000m));
        store.Apply(Trade("A", "w", T0.AddSeconds(10), buy: false, sol: 1m, tokens: 995m)); // 0.5 % dust left
        DateTimeOffset now = T0.AddSeconds(11);
        store.Advance(now);

        (int trips, int wins, double pnl) = store.TripStats("w", now);
        Assert.Equal((1, 0), (trips, wins));
        Assert.Equal(1 - (2 * 0.995), pnl, 9);
    }

    [Fact]
    public void Sells_without_a_known_buy_and_trades_not_in_sol_make_no_round_trip()
    {
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0, buy: false)); // bought before the history
        store.Apply(Trade("B", "w", T0.AddSeconds(1)) with { QuoteMint = "USDC" });
        store.Apply(Trade("B", "w", T0.AddSeconds(2), buy: false) with { QuoteMint = "USDC" });
        store.Apply(Trade("C", "w", T0.AddSeconds(3)) with { QuoteMint = null }); // unknown quote: not SOL for P&L
        store.Apply(Trade("C", "w", T0.AddSeconds(4), buy: false) with { QuoteMint = null });
        DateTimeOffset now = T0.AddSeconds(5);
        store.Advance(now);

        Assert.Equal((0, 0, 0.0), store.TripStats("w", now));
        Assert.Equal(5, store.Get("w", now)!.TradeCount); // still counted as trades
    }

    [Fact]
    public void A_position_open_across_a_bucket_boundary_is_dropped_and_never_counts()
    {
        // T0 is 12:00 UTC, a 6 h boundary: [T0, T0 + 6 h) is one bucket.
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0));
        store.Apply(Trade("A", "w", T0.AddHours(6).AddSeconds(-1), buy: false, sol: 5m)); // same bucket: a trip
        store.Apply(Trade("B", "w", T0.AddHours(6).AddSeconds(-1)));
        store.Apply(Trade("B", "w", T0.AddHours(6).AddSeconds(1), buy: false, sol: 5m)); // crossed it: dropped
        DateTimeOffset now = T0.AddHours(7);
        store.Advance(now);

        Assert.Equal(1, store.TripStats("w", now).Trips);
        Assert.Equal(0, store.PositionCount);
    }

    [Fact]
    public void The_sweep_drops_stale_open_positions()
    {
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0));
        store.Advance(T0.AddSeconds(1));
        Assert.Equal(1, store.PositionCount);

        store.Advance(T0.AddHours(7)); // past the next hourly sweep
        Assert.Equal(0, store.PositionCount);
    }

    [Fact]
    public void Round_trips_only_count_when_opened_within_the_cap_minus_the_position_life()
    {
        WalletStore store = Covered();
        RoundTrips(store.Apply, "w", trips: 1, wins: 1, start: T0.AddHours(-18).AddSeconds(-1));
        RoundTrips(store.Apply, "w2", trips: 1, wins: 1, start: T0.AddHours(-18));
        store.Advance(T0);

        Assert.Equal(0, store.TripStats("w", T0).Trips);
        Assert.Equal(1, store.TripStats("w2", T0).Trips);
    }

    [Fact]
    public void A_round_trip_is_seen_only_by_snapshots_strictly_after_its_closing_sell()
    {
        WalletStore store = Covered();
        store.Apply(Trade("A", "w", T0));
        store.Apply(Trade("A", "w", T0.AddSeconds(10), buy: false, sol: 2m));

        store.Advance(T0.AddSeconds(10));
        Assert.Equal(0, store.TripStats("w", T0.AddSeconds(10)).Trips);
        store.Advance(T0.AddSeconds(11));
        Assert.Equal(1, store.TripStats("w", T0.AddSeconds(11)).Trips);
    }

    [Fact]
    public void An_unknown_wallet_has_no_trips_and_is_neither_known_nor_smart()
    {
        WalletStore store = Covered();
        RoundTrips(store.Apply, "two", trips: 2, wins: 2);
        RoundTrips(store.Apply, "pro", trips: 4, wins: 3);
        RoundTrips(store.Apply, "meh", trips: 5, wins: 3);
        store.Advance(T0);

        Assert.Equal((0, 0, 0.0), store.TripStats("nobody", T0));
        Assert.False(store.IsKnown("nobody", T0));
        Assert.False(store.IsKnown("two", T0)); // under ReputationMinTrips (3)
        Assert.False(store.IsSmart("two", T0));
        Assert.True(store.IsSmart("pro", T0));  // 0.75 > 0.6
        Assert.False(store.IsSmart("meh", T0)); // 0.6 is not above 0.6
    }

    // --- early buyers ---

    [Fact]
    public void Early_buyer_quality_averages_the_rated_buyers()
    {
        WalletStore store = Covered();
        RoundTrips(store.Apply, "pro", trips: 4, wins: 4);  // win rate 1, +1 SOL / trip
        RoundTrips(store.Apply, "loser", trips: 4, wins: 0); // win rate 0, -0.5 SOL / trip
        store.Advance(T0);

        (double? known, double? winRate, double? pnl) = store.EarlyBuyerStats(["pro", "loser", "unknown", "two"], T0);

        Assert.Equal(0.5, known);
        Assert.Equal(0.5, winRate);
        Assert.Equal(0.25, pnl!.Value, 9);
    }

    [Fact]
    public void Early_buyer_quality_is_null_in_warm_up_or_without_buyers_and_unrated_without_a_known_one()
    {
        WalletStore warm = Store();
        RoundTrips(warm.Apply, "pro", trips: 4, wins: 4);
        warm.Advance(T0);
        Assert.Equal((null, null, null), warm.EarlyBuyerStats(["pro"], T0));

        WalletStore covered = Covered();
        covered.Advance(T0);
        Assert.Equal((null, null, null), covered.EarlyBuyerStats([], T0));
        Assert.Equal((0.0, null, null), covered.EarlyBuyerStats(["stranger"], T0));
    }

    [Fact]
    public void The_snapshot_rates_the_first_distinct_buyers_leaving_the_creator_out()
    {
        TokenWindowStore store = Window();
        RoundTrips(store.Apply, "pro", trips: 3, wins: 3, start: T0.AddMinutes(-30));
        store.Apply(Create("TOK", "dev", T0));
        store.Apply(Trade("TOK", "dev", T0.AddSeconds(1)));
        store.Apply(Trade("TOK", "pro", T0.AddSeconds(2)));
        store.Apply(Trade("TOK", "pro", T0.AddSeconds(3))); // same buyer again
        store.Apply(Trade("TOK", "a", T0.AddSeconds(4)));
        store.Apply(Trade("TOK", "b", T0.AddSeconds(5)));
        store.Apply(Trade("TOK", "late", T0.AddSeconds(6))); // fourth distinct buyer: past N = 3

        ReputationFeatures r = store.SnapshotActive(T0.AddSeconds(30)).Single(s => s.Mint == "TOK").Reputation!;

        Assert.Equal(3, r.EarlyBuyers); // pro, a, b
        Assert.Equal(1.0 / 3, r.EarlyKnownShare!.Value, 9);
        Assert.Equal(1.0, r.EarlyWinRate);
    }

    // --- creator track record ---

    [Fact]
    public void A_launch_outcome_is_recorded_only_after_the_horizon_plus_the_embargo()
    {
        WalletStore store = Covered();
        store.Apply(Create("OLD", "dev", T0)); // first price 0.001
        store.Apply(Priced(Trade("OLD", "x", T0.AddMinutes(10)), 0.004m)); // x4: pumped (>= x3)
        store.Apply(Priced(Trade("OLD", "x", T0.AddMinutes(20), buy: false), 0.001m)); // -75 % from the peak: rugged
        DateTimeOffset settle = T0.AddSeconds(2 * Hour);

        store.Advance(settle);
        Assert.Equal((0, null, null), store.CreatorReputation("dev", settle, settle)); // not over yet at its ts
        Assert.Equal(1, store.TrackedLaunchCount);

        DateTimeOffset now = settle.AddSeconds(1);
        store.Advance(now);
        Assert.Equal((1, 1.0, 1.0), store.CreatorReputation("dev", now, now));
        Assert.Equal(0, store.TrackedLaunchCount);
    }

    [Fact]
    public void Prices_after_the_outcome_horizon_and_unsigned_trades_are_handled()
    {
        WalletStore store = Covered();
        store.Apply(Create("A", "dev", T0));
        store.Apply(Priced(Trade("A", "x", T0.AddMinutes(5)), 0.0025m) with { Trader = null }); // unsigned: still counted (x2.5)
        store.Apply(Priced(Trade("A", "x", T0.AddSeconds(Hour + 1)), 1m) with { Trader = null }); // after the horizon: ignored
        store.Apply(Create("B", "dev", T0.AddSeconds(1)));
        store.Apply(Priced(Trade("B", "x", T0.AddMinutes(5)), 0.0035m) with { Trader = null }); // x3.5: pumped
        DateTimeOffset now = T0.AddHours(3);
        store.Advance(now);

        (int? settled, double? rug, double? pump) = store.CreatorReputation("dev", now, now);
        Assert.Equal(2, settled);
        Assert.Equal(0.0, rug);
        Assert.Equal(0.5, pump);
        Assert.Null(store.Get("x", now)); // an unsigned trade creates no wallet
    }

    [Fact]
    public void Creator_reputation_counts_earlier_launches_within_the_cap_only()
    {
        WalletStore store = Covered();
        store.Apply(Create("ANCIENT", "dev", T0.AddSeconds(-Day - 1)));
        store.Apply(Create("OLD", "dev", T0));
        store.Apply(Create("THIS", "dev", T0.AddHours(3)));
        DateTimeOffset now = T0.AddHours(3).AddSeconds(10);
        store.Advance(now);

        // ANCIENT launched more than 24 h before now: out of the cap. THIS isn't before its own creation (nor settled).
        Assert.Equal(1, store.CreatorReputation("dev", T0.AddHours(3), now).Settled);
        Assert.Equal(0, store.CreatorReputation("dev", T0, now).Settled); // OLD is not "previous" to itself
    }

    [Fact]
    public void Creator_reputation_is_null_in_warm_up_and_zero_for_an_unknown_creator()
    {
        WalletStore warm = Store();
        warm.Apply(Create("A", "dev", T0));
        warm.Advance(T0.AddHours(3));
        Assert.Equal((null, null, null), warm.CreatorReputation("dev", T0.AddHours(3), T0.AddHours(3)));

        WalletStore covered = Covered();
        covered.Advance(T0);
        Assert.Equal((0, null, null), covered.CreatorReputation("newdev", T0, T0));
    }

    [Fact]
    public void A_token_without_a_create_has_no_creator_reputation()
    {
        TokenWindowStore store = Window();
        store.Apply(Trade("TOK", "a", T0));

        ReputationFeatures r = store.SnapshotActive(T0.AddSeconds(10)).Single(s => s.Mint == "TOK").Reputation!;

        Assert.Null(r.CreatorSettledLaunches);
        Assert.Null(r.CreatorRugShare);
    }

    // --- smart-money flow ---

    [Fact]
    public void Smart_flow_is_the_windows_sol_volume_from_smart_wallets()
    {
        TokenWindowStore store = Window();
        RoundTrips(store.Apply, "pro", trips: 3, wins: 3, start: T0.AddMinutes(-30));
        store.Apply(Create("TOK", "dev", T0));
        store.Apply(Trade("TOK", "pro", T0.AddSeconds(1), sol: 3m));
        store.Apply(Trade("TOK", "pro", T0.AddSeconds(2), buy: false, sol: 1m, tokens: 100m));
        store.Apply(Trade("TOK", "a", T0.AddSeconds(3), sol: 1m));
        store.Apply(Trade("TOK", "b", T0.AddSeconds(4), sol: 5m) with { QuoteMint = "USDC" }); // not SOL: not counted

        ReputationFeatures r = store.SnapshotActive(T0.AddSeconds(30)).Single(s => s.Mint == "TOK").Reputation!;

        Assert.Equal(0.75, r.SmartBuyShare);
        Assert.Equal(2.0, r.SmartNetFlowSol);
        Assert.Equal(1, r.SmartTraders);
    }

    [Fact]
    public void Smart_flow_is_null_in_warm_up_and_zero_without_a_smart_wallet()
    {
        var warm = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300 });
        warm.Apply(Trade("TOK", "a", T0));
        ReputationFeatures w = warm.SnapshotActive(T0.AddSeconds(10)).Single().Reputation!;
        Assert.Null(w.SmartBuyShare);
        Assert.Null(w.SmartNetFlowSol);
        Assert.Null(w.SmartTraders);

        TokenWindowStore covered = Window();
        covered.Apply(Trade("TOK", "a", T0, sol: 2m));
        covered.Apply(Trade("TOK", "b", T0.AddSeconds(1), buy: false, sol: 1m));
        ReputationFeatures c = covered.SnapshotActive(T0.AddSeconds(10)).Single(s => s.Mint == "TOK").Reputation!;
        Assert.Equal(0.0, c.SmartBuyShare);
        Assert.Equal(0.0, c.SmartNetFlowSol);
        Assert.Equal(0, c.SmartTraders);
    }

    [Fact]
    public void A_wallet_becoming_smart_later_never_changes_an_earlier_snapshot()
    {
        TokenWindowStore store = Window();
        RoundTrips(store.Apply, "pro", trips: 2, wins: 2, start: T0.AddMinutes(-30));
        store.Apply(Trade("TOK", "pro", T0, sol: 1m));
        store.Apply(Trade("OTHER", "pro", T0.AddSeconds(20)));
        store.Apply(Trade("OTHER", "pro", T0.AddSeconds(25), buy: false, sol: 2m)); // third win, after the first snapshot

        ReputationFeatures before = store.SnapshotActive(T0.AddSeconds(25)).Single(s => s.Mint == "TOK").Reputation!;
        ReputationFeatures after = store.SnapshotActive(T0.AddSeconds(26)).Single(s => s.Mint == "TOK").Reputation!;

        Assert.Equal(0, before.SmartTraders); // the closing sell is at the snapshot's ts: not yet seen
        Assert.Equal(1, after.SmartTraders);
    }

    // --- launch bundling ---

    [Fact]
    public void Bundling_shares_of_the_first_buys_in_the_creation_block_and_the_busiest_block()
    {
        TokenWindowStore store = Window(new PipelineOptions { WindowSeconds = 300, ReputationEarlyBuyers = 4 });
        store.Apply(Create("TOK", "dev", T0) with { Block = 100 });
        store.Apply(Trade("TOK", "a", T0) with { Block = 100 });
        store.Apply(Trade("TOK", "b", T0) with { Block = 100 });
        store.Apply(Trade("TOK", "c", T0.AddSeconds(1)) with { Block = 101 });
        store.Apply(Trade("TOK", "d", T0.AddSeconds(2)) with { Block = 102 });
        store.Apply(Trade("TOK", "e", T0.AddSeconds(3)) with { Block = 103 }); // fifth buy: past N = 4
        store.Apply(Trade("TOK", "a", T0.AddSeconds(3), buy: false) with { Block = 100 }); // a sell: not a buy

        ReputationFeatures r = store.SnapshotActive(T0.AddSeconds(30)).Single(s => s.Mint == "TOK").Reputation!;

        Assert.Equal(0.5, r.BundleCreateBlockShare);
        Assert.Equal(0.5, r.BundleTopBlockShare);
    }

    [Fact]
    public void Bundling_needs_blocks_and_the_creation_share_a_creation_block()
    {
        TokenWindowStore store = Window();
        store.Apply(Create("NOBLOCK", "dev", T0));
        store.Apply(Trade("NOBLOCK", "a", T0));
        store.Apply(Trade("NOCREATE", "a", T0) with { Block = 7 });
        store.Apply(Trade("NOCREATE", "b", T0.AddSeconds(1)) with { Block = 8 });

        IReadOnlyList<FeatureSnapshot> snaps = store.SnapshotActive(T0.AddSeconds(30));
        ReputationFeatures none = snaps.Single(s => s.Mint == "NOBLOCK").Reputation!;
        ReputationFeatures noCreate = snaps.Single(s => s.Mint == "NOCREATE").Reputation!;

        Assert.Null(none.BundleCreateBlockShare);
        Assert.Null(none.BundleTopBlockShare);
        Assert.Null(noCreate.BundleCreateBlockShare);
        Assert.Equal(0.5, noCreate.BundleTopBlockShare);
    }

    // --- persistence & parity ---

    [Fact]
    public void Trips_and_outcomes_are_saved_and_restored()
    {
        WalletStore before = Covered(track: true);
        RoundTrips(before.Apply, "pro", trips: 3, wins: 3, start: T0.AddHours(-4));
        before.Apply(Create("OLD", "dev", T0.AddHours(-3)));
        before.Advance(T0);
        WalletStateChanges saved = before.DrainChanges()!;
        Assert.NotNull(saved.Upserts.Single(w => w.Wallet == "dev").Outcomes);

        WalletStore after = Store(track: true);
        after.Load(new WalletStateSnapshot(saved.HistoryStart, saved.LastEvent, saved.Upserts));
        after.Advance(T0);

        Assert.True(after.IsSmart("pro", T0));
        Assert.Equal(1, after.CreatorReputation("dev", T0, T0).Settled);
    }

    [Fact]
    public void Persisted_arrays_pair_back_into_trips_and_outcomes()
    {
        var outcome = new LaunchOutcome(T0.ToUnixTimeSeconds(), Rugged: true, Pumped: true);
        Assert.Equal(3, PostgresWalletStateStore.Flags(outcome));
        Assert.Equal([outcome], PostgresWalletStateStore.ToOutcomes([T0], [3])!);
        Assert.Equal([new LaunchOutcome(T0.ToUnixTimeSeconds(), false, true)], PostgresWalletStateStore.ToOutcomes([T0], [2])!);
        Assert.Equal([new RoundTrip(T0.ToUnixTimeSeconds(), -0.5)], PostgresWalletStateStore.ToTrips([T0, T0], [-0.5])!);
        Assert.Null(PostgresWalletStateStore.ToTrips([], []));
        Assert.Null(PostgresWalletStateStore.ToOutcomes([], []));
    }

    /// <summary>A deterministic stream: wallets trading tokens in round trips, creators launching, over <paramref name="hours"/>.</summary>
    private static List<RawEvent> Stream(DateTimeOffset start, int hours)
    {
        var events = new List<RawEvent>();
        var rng = new Random(190);
        for (int s = 0; s < hours * Hour; s += 30)
        {
            DateTimeOffset at = start.AddSeconds(s);
            string wallet = $"w{rng.Next(40)}";
            string mint = $"m{rng.Next(25)}";
            decimal price = 0.001m * (decimal)(0.2 + (rng.NextDouble() * 4));
            events.Add(Priced(Trade(mint, wallet, at, buy: rng.Next(2) == 0, sol: 0.5m + rng.Next(3), tokens: 500m + rng.Next(1_000)), price)
                with
            { Block = s / 400 });
            if (s % 1_800 == 0)
            {
                events.Add(Create($"L{s}", $"dev{rng.Next(3)}", at) with { Block = s / 400 });
            }
        }

        return events;
    }

    private static ReputationFeatures Final(IEnumerable<RawEvent> events, DateTimeOffset now, int cadenceSeconds, DateTimeOffset from)
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 3_600, WalletTtlHours = 48, MaxTrackedTokens = 10_000 });
        DateTimeOffset next = from;
        foreach (RawEvent e in events)
        {
            while (cadenceSeconds > 0 && e.Timestamp >= next)
            {
                store.SnapshotActive(next);
                next = next.AddSeconds(cadenceSeconds);
            }

            store.Apply(e);
        }

        store.Apply(Trade("PROBE", "w1", now.AddSeconds(-1)));
        return store.SnapshotActive(now).Single(s => s.Mint == "PROBE").Reputation!;
    }

    [Fact]
    public void The_same_events_give_the_same_reputation_whatever_the_snapshot_cadence()
    {
        DateTimeOffset start = T0.AddHours(-30);
        List<RawEvent> events = Stream(start, hours: 30);
        DateTimeOffset now = T0;

        ReputationFeatures everyMinute = Final(events, now, cadenceSeconds: 60, from: start);
        ReputationFeatures atTheEnd = Final(events, now, cadenceSeconds: 0, from: start);

        Assert.Equal(everyMinute, atTheEnd);
        Assert.NotNull(atTheEnd.EarlyKnownShare);
    }

    [Fact]
    public void A_replay_warmed_up_over_the_age_cap_matches_a_longer_live_history()
    {
        DateTimeOffset start = T0.AddHours(-40);
        List<RawEvent> live = Stream(start, hours: 40);
        DateTimeOffset warmupStart = T0.AddHours(-24).AddSeconds(-1);
        List<RawEvent> replay = live.Where(e => e.Timestamp >= warmupStart).ToList();

        // The PROBE token is bought by w1: its window features match; the smart / early stats come from the wallet memory.
        var liveFeatures = Final(live, T0, cadenceSeconds: 0, from: start);
        var replayFeatures = Final(replay, T0, cadenceSeconds: 0, from: warmupStart);

        Assert.Equal(liveFeatures.EarlyKnownShare, replayFeatures.EarlyKnownShare);
        Assert.Equal(liveFeatures.EarlyWinRate, replayFeatures.EarlyWinRate);
        Assert.Equal(liveFeatures.EarlyPnlSol, replayFeatures.EarlyPnlSol);
        Assert.Equal(liveFeatures.SmartTraders, replayFeatures.SmartTraders);
    }

    [Fact]
    public void A_replay_warmed_up_over_the_age_cap_matches_the_live_creator_record()
    {
        DateTimeOffset start = T0.AddHours(-40);
        List<RawEvent> live = Stream(start, hours: 40);
        DateTimeOffset warmupStart = T0.AddHours(-24).AddSeconds(-1);

        WalletStore Run(IEnumerable<RawEvent> events)
        {
            WalletStore store = Store();
            foreach (RawEvent e in events)
            {
                store.Apply(e);
            }

            store.Advance(T0);
            return store;
        }

        WalletStore full = Run(live);
        WalletStore warmed = Run(live.Where(e => e.Timestamp >= warmupStart));
        for (int dev = 0; dev < 3; dev++)
        {
            Assert.Equal(full.CreatorReputation($"dev{dev}", T0, T0), warmed.CreatorReputation($"dev{dev}", T0, T0));
            Assert.NotNull(full.CreatorReputation($"dev{dev}", T0, T0).Settled);
        }

        Assert.True(Enumerable.Range(0, 40).Sum(w => full.TripStats($"w{w}", T0).Trips) > 40); // a real comparison
        for (int w = 0; w < 40; w++)
        {
            Assert.Equal(full.TripStats($"w{w}", T0).Trips, warmed.TripStats($"w{w}", T0).Trips);
            Assert.Equal(full.TripStats($"w{w}", T0).PnlSol, warmed.TripStats($"w{w}", T0).PnlSol, 9);
        }
    }
}
