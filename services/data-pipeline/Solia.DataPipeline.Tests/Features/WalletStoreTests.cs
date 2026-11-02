using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests.Features;

/// <summary>FE-T9 (#189): the cross-token wallet memory and the wallet features built from it.</summary>
public class WalletStoreTests
{
    internal static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private static readonly PipelineOptions Options = new() { WalletTtlHours = 48, WalletAgeCapHours = 24 };
    private const int Day = 86_400;

    internal static RawEvent Trade(string mint, string trader, DateTimeOffset at, bool buy = true, decimal sol = 1m, decimal tokens = 1_000m) => new()
    {
        Mint = mint,
        Kind = buy ? EventKind.Buy : EventKind.Sell,
        Timestamp = at,
        Trader = trader,
        Price = 0.001m,
        QuoteMint = RawEvent.WrappedSolMint,
        QuoteAmount = sol,
        TokenAmount = tokens,
    };

    internal static RawEvent Create(string mint, string creator, DateTimeOffset at, decimal sol = 2m, decimal tokens = 50_000_000m) => new()
    {
        Mint = mint,
        Kind = EventKind.Create,
        Timestamp = at,
        Trader = creator,
        Price = 0.001m,
        QuoteMint = RawEvent.WrappedSolMint,
        QuoteAmount = sol,
        TokenAmount = tokens,
        Supply = 1_000_000_000m,
    };

    private static WalletStore Store(bool track = false) => new(Options, track);

    /// <summary>A store whose history already covers the age cap (a filler wallet traded a day before T0).</summary>
    private static WalletStore Covered(bool track = false)
    {
        WalletStore store = Store(track);
        store.Apply(Trade("FILL", "filler", T0.AddSeconds(-Day - 1)));
        return store;
    }

    // --- wallet-state updates ---

    [Fact]
    public void Tracks_first_and_last_seen_trades_tokens_traded_and_tokens_created()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0));
        store.Apply(Trade("A", "w", T0.AddSeconds(10), buy: false));
        store.Apply(Trade("B", "w", T0.AddSeconds(20)));
        store.Apply(Create("C", "w", T0.AddSeconds(30)));
        store.Apply(Trade("A", "w", T0.AddSeconds(40))); // A is still among the last mints: not recounted
        store.Advance(T0.AddMinutes(1));

        WalletRecord w = store.Get("w", T0.AddMinutes(1))!;
        Assert.Equal(T0.ToUnixTimeSeconds(), w.FirstSeen);
        Assert.Equal(T0.AddSeconds(40).ToUnixTimeSeconds(), w.LastSeen);
        Assert.Equal(4, w.TradeCount);
        Assert.Equal(2, w.TokensTraded);
        Assert.Equal(1, w.TokensCreated);
        Assert.Equal([T0.AddSeconds(30).ToUnixTimeSeconds()], w.RecentLaunches);
    }

    [Fact]
    public void Unsigned_events_and_non_trades_are_ignored()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0) with { Trader = null });
        store.Apply(Trade("A", "w", T0) with { Kind = EventKind.Transfer });
        store.Apply(Trade("A", "w", T0) with { Kind = EventKind.Migrate });
        store.Advance(T0.AddMinutes(1));

        Assert.Equal(0, store.Count);
        Assert.Equal(0, store.CoverageSec(T0.AddMinutes(1)));
    }

    [Fact]
    public void A_late_event_moves_first_seen_back()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0.AddSeconds(30)));
        store.Advance(T0.AddMinutes(1));
        store.Apply(Trade("A", "w", T0)); // older than what was advanced: applied at once

        Assert.Equal(T0.ToUnixTimeSeconds(), store.Get("w", T0.AddMinutes(1))!.FirstSeen);
    }

    [Fact]
    public void Keeps_only_the_latest_launches()
    {
        WalletStore store = Store();
        for (int i = 0; i < WalletStore.MaxRecentLaunches + 4; i++)
        {
            store.Apply(Create($"M{i}", "dev", T0.AddSeconds(i)));
        }

        store.Advance(T0.AddMinutes(1));

        WalletRecord dev = store.Get("dev", T0.AddMinutes(1))!;
        Assert.Equal(WalletStore.MaxRecentLaunches + 4, dev.TokensCreated);
        Assert.Equal(WalletStore.MaxRecentLaunches, dev.RecentLaunches.Length);
        Assert.Equal(T0.AddSeconds(4).ToUnixTimeSeconds(), dev.RecentLaunches[0]);
    }

    [Fact]
    public void A_far_future_event_is_dropped_instead_of_queued_forever()
    {
        WalletStore store = Store();
        store.Advance(T0);
        store.Apply(Trade("A", "w", T0.AddSeconds(1)));
        store.Apply(Trade("A", "bogus", T0.AddDays(100)));
        store.Apply(Trade("A", "ancient", T0.AddDays(-100)));

        Assert.Equal(1, store.PendingCount);
        Assert.Null(store.Get("ancient", T0));
    }

    // --- point-in-time ordering ---

    [Fact]
    public void Lookups_only_see_events_strictly_before_the_snapshot()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "early", T0.AddSeconds(5)));
        store.Apply(Trade("A", "at", T0.AddSeconds(10)));
        store.Apply(Trade("A", "after", T0.AddSeconds(12))); // within the clock skew: queued, not applied

        store.Advance(T0.AddSeconds(10));

        Assert.NotNull(store.Get("early", T0.AddSeconds(10)));
        Assert.Null(store.Get("at", T0.AddSeconds(10)));
        Assert.Null(store.Get("after", T0.AddSeconds(10)));
        Assert.Equal(2, store.PendingCount);
    }

    [Fact]
    public void A_later_trade_never_changes_an_earlier_snapshot()
    {
        // The same token snapshotted at T0+60 s, with and without trades that arrive before that flush but are
        // dated after it (within the skew): the snapshot is identical.
        RawEvent[] history =
        [
            Trade("OLD", "veteran", T0.AddSeconds(-Day - 10)),
            Create("TOK", "dev", T0),
            Trade("TOK", "veteran", T0.AddSeconds(20)),
            Trade("TOK", "fresh", T0.AddSeconds(30)),
        ];
        RawEvent[] later =
        [
            Trade("TOK", "fresh", T0.AddSeconds(61)),
            Trade("TOK", "dev", T0.AddSeconds(62), buy: false),
            Create("OTHER", "dev", T0.AddSeconds(63)),
        ];

        FeatureSnapshot without = SnapshotAt(history, T0.AddSeconds(60));
        FeatureSnapshot with = SnapshotAt([.. history, .. later], T0.AddSeconds(60));

        Assert.Equal(without.Wallet, with.Wallet);
        Assert.Null(with.Wallet!.CreatorFirstSellSec);
    }

    private static FeatureSnapshot SnapshotAt(IEnumerable<RawEvent> events, DateTimeOffset at)
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300, WalletTtlHours = 48 });
        foreach (RawEvent e in events)
        {
            store.Apply(e);
        }

        return store.SnapshotActive(at).Single(s => s.Mint == "TOK");
    }

    // --- features ---

    [Fact]
    public void Freshness_counts_unknown_wallets_as_fresh_and_caps_ages()
    {
        WalletStore store = Store();
        store.Apply(Trade("X", "ancient", T0.AddSeconds(-2 * Day + 100))); // older than the cap: capped at 24 h
        store.Apply(Trade("X", "hours", T0.AddHours(-3)));
        store.Apply(Trade("X", "minutes", T0.AddMinutes(-10)));
        store.Advance(T0);

        (double? fresh1h, double? fresh24h, double? median) =
            store.TraderStats(["ancient", "hours", "minutes", "unknown"], T0);

        Assert.Equal(0.5, fresh1h);   // minutes + unknown
        Assert.Equal(0.75, fresh24h); // all but ancient
        Assert.Equal((600 + (3 * 3600)) / 2.0, median); // ages 0, 600, 10 800, 86 400
    }

    [Fact]
    public void History_dependent_features_stay_null_until_their_horizon_is_covered()
    {
        WalletStore store = Store();
        store.Apply(Trade("X", "w", T0));
        store.Apply(Create("TOK", "dev", T0.AddMinutes(1)));

        store.Advance(T0.AddMinutes(30));
        Assert.Equal((null, null, null), store.TraderStats(["w"], T0.AddMinutes(30)));
        Assert.Equal((null, null), store.CreatorStats("dev", T0.AddMinutes(1), T0.AddMinutes(30)));
        Assert.Equal(1800, store.CoverageSec(T0.AddMinutes(30)));

        store.Advance(T0.AddHours(2));
        (double? fresh1h, double? fresh24h, double? median) = store.TraderStats(["w"], T0.AddHours(2));
        Assert.Equal(0.0, fresh1h);
        Assert.Null(fresh24h);
        Assert.Null(median);

        store.Advance(T0.AddHours(25));
        Assert.Equal(Day, store.CoverageSec(T0.AddHours(25))); // capped at the age cap
        Assert.NotNull(store.TraderStats(["w"], T0.AddHours(25)).MedianAgeSec);
        Assert.NotNull(store.CreatorStats("dev", T0.AddMinutes(1), T0.AddHours(25)).AgeSec);
    }

    [Fact]
    public void No_trader_means_no_freshness()
    {
        WalletStore store = Covered();
        store.Advance(T0);

        Assert.Equal((null, null, null), store.TraderStats([], T0));
    }

    [Fact]
    public void Creator_previous_launches_count_earlier_ones_within_the_cap()
    {
        WalletStore store = Covered();
        store.Apply(Create("OLD", "dev", T0.AddHours(-1)));          // more than the cap before now: not counted
        store.Apply(Create("RECENT", "dev", T0.AddHours(10)));
        store.Apply(Create("TOK", "dev", T0.AddHours(20)));
        store.Apply(Create("LATER", "dev", T0.AddHours(21)));        // after this launch: not "previous"
        DateTimeOffset now = T0.AddHours(23).AddMinutes(30);
        store.Advance(now);

        (int? age, int? previous) = store.CreatorStats("dev", T0.AddHours(20), now);

        Assert.Equal(Day, age); // first seen 24.5 h ago: capped
        Assert.Equal(1, previous); // RECENT only: OLD is beyond the cap, LATER is after this launch
    }

    [Fact]
    public void A_creator_not_seen_before_the_snapshot_is_a_fresh_wallet()
    {
        WalletStore store = Covered();
        store.Advance(T0);

        Assert.Equal((0, 0), store.CreatorStats("newdev", T0, T0));
    }

    [Fact]
    public void Creator_behaviour_on_the_launch()
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300 });
        store.Apply(Create("TOK", "dev", T0, sol: 1.5m, tokens: 50_000_000m));
        store.Apply(Trade("TOK", "buyer", T0.AddSeconds(5)));
        store.Apply(Trade("TOK", "dev", T0.AddSeconds(10), tokens: 50_000_000m)); // bought 100 M in all
        store.Apply(Trade("TOK", "dev", T0.AddSeconds(20), buy: false, tokens: 25_000_000m));
        store.Apply(Trade("TOK", "dev", T0.AddSeconds(30), buy: false, tokens: 15_000_000m));

        WalletFeatures w = store.SnapshotActive(T0.AddSeconds(60)).Single().Wallet!;

        Assert.Equal(1.5, w.CreatorBuySol);
        Assert.Equal(0.05, w.CreatorBuySupplyShare);
        Assert.Equal(0.4, w.CreatorSoldShare);
        Assert.Equal(40, w.CreatorFirstSellSec);
        Assert.Null(w.CreatorAgeSec); // history doesn't cover the age cap yet
        Assert.Equal(60, w.CoverageSec);
    }

    [Fact]
    public void Creator_features_are_null_without_a_create_and_zero_without_a_dev_buy()
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300 });
        store.Apply(Trade("NOCREATE", "buyer", T0));
        store.Apply(Create("NODEV", "dev", T0, sol: 0m, tokens: 0m));
        store.Apply(Trade("NODEV", "buyer", T0.AddSeconds(1)));

        IReadOnlyList<FeatureSnapshot> snaps = store.SnapshotActive(T0.AddSeconds(30));
        WalletFeatures none = snaps.Single(s => s.Mint == "NOCREATE").Wallet!;
        WalletFeatures noDev = snaps.Single(s => s.Mint == "NODEV").Wallet!;

        Assert.Null(none.CreatorBuySol);
        Assert.Null(none.CreatorBuySupplyShare);
        Assert.Null(none.CreatorSoldShare);
        Assert.Null(none.CreatorFirstSellSec);
        Assert.Equal(0.0, noDev.CreatorBuySol);
        Assert.Equal(0.0, noDev.CreatorBuySupplyShare);
        Assert.Null(noDev.CreatorSoldShare); // bought nothing: no share to sell
    }

    [Fact]
    public void Snapshot_freshness_uses_the_windows_signers()
    {
        var store = new TokenWindowStore(new PipelineOptions { WindowSeconds = 300, WalletTtlHours = 48 });
        store.Apply(Trade("ELSEWHERE", "veteran", T0.AddSeconds(-Day - 10)));
        store.Apply(Trade("TOK", "veteran", T0.AddSeconds(10)));
        store.Apply(Trade("TOK", "newbie", T0.AddSeconds(20)));

        WalletFeatures w = store.SnapshotActive(T0.AddSeconds(30)).Single(s => s.Mint == "TOK").Wallet!;

        Assert.Equal(0.5, w.FreshShare1h);
        Assert.Equal(0.5, w.FreshShare24h);
        Assert.Equal((10 + Day) / 2.0, w.MedianTraderAgeSec); // ages 10 s and 86 400 (capped)
    }

    // --- TTL eviction ---

    [Fact]
    public void A_wallet_inactive_beyond_the_ttl_is_forgotten_even_before_the_sweep()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0));
        store.Advance(T0.AddSeconds(1)); // sweeps now; the next sweep is an hour later

        DateTimeOffset expired = T0.AddHours(48).AddSeconds(2);
        Assert.NotNull(store.Get("w", T0.AddHours(48)));
        Assert.Null(store.Get("w", expired)); // logically gone, whenever the sweep runs
        Assert.Equal(0.0, store.TraderStats(["w"], expired).MedianAgeSec); // an expired wallet is fresh

        store.Advance(T0.AddHours(48));
        store.Apply(Trade("A", "w", expired));
        store.Advance(expired.AddSeconds(1));
        Assert.Equal(expired.ToUnixTimeSeconds(), store.Get("w", expired.AddSeconds(1))!.FirstSeen); // fresh again
    }

    [Fact]
    public void The_sweep_drops_expired_wallets_and_reports_them_for_deletion()
    {
        WalletStore store = Store(track: true);
        store.Apply(Trade("A", "old", T0));
        store.Apply(Trade("A", "active", T0.AddHours(47)));
        store.Advance(T0.AddHours(47).AddMinutes(1));
        Assert.Equal(2, store.Count);
        store.DrainChanges();

        store.Advance(T0.AddHours(49));

        Assert.Equal(1, store.Count);
        WalletStateChanges changes = store.DrainChanges()!;
        Assert.Equal(["old"], changes.Deletes);
        Assert.Empty(changes.Upserts);
    }

    // --- restart persistence ---

    [Fact]
    public void Changes_are_drained_once_and_restored_after_a_failed_save()
    {
        WalletStore store = Store(track: true);
        Assert.Null(store.DrainChanges()); // nothing observed yet
        store.Apply(Trade("A", "w1", T0));
        store.Apply(Trade("A", "w2", T0.AddSeconds(1)));
        store.Advance(T0.AddSeconds(5));

        WalletStateChanges first = store.DrainChanges()!;
        Assert.Equal(["w1", "w2"], first.Upserts.Select(u => u.Wallet).Order());
        Assert.Equal(T0, first.HistoryStart);
        Assert.Equal(T0.AddSeconds(1), first.LastEvent);
        Assert.Empty(store.DrainChanges()!.Upserts);

        store.RestoreChanges(first);
        Assert.Equal(2, store.DrainChanges()!.Upserts.Count);
    }

    [Fact]
    public void An_untracked_store_reports_no_changes()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0));
        store.Advance(T0.AddSeconds(5));

        Assert.False(store.TracksChanges);
        Assert.Null(store.DrainChanges());
    }

    [Fact]
    public void A_restored_state_keeps_wallets_known_and_the_coverage()
    {
        WalletStore before = Store(track: true);
        before.Apply(Trade("FILL", "filler", T0.AddSeconds(-2 * Day + 3600)));
        before.Apply(Trade("A", "veteran", T0.AddHours(-5)));
        before.Apply(Create("TOK", "dev", T0.AddHours(-2)));
        before.Advance(T0);
        WalletStateChanges saved = before.DrainChanges()!;

        WalletStore after = Store(track: true);
        after.Load(new WalletStateSnapshot(saved.HistoryStart, saved.LastEvent, saved.Upserts));
        after.Apply(Trade("B", "newbie", T0.AddMinutes(-110))); // 10 min after the last saved event: a short restart
        DateTimeOffset now = T0.AddMinutes(-100);
        after.Advance(now);

        Assert.Equal(Day, after.CoverageSec(now));
        WalletRecord veteran = after.Get("veteran", now)!;
        Assert.Equal(before.Get("veteran", now)!.FirstSeen, veteran.FirstSeen);
        Assert.Equal(1, veteran.TradeCount);
        (double? fresh1h, _, _) = after.TraderStats(["veteran", "newbie"], now);
        Assert.Equal(0.5, fresh1h); // the veteran is not reset to fresh
        Assert.Equal(1, after.CreatorStats("dev", now, now).PreviousLaunches);
    }

    [Fact]
    public void A_long_outage_restarts_the_coverage()
    {
        WalletStore before = Covered(track: true);
        before.Apply(Trade("A", "veteran", T0));
        before.Advance(T0.AddSeconds(1));
        WalletStateChanges saved = before.DrainChanges()!;

        WalletStore after = Store();
        after.Load(new WalletStateSnapshot(saved.HistoryStart, saved.LastEvent, saved.Upserts));
        DateTimeOffset back = T0 + WalletStore.StaleGap + TimeSpan.FromMinutes(5);
        after.Apply(Trade("B", "w", back));
        after.Advance(back.AddSeconds(10));

        Assert.Equal(10, after.CoverageSec(back.AddSeconds(10)));
        Assert.NotNull(after.Get("veteran", back)); // wallets are kept, only the coverage restarts
    }

    [Fact]
    public void Load_is_refused_once_events_were_applied()
    {
        WalletStore store = Store();
        store.Apply(Trade("A", "w", T0));
        store.Advance(T0.AddSeconds(1));

        Assert.Throws<InvalidOperationException>(() => store.Load(new WalletStateSnapshot(T0, T0, [])));
    }
}
