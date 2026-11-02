using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Tests.Features;

/// <summary>FE-T13 (#197): stream-derived holder positions and the holder-supply features.</summary>
public class HolderSupplyTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private static RawEvent Ev(EventKind kind, string trader, decimal tokens, int sec, string? pool = null) => new()
    {
        Mint = "M",
        Kind = kind,
        Timestamp = T0.AddSeconds(sec),
        Trader = trader,
        TokenAmount = tokens,
        QuoteAmount = 1m,
        Price = 1m,
        PoolId = pool,
    };

    private static RawEvent Create(string creator, decimal devBuy, int sec = 0, decimal supply = 1_000m) =>
        Ev(EventKind.Create, creator, devBuy, sec, pool: "POOL") with { Supply = supply };

    private static RawEvent Transfer(int sec, params TokenTransfer[] legs) => new()
    {
        Mint = "M",
        Kind = EventKind.Transfer,
        Timestamp = T0.AddSeconds(sec),
        TokenTransfers = new TransferLegs(legs),
    };

    private static TokenState State(int window = 300, double largeShare = 0.02, int maxWallets = HolderLedger.DefaultMaxWallets) =>
        new("M", window, 20, 14, T0, holderLargeShare: largeShare, holderMaxWallets: maxWallets);

    // --- position accounting ---

    [Fact]
    public void Dev_buy_buys_sells_and_liquidity_move_positions()
    {
        TokenState state = State();
        state.Apply(Create("dev", 100m));
        state.Apply(Ev(EventKind.Buy, "a", 50m, 1));
        state.Apply(Ev(EventKind.Sell, "a", 20m, 2));
        state.Apply(Ev(EventKind.AddLiquidity, "dev", 40m, 3));
        state.Apply(Ev(EventKind.RemoveLiquidity, "dev", 10m, 4));

        HolderSupplyFeatures h = state.Snapshot(T0.AddSeconds(5)).HolderSupply!;

        // dev 100 - 40 + 10 = 70, a 30: circulating 100.
        Assert.Equal(2, h.Holders);
        Assert.Equal(0.7, h.Top1Share!.Value, 10);
        Assert.Equal(1.0, h.Top10Share!.Value, 10);
        Assert.False(h.Incomplete);
    }

    [Fact]
    public void Transfers_move_tokens_between_wallets_and_skip_pools()
    {
        TokenState state = State();
        state.Apply(Create("dev", 100m));
        state.Apply(Transfer(1, new TokenTransfer("dev", "alt1", 30m), new TokenTransfer("dev", "alt2", 30m)));
        state.Apply(Transfer(2, new TokenTransfer("alt2", "POOL", 10m))); // into the pool: not a holder
        state.Apply(Transfer(3, new TokenTransfer("dev", "M", 5m)));       // to the mint account: ignored too

        HolderSupplyFeatures h = state.Snapshot(T0.AddSeconds(4)).HolderSupply!;

        // dev 35, alt1 30, alt2 20 = 85 circulating; the pool and mint never hold.
        Assert.Equal(3, h.Holders);
        Assert.Equal(35.0 / 85.0, h.Top1Share!.Value, 10);
        Assert.Equal(1.0, h.Top10Share!.Value, 10);
    }

    [Fact]
    public void A_pool_learned_later_drops_the_position_it_received_earlier()
    {
        var ledger = new HolderLedger();
        ledger.Add("a", 60m, T0);
        ledger.Add("NEWPOOL", 40m, T0);
        Assert.Equal(100m, ledger.Circulating);

        ledger.Exclude("NEWPOOL");
        ledger.Add("NEWPOOL", 10m, T0);

        Assert.Equal(60m, ledger.Circulating);
        Assert.Equal(1, ledger.Count);
    }

    [Fact]
    public void Selling_more_than_seen_counts_as_zero_not_negative_supply()
    {
        var ledger = new HolderLedger();
        ledger.Add("a", 50m, T0);
        ledger.Add("old", -80m, T0); // bought before the history started
        ledger.Add("old", 30m, T0);  // still below zero

        HolderSupplyFeatures h = ledger.Snapshot(1_000m, 0.02, incomplete: true);

        Assert.Equal(50m, ledger.Circulating);
        Assert.Equal(1, h.Holders);
        Assert.Equal(1.0, h.Top1Share!.Value, 10);
        Assert.True(h.Incomplete);
    }

    // --- top-k ---

    [Fact]
    public void Top10_takes_the_ten_largest_positions_with_a_stable_tie_break()
    {
        var ledger = new HolderLedger();
        for (int i = 1; i <= 15; i++)
        {
            ledger.Add($"w{i:00}", i, T0); // 1..15, Σ = 120
        }

        ledger.Add("tie", 15m, T0); // ties w15

        HolderSupplyFeatures h = ledger.Snapshot(1_000m, 0.012, incomplete: false);

        // circulating 135; top 10 = 15 + 15 + 14 + ... + 7 = 114
        Assert.Equal(16, h.Holders);
        Assert.Equal(15.0 / 135.0, h.Top1Share!.Value, 10);
        Assert.Equal(114.0 / 135.0, h.Top10Share!.Value, 10);
        Assert.Equal(4, h.LargeHolders); // > 12 tokens: 13, 14, 15, tie
    }

    [Fact]
    public void Nothing_circulating_leaves_shares_null_and_unknown_supply_leaves_large_holders_null()
    {
        HolderSupplyFeatures empty = new HolderLedger().Snapshot(1_000m, 0.02, incomplete: false);
        Assert.Equal(0, empty.Holders);
        Assert.Null(empty.Top1Share);
        Assert.Null(empty.Top10Share);
        Assert.Null(empty.Top10NetFlow);
        Assert.Equal(0, empty.LargeHolders);

        var ledger = new HolderLedger();
        ledger.Add("a", 5m, T0);
        Assert.Null(ledger.Snapshot(null, 0.02, incomplete: false).LargeHolders);
    }

    [Fact]
    public void Large_holder_share_is_configurable()
    {
        TokenState state = State(largeShare: 0.05);
        state.Apply(Create("dev", 60m)); // 6 % of 1 000
        state.Apply(Ev(EventKind.Buy, "a", 40m, 1)); // 4 %

        Assert.Equal(1, state.Snapshot(T0.AddSeconds(2)).HolderSupply!.LargeHolders);
    }

    // --- net flow of the top holders over the window ---

    [Fact]
    public void Top10_net_flow_is_negative_when_top_holders_distribute_within_the_window()
    {
        TokenState state = State(window: 60);
        state.Apply(Create("dev", 100m));
        state.Apply(Ev(EventKind.Buy, "a", 100m, 1));
        state.Apply(Ev(EventKind.Sell, "dev", 50m, 100)); // inside the window ending at 120
        state.Apply(Transfer(110, new TokenTransfer("a", "b", 10m)));

        HolderSupplyFeatures h = state.Snapshot(T0.AddSeconds(120)).HolderSupply!;

        // dev 50, a 90, b 10 = 150 circulating; flows in (60, 120]: dev -50, a -10, b +10 → all three are top holders.
        Assert.Equal(-50.0 / 150.0, h.Top10NetFlow!.Value, 10);
        Assert.Equal(90.0 / 150.0, h.Top1Share!.Value, 10);
    }

    [Fact]
    public void Flows_older_than_the_window_no_longer_count()
    {
        TokenState state = State(window: 60);
        state.Apply(Create("dev", 100m));
        state.Apply(Ev(EventKind.Buy, "a", 100m, 1));

        Assert.Equal(1.0, state.Snapshot(T0.AddSeconds(2)).HolderSupply!.Top10NetFlow!.Value, 10);
        Assert.Equal(0.0, state.Snapshot(T0.AddSeconds(200)).HolderSupply!.Top10NetFlow!.Value, 10);
    }

    // --- incomplete-history flag ---

    [Fact]
    public void A_token_whose_create_was_not_seen_is_flagged_incomplete()
    {
        TokenState state = State();
        state.Apply(Ev(EventKind.Buy, "a", 10m, 0));
        Assert.True(state.Snapshot(T0.AddSeconds(1)).HolderSupply!.Incomplete);

        TokenState complete = State();
        complete.Apply(Create("dev", 10m));
        Assert.False(complete.Snapshot(T0.AddSeconds(1)).HolderSupply!.Incomplete);
    }

    // --- point in time ---

    [Fact]
    public void A_snapshot_never_sees_positions_from_events_after_its_ts()
    {
        TokenState state = State();
        state.Apply(Create("dev", 100m));
        state.Apply(Ev(EventKind.Buy, "whale", 900m, 10)); // queued, dated after the first snapshot

        HolderSupplyFeatures before = state.Snapshot(T0.AddSeconds(5)).HolderSupply!;
        HolderSupplyFeatures after = state.Snapshot(T0.AddSeconds(10)).HolderSupply!;

        Assert.Equal(1, before.Holders);
        Assert.Equal(1.0, before.Top1Share!.Value, 10);
        Assert.Equal(2, after.Holders);
        Assert.Equal(0.9, after.Top1Share!.Value, 10);
    }

    [Fact]
    public void Live_and_replay_orders_give_the_same_features()
    {
        // Live: events arrive one by one with a snapshot every second; replay: all queued up front, out of order.
        RawEvent[] events =
        [
            Create("dev", 100m),
            Ev(EventKind.Buy, "a", 40m, 1),
            Transfer(2, new TokenTransfer("dev", "b", 25m)),
            Ev(EventKind.Sell, "a", 15m, 3),
            Ev(EventKind.Buy, "c", 60m, 3),
        ];

        TokenState live = State();
        for (int i = 0; i < events.Length; i++)
        {
            live.Apply(events[i]);
            live.Snapshot(events[i].Timestamp);
        }

        TokenState replay = State();
        foreach (RawEvent e in events.Reverse())
        {
            replay.Apply(e);
        }

        DateTimeOffset now = T0.AddSeconds(4);
        Assert.Equal(live.Snapshot(now).HolderSupply, replay.Snapshot(now).HolderSupply);
    }

    // --- memory bound ---

    [Fact]
    public void Ledger_stays_bounded_and_keeps_the_largest_holders()
    {
        var ledger = new HolderLedger(maxWallets: 16);
        ledger.Add("whale", 1_000m, T0);
        for (int i = 0; i < 100; i++)
        {
            ledger.Add($"small{i}", 1m, T0);
        }

        Assert.True(ledger.Count <= 16);
        Assert.True(ledger.Truncated);
        Assert.Equal(1_000m, ledger.PositionOf("whale"));
        Assert.Equal(1_100m, ledger.Circulating); // evicted tokens still circulate
        Assert.Equal(1_000.0 / 1_100.0, ledger.Snapshot(null, 0.02, false).Top1Share!.Value, 10);
    }

    [Fact]
    public void Eviction_drops_empty_positions_before_holders()
    {
        var ledger = new HolderLedger(maxWallets: 12);
        for (int i = 0; i < 12; i++)
        {
            ledger.Add($"flat{i}", 5m, T0);
            ledger.Add($"flat{i}", -5m, T0);
        }

        ledger.Add("holder", 3m, T0);

        Assert.False(ledger.Truncated);
        Assert.Equal(1, ledger.Count);
    }

    [Fact]
    public void Ledger_rejects_a_bound_below_top_k() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new HolderLedger(maxWallets: 5));

    // --- parsing ---

    [Fact]
    public void Transfer_legs_of_the_event_mint_are_parsed()
    {
        RawEvent? e = PumpApiEventParser.Parse("""
        {"txType":"transfer","timestamp":1700000000000,"txSigner":"A",
         "transfers":[{"from":"A","to":"B","amount":18813.425869,"isSolana":false,"mint":"MINTX"},
                      {"from":"A","to":"B","amount":0.008,"isSolana":true,"mint":"So11111111111111111111111111111111111111111"},
                      {"from":"A","to":"C","amount":5,"isSolana":false,"mint":"OTHER"},
                      {"from":"A","to":"A","amount":5,"isSolana":false,"mint":"MINTX"},
                      {"from":"A","to":"D","amount":0,"isSolana":false,"mint":"MINTX"},
                      {"to":"D","amount":3,"isSolana":false,"mint":"MINTX"},
                      {"from":"C","to":"D","amount":2,"mint":"MINTX"}]}
        """);

        Assert.NotNull(e);
        Assert.Equal("MINTX", e!.Mint);
        Assert.Equal([new TokenTransfer("A", "B", 18813.425869m), new TokenTransfer("C", "D", 2m)], e.TokenTransfers!);
    }

    [Fact]
    public void Trades_carry_no_transfer_legs()
    {
        RawEvent? e = PumpApiEventParser.Parse("""
        {"txType":"buy","mint":"M","timestamp":1700000000000,"txSigner":"A","tokenAmount":5,"solAmount":1,
         "transfers":[{"from":"A","to":"B","amount":5,"mint":"M"}]}
        """);

        Assert.Null(e!.TokenTransfers);
    }
}
