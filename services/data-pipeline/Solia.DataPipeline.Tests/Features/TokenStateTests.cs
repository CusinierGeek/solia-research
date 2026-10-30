using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Tests.Features;

public class TokenStateTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private static RawEvent Trade(EventKind kind, decimal price, decimal quote, string trader, int offsetSec) =>
        new()
        {
            Mint = "M",
            Kind = kind,
            Timestamp = T0.AddSeconds(offsetSec),
            Venue = "pumpfun",
            Price = price,
            QuoteAmount = quote,
            Trader = trader,
        };

    [Fact]
    public void Snapshot_aggregates_window_trades()
    {
        var state = new TokenState("M", windowSeconds: 300, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        state.Apply(Trade(EventKind.Buy, 1.0m, 2m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1.1m, 3m, "B", 1));
        state.Apply(Trade(EventKind.Sell, 1.05m, 1m, "A", 2));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(2, s.BuyEvents);
        Assert.Equal(1, s.SellEvents);
        Assert.Equal(6m, s.Volume!.Value);          // 2 + 3 + 1
        Assert.Equal(2.0 / 3.0, s.BuySellRatio, 10);
        Assert.Equal(2, s.Holders);                 // A, B (A counted once)
        Assert.Equal(1.05m, s.Price!.Value);        // latest
        Assert.Equal("pumpfun", s.Venue);
        Assert.True(s.TokenAgeSec >= 3);
        Assert.NotNull(s.Ema);
    }

    [Fact]
    public void Second_snapshot_computes_growth_versus_previous()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 10m, "A", 0));
        state.Snapshot(T0); // establishes previous: volume 10, holders 1, price 1

        state.Apply(Trade(EventKind.Buy, 2m, 10m, "B", 10));
        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(0.1, s.HoldersGrowthRate!.Value, 10); // 1 -> 2 over 10s
        Assert.Equal(1.0, s.VolumeGrowthRate!.Value, 10);  // 10 -> 20 over 10s
        Assert.Equal(0.1, s.PriceDerivative!.Value, 10);   // 1 -> 2 over 10s
    }

    private const string Wsol = "So11111111111111111111111111111111111111112";
    private const string Usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

    private static RawEvent Quoted(EventKind kind, decimal price, decimal quote, string quoteMint, int offsetSec) =>
        Trade(kind, price, quote, "T", offsetSec) with { QuoteMint = quoteMint };

    [Fact]
    public void First_snapshot_leaves_trending_score_and_growth_rates_null()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 10m, "A", 0));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(1));

        Assert.Null(s.VolumeGrowthRate);
        Assert.Null(s.HoldersGrowthRate);
        Assert.Null(s.TrendingScore);
    }

    [Fact]
    public void Trending_score_is_volume_growth_times_buy_ratio_once_derivable()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 10m, "A", 0));
        state.Snapshot(T0);
        state.Apply(Trade(EventKind.Buy, 1m, 10m, "B", 10));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(1.0 * 1.0, s.TrendingScore!.Value, 10); // growth 1/s, all buys
    }

    [Fact]
    public void Tx_rate_of_a_token_younger_than_the_window_uses_its_observed_span()
    {
        var state = new TokenState("M", windowSeconds: 300, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        for (int i = 0; i < 60; i++)
        {
            state.Apply(Trade(EventKind.Buy, 1m, 1m, $"W{i}", i / 2));
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(30));

        Assert.Equal(2.0, s.TxCreationRate, 10); // 60 trades / 30 s, not / 300 s
    }

    [Fact]
    public void Late_trade_is_counted_in_the_window_but_does_not_replace_the_latest_price()
    {
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        state.Apply(Trade(EventKind.Buy, 2m, 1m, "A", 200));
        state.Snapshot(T0.AddSeconds(200));                    // T+200 is applied
        state.Apply(Trade(EventKind.Sell, 1m, 1m, "B", 190)); // arrives late, still inside the window
        state.Apply(Trade(EventKind.Buy, 9m, 1m, "C", 10));   // arrives late, outside the window

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(220));

        Assert.Equal(2m, s.Price);              // newest event's price wins
        Assert.Equal(1, s.BuyEvents);           // the T+10 trade is pruned even though it sits behind T+200
        Assert.Equal(1, s.SellEvents);
        Assert.Equal(2.0, s.Ema!.Value, 10);    // late prices don't enter the EMA
        Assert.Equal(T0.AddSeconds(200), state.LastActivity);
    }

    [Fact]
    public void Events_received_out_of_order_between_snapshots_are_applied_in_time_order()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 2m, 1m, "A", 20));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 10)); // received late, before any snapshot

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(30));

        Assert.Equal(2m, s.Price); // the newest by time wins
        Assert.Equal(2, s.BuyEvents);
        double alpha = 2.0 / 21.0;
        Assert.Equal(1.0 + (alpha * (2.0 - 1.0)), s.Ema!.Value, 10); // EMA built 1 then 2, as if in order
    }

    [Fact]
    public void An_event_dated_after_the_snapshot_is_left_out_of_every_field_until_its_time()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0) with { MarketCap = 10m, QuoteInPool = 5m, PoolId = "P1" });
        state.Apply(Trade(EventKind.Buy, 9m, 4m, "B", 15) with
        {
            MarketCap = 90m,
            QuoteInPool = 50m,
            Venue = "raydium",
            PoolId = "P2",
            IsMigrated = true,
            Supply = 7m,
        });

        FeatureSnapshot before = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(1m, before.Price);
        Assert.Equal(10m, before.MarketCap);
        Assert.Equal(5m, before.Liquidity);
        Assert.Equal("pumpfun", before.Venue);
        Assert.Equal("P1", before.PoolAddress);
        Assert.False(before.IsMigrated);
        Assert.Null(before.OnchainSignals["supply"]);
        Assert.Equal(1, before.Holders);
        Assert.Equal(1, before.BuyEvents);
        Assert.Equal(1m, before.Volume);
        Assert.Equal(1.0, before.Ema!.Value, 10);
        Assert.Equal(T0.AddSeconds(15), state.LastActivity); // still known to the store (pruning, recency)

        FeatureSnapshot after = state.Snapshot(T0.AddSeconds(20));

        Assert.Equal(9m, after.Price);
        Assert.Equal(90m, after.MarketCap);
        Assert.Equal(50m, after.Liquidity);
        Assert.Equal("raydium", after.Venue);
        Assert.Equal("P2", after.PoolAddress);
        Assert.True(after.IsMigrated);
        Assert.Equal(7m, after.OnchainSignals["supply"]);
        Assert.Equal(2, after.Holders);
        Assert.Equal(2, after.BuyEvents); // applied once, not lost or double-counted
        Assert.Equal(5m, after.Volume);

        Assert.Equal(2, state.Snapshot(T0.AddSeconds(30)).BuyEvents);
    }

    [Fact]
    public void A_token_whose_only_event_is_in_the_future_has_no_activity_yet()
    {
        var state = new TokenState("M", 300, 20, 14, T0.AddSeconds(50));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 50));

        Assert.False(state.HasActivityInWindow(T0.AddSeconds(10)));
        Assert.True(state.HasActivityInWindow(T0.AddSeconds(50)));
    }

    [Fact]
    public void Trades_after_the_snapshot_time_are_not_counted()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 50));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(1, s.BuyEvents);
    }

    [Fact]
    public void Volatility_and_rsi_use_only_prices_inside_the_window()
    {
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 2, firstSeen: T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));   // wild old swings
        state.Apply(Trade(EventKind.Buy, 10m, 1m, "A", 1));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 2));
        state.Apply(Trade(EventKind.Buy, 5m, 1m, "A", 3600)); // an hour later: flat
        state.Apply(Trade(EventKind.Buy, 5m, 1m, "A", 3601));
        state.Apply(Trade(EventKind.Buy, 5m, 1m, "A", 3602));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3602));

        Assert.Equal(0, s.Volatility);
        Assert.Equal(50, s.Rsi); // flat -> neutral
    }

    [Fact]
    public void A_trade_in_another_quote_currency_leaves_the_SOL_series_alone()
    {
        var state = new TokenState("M", windowSeconds: 300, emaSpan: 20, rsiPeriod: 2, firstSeen: T0);
        state.Apply(Quoted(EventKind.Buy, 0.0000001m, 1m, Wsol, 0));
        RawEvent solPool = Quoted(EventKind.Buy, 0.0000001m, 1m, Wsol, 1);
        state.Apply(solPool with { MarketCap = 30m, QuoteInPool = 80m, Venue = "pumpswap", PoolId = "SOLPOOL" });
        state.Snapshot(T0.AddSeconds(1));

        RawEvent usdcPool = Quoted(EventKind.Buy, 0.00002m, 50m, Usdc, 10); // the same token on a USDC pool
        state.Apply(usdcPool with { MarketCap = 5000m, QuoteInPool = 9000m, Venue = "meteora", PoolId = "USDCPOOL", Supply = 1e9m });
        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(0.0000001m, s.Price);   // a USDC price is another unit, not a 200x jump
        Assert.Equal(0, s.Volatility);
        Assert.Equal(0, s.PriceDerivative!.Value, 15);
        Assert.Equal(30m, s.MarketCap);      // cap, liquidity, venue and pool stay the SOL pool's
        Assert.Equal(80m, s.Liquidity);
        Assert.Equal("pumpswap", s.Venue);
        Assert.Equal("SOLPOOL", s.PoolAddress);
        Assert.Equal(2m, s.Volume);          // USDC amounts aren't summed with SOL
        Assert.Equal(3, s.BuyEvents);        // counts still include every trade
        Assert.Equal(1e9m, s.OnchainSignals["supply"]); // the supply doesn't depend on the pool
        Assert.Equal(Wsol, s.OnchainSignals["quoteMint"]);
    }

    [Fact]
    public void A_token_trading_on_several_pools_keeps_one_continuous_SOL_price()
    {
        // Real shape (2026-10-28): one migrated token on SOL, USDC and token-quoted pools at once.
        var state = new TokenState("M", 300, emaSpan: 20, rsiPeriod: 2, T0);
        (string Quote, decimal Price)[] trades =
        [
            (Wsol, 0.000160m), (Usdc, 0.0165m), ("BgKtxB44DWf2zMtQpjwVNwjWCjKgAupoVdGVDuR9pump", 1480m),
            (Wsol, 0.000162m), (Usdc, 0.0166m), (Wsol, 0.000164m),
        ];
        for (int t = 0; t < trades.Length; t++)
        {
            state.Apply(Quoted(EventKind.Buy, trades[t].Price, 1m, trades[t].Quote, t));
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(trades.Length));

        Assert.Equal(0.000164m, s.Price);
        Assert.InRange(s.Ema!.Value, 0.000160, 0.000164);
        Assert.InRange(s.Volatility!.Value, 0, 0.02); // ~1.2 % steps, not 10^7x swings
        Assert.Equal(3m, s.Volume);
        Assert.Equal(6, s.BuyEvents);
    }

    [Fact]
    public void A_token_never_traded_against_SOL_has_no_price_but_counts_its_trades()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Quoted(EventKind.Buy, 0.02m, 50m, Usdc, 0) with { MarketCap = 5000m, QuoteInPool = 9000m });
        state.Apply(Quoted(EventKind.Sell, 0.02m, 10m, Usdc, 1));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(2));

        Assert.Null(s.Price);
        Assert.Null(s.Ema);
        Assert.Null(s.MarketCap);
        Assert.Null(s.Liquidity);
        Assert.Equal(0m, s.Volume);
        Assert.Equal(1, s.BuyEvents);
        Assert.Equal(1, s.SellEvents);
        Assert.Null(s.OnchainSignals["quoteMint"]);
    }

    [Fact]
    public void Repeated_snapshots_at_the_same_time_are_stable_apart_from_zero_interval_growth()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 10m, "A", 0));

        FeatureSnapshot first = state.Snapshot(T0.AddSeconds(5));
        FeatureSnapshot second = state.Snapshot(T0.AddSeconds(5));

        Assert.Equal(first.Volume, second.Volume);
        Assert.Equal(first.BuyEvents, second.BuyEvents);
        Assert.Equal(first.Holders, second.Holders);
    }

    [Fact]
    public void Observations_outside_the_window_are_dropped()
    {
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        state.Apply(Trade(EventKind.Buy, 1m, 5m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 7m, "B", 120)); // two minutes later

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(120));

        Assert.Equal(1, s.BuyEvents);        // the T0 trade is outside the 60s window
        Assert.Equal(7m, s.Volume!.Value);
    }

    // --- SOL-volume buy pressure (#111) ---

    private static RawEvent Sol(EventKind kind, decimal sol, int offsetSec) => Quoted(kind, 1m, sol, Wsol, offsetSec);

    [Fact]
    public void Buy_pressure_weights_trades_by_their_SOL_size()
    {
        var state = new TokenState("M", 300, 20, 14, T0, largeBuySol: 1m);
        state.Apply(Sol(EventKind.Buy, 0.5m, 0));
        state.Apply(Sol(EventKind.Buy, 3m, 1));   // the whale
        state.Apply(Sol(EventKind.Buy, 1m, 2));   // exactly the threshold: counts as large
        state.Apply(Sol(EventKind.Sell, 1.5m, 3));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(4));

        Assert.Equal(3m, s.NetSolFlow);                     // 4.5 bought - 1.5 sold
        Assert.Equal(0.75, s.BuyVolumeRatio!.Value, 10);    // 4.5 / 6
        Assert.Equal(1.5m, s.AvgBuySol);                    // 4.5 / 3 buys
        Assert.Equal(3m, s.MaxBuySol);
        Assert.Equal(2, s.LargeBuyCount);
    }

    [Fact]
    public void One_whale_and_ten_dust_buys_no_longer_look_alike()
    {
        var whale = new TokenState("M", 300, 20, 14, T0);
        whale.Apply(Sol(EventKind.Buy, 5m, 0));
        var dust = new TokenState("M", 300, 20, 14, T0);
        for (int i = 0; i < 10; i++)
        {
            dust.Apply(Sol(EventKind.Buy, 0.01m, i));
        }

        FeatureSnapshot w = whale.Snapshot(T0.AddSeconds(10));
        FeatureSnapshot d = dust.Snapshot(T0.AddSeconds(10));

        Assert.Equal(w.BuySellRatio, d.BuySellRatio);       // count-based: identical
        Assert.Equal(5m, w.NetSolFlow);
        Assert.Equal(0.1m, d.NetSolFlow);
        Assert.Equal(1, w.LargeBuyCount);
        Assert.Equal(0, d.LargeBuyCount);
    }

    [Fact]
    public void Empty_window_has_no_buy_pressure()
    {
        var state = new TokenState("M", 60, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 2m, 0));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(120)); // the buy has left the window

        Assert.Equal(0m, s.NetSolFlow);
        Assert.Null(s.BuyVolumeRatio);
        Assert.Null(s.AvgBuySol);
        Assert.Null(s.MaxBuySol);
        Assert.Equal(0, s.LargeBuyCount);
    }

    [Fact]
    public void SOL_trades_without_an_amount_count_as_trades_but_not_as_SOL()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 2m, 0));
        state.Apply(Sol(EventKind.Buy, 0m, 1) with { QuoteAmount = null });
        state.Apply(Sol(EventKind.Sell, 0m, 2) with { QuoteAmount = null });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(2, s.BuyEvents);
        Assert.Equal(2m, s.NetSolFlow);
        Assert.Equal(1.0, s.BuyVolumeRatio!.Value, 10);
        Assert.Equal(2m, s.AvgBuySol);                      // not 1: the amount-less buy is skipped
        Assert.Equal(2m, s.MaxBuySol);
    }

    [Fact]
    public void Only_amountless_SOL_buys_leave_buy_sizes_null()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 0m, 0) with { QuoteAmount = null });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(1));

        Assert.Equal(1, s.BuyEvents);
        Assert.Null(s.BuyVolumeRatio);
        Assert.Null(s.AvgBuySol);
        Assert.Null(s.MaxBuySol);
        Assert.Equal(0, s.LargeBuyCount);
    }

    [Fact]
    public void Wsol_mint_is_the_shared_ingestion_constant() => Assert.Equal(Wsol, RawEvent.WrappedSolMint);

    [Fact]
    public void Sells_only_window_has_negative_flow_and_no_buy_sizes()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Sell, 2m, 0));
        state.Apply(Sol(EventKind.Sell, 0.5m, 1));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(2));

        Assert.Equal(-2.5m, s.NetSolFlow);
        Assert.Equal(0.0, s.BuyVolumeRatio);
        Assert.Null(s.AvgBuySol);
        Assert.Null(s.MaxBuySol);
        Assert.Equal(0, s.LargeBuyCount);
    }

    [Fact]
    public void Buys_only_window_has_a_full_buy_volume_ratio()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 0.2m, 0));
        state.Apply(Sol(EventKind.Buy, 0.4m, 1));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(2));

        Assert.Equal(0.6m, s.NetSolFlow);
        Assert.Equal(1.0, s.BuyVolumeRatio);
        Assert.Equal(0.3m, s.AvgBuySol);
        Assert.Equal(0.4m, s.MaxBuySol);
    }

    [Fact]
    public void Non_SOL_and_unknown_quote_trades_are_left_out_of_buy_pressure()
    {
        var state = new TokenState("M", 300, 20, 14, T0, largeBuySol: 1m);
        state.Apply(Sol(EventKind.Buy, 0.5m, 0));
        state.Apply(Quoted(EventKind.Buy, 1m, 500m, Usdc, 1));  // 500 USDC is not 500 SOL
        state.Apply(Quoted(EventKind.Sell, 1m, 200m, Usdc, 2));
        state.Apply(Trade(EventKind.Buy, 1m, 9m, "T", 3));      // no quote mint: unknown currency

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(4));

        Assert.Equal(4, s.BuyEvents + s.SellEvents);            // still counted as trades
        Assert.Equal(0.5m, s.NetSolFlow);
        Assert.Equal(1.0, s.BuyVolumeRatio);
        Assert.Equal(0.5m, s.AvgBuySol);
        Assert.Equal(0.5m, s.MaxBuySol);
        Assert.Equal(0, s.LargeBuyCount);
    }

    [Fact]
    public void Volume_and_buy_pressure_only_count_SOL_trades()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 2m, 0));
        state.Apply(Quoted(EventKind.Buy, 1m, 100m, Usdc, 1)); // 100 USDC is not 100 SOL

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(2));

        Assert.Equal(2m, s.Volume);
        Assert.Equal(2m, s.NetSolFlow);
    }

    [Fact]
    public void Late_trades_inside_the_window_count_towards_buy_pressure()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Sol(EventKind.Buy, 1m, 10));
        state.Apply(Sol(EventKind.Sell, 0.25m, 5)); // out of order

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(11));

        Assert.Equal(0.75m, s.NetSolFlow);
        Assert.Equal(0.8, s.BuyVolumeRatio!.Value, 10);
    }

    // --- Momentum & acceleration (default windows 10/30/60 s) ---

    /// <summary>One buy per second over [0, seconds], with the given price and quote-amount curves.</summary>
    private static TokenState Curve(Func<int, double> price, Func<int, decimal> quote, int seconds,
        MomentumWindows? windows = null)
    {
        var state = new TokenState("M", 300, 20, 14, T0, momentum: windows);
        for (int t = 0; t <= seconds; t++)
        {
            state.Apply(Trade(EventKind.Buy, (decimal)price(t), quote(t), "A", t));
        }

        return state;
    }

    [Fact]
    public void Flat_curve_has_no_momentum()
    {
        FeatureSnapshot s = Curve(_ => 1, _ => 1m, 60).Snapshot(T0.AddSeconds(60));

        Assert.Equal(0, s.ReturnShort!.Value, 12);
        Assert.Equal(0, s.ReturnMedium!.Value, 12);
        Assert.Equal(0, s.ReturnLong!.Value, 12);
        Assert.Equal(0, s.PriceAcceleration!.Value, 12);
        Assert.Equal(0, s.VolumeAcceleration!.Value, 12);
        Assert.Equal(1, s.VolumeBurstRatio, 12); // 10 trades / 10 s vs 60 trades / 60 s
    }

    [Fact]
    public void Linear_curve_returns_and_accelerations()
    {
        // price 1 + 0.1 t -> p(60) = 7, p(50) = 6, p(40) = 5, p(30) = 4, p(0) = 1; volume t per second.
        FeatureSnapshot s = Curve(t => 1 + (0.1 * t), t => t, 60).Snapshot(T0.AddSeconds(60));

        Assert.Equal((7.0 / 6) - 1, s.ReturnShort!.Value, 10);
        Assert.Equal(0.75, s.ReturnMedium!.Value, 10);
        Assert.Equal(6, s.ReturnLong!.Value, 10);
        Assert.Equal((1.0 / 6) - (1.0 / 5), s.PriceAcceleration!.Value, 10); // a constant slope decelerates relatively
        Assert.Equal(1, s.VolumeAcceleration!.Value, 10);                     // (sum 51..60 - sum 41..50) / 10 / 10
        Assert.Equal(55.5 / 30.5, s.VolumeBurstRatio, 10);                    // 555 / 10 vs 1830 / 60
    }

    [Fact]
    public void Exponential_price_has_constant_returns_and_no_acceleration()
    {
        FeatureSnapshot s = Curve(t => Math.Pow(2, t / 10.0), _ => 1m, 60).Snapshot(T0.AddSeconds(60));

        Assert.Equal(1, s.ReturnShort!.Value, 9); // doubles every 10 s
        Assert.Equal(7, s.ReturnMedium!.Value, 9);
        Assert.Equal(0, s.PriceAcceleration!.Value, 9);
    }

    [Fact]
    public void Accelerating_curve_has_positive_accelerations_and_a_burst()
    {
        // price exp(t^2 / 1000): r(50->60) = e^1.1 - 1 vs r(40->50) = e^0.9 - 1; volume t^2.
        FeatureSnapshot s = Curve(t => Math.Exp(t * t / 1000.0), t => t * t, 60).Snapshot(T0.AddSeconds(60));

        Assert.Equal(Math.Exp(1.1) - Math.Exp(0.9), s.PriceAcceleration!.Value, 9);
        Assert.True(s.PriceAcceleration > 0);
        Assert.True(s.VolumeAcceleration > 0);
        Assert.True(s.VolumeBurstRatio > 1);
    }

    [Fact]
    public void Single_trade_has_no_momentum_history()
    {
        FeatureSnapshot s = Curve(_ => 1, _ => 1m, 0).Snapshot(T0);

        Assert.Null(s.ReturnShort);
        Assert.Null(s.ReturnMedium);
        Assert.Null(s.ReturnLong);
        Assert.Null(s.PriceAcceleration);
        Assert.Null(s.EmaSlope);
        Assert.Null(s.VolumeAcceleration);
        Assert.Equal(1, s.VolumeBurstRatio); // a lone trade is both windows' whole volume
    }

    [Fact]
    public void Insufficient_history_leaves_the_longer_look_backs_null()
    {
        FeatureSnapshot s = Curve(t => 1 + (0.1 * t), _ => 1m, 15).Snapshot(T0.AddSeconds(15));

        Assert.Equal((2.5 / 1.5) - 1, s.ReturnShort!.Value, 10); // p(15) / p(5)
        Assert.Null(s.ReturnMedium);
        Assert.Null(s.ReturnLong);
        Assert.Null(s.PriceAcceleration);  // needs p(-5)
        Assert.Null(s.VolumeAcceleration); // 15 s observed < two short windows
        Assert.Equal(15.0 / 16, s.VolumeBurstRatio, 10); // 10 trades / 10 s vs 16 trades over 15 observed s
    }

    [Fact]
    public void No_volume_over_the_long_window_is_a_neutral_burst_ratio()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 5m, "A", 0));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(100)); // last trade is older than the 60 s long window

        Assert.Equal(1, s.VolumeBurstRatio);
        Assert.Equal(0, s.VolumeAcceleration!.Value);
    }

    [Fact]
    public void A_long_look_back_of_the_whole_window_survives_pruning()
    {
        // Rolling window = long window = 60 s: the price in force at now - 60 is older than the cutoff.
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        for (int t = 0; t <= 120; t += 7)
        {
            state.Apply(Trade(EventKind.Buy, 1 + (0.1m * t), 1m, "A", t)); // no trade exactly at 60
        }

        for (int now = 70; now <= 120; now += 10)
        {
            state.Snapshot(T0.AddSeconds(now)); // prunes as the live cadence would
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(120));

        Assert.Equal((12.9 / 6.6) - 1, s.ReturnLong!.Value, 10);          // p(119) / p(56)
        Assert.Equal((12.9 / 11.5) - 1, s.ReturnShort!.Value, 10);        // p(119) / p(105)
        Assert.NotNull(s.PriceAcceleration);
        Assert.Equal(60, s.PriceHistorySec);
    }

    [Fact]
    public void A_busy_token_keeps_its_long_look_back_beyond_the_volatility_point_cap()
    {
        // 20 prices per second for 60 s = 1220 points, past the 512-point volatility series cap.
        var state = new TokenState("M", 300, 20, 14, T0);
        for (int i = 0; i <= 1200; i++)
        {
            var e = Trade(EventKind.Buy, 1 + (0.001m * i), 1m, "A", 0) with { Timestamp = T0.AddMilliseconds(50 * i) };
            state.Apply(e);
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(60));

        Assert.Equal((2.2 / 1.0) - 1, s.ReturnLong!.Value, 10);           // p(60 s) = 1 + 1.2, p(0) = 1
        Assert.Equal((2.2 / 2.0) - 1, s.ReturnShort!.Value, 10);          // p(50 s) = 1 + 1.0
        Assert.Equal(60, s.PriceHistorySec);
    }

    [Fact]
    public void Ema_and_its_slope_ignore_prices_after_the_snapshot_time()
    {
        var withFuture = new TokenState("M", 300, 20, 14, T0);
        var without = new TokenState("M", 300, 20, 14, T0);
        foreach (TokenState state in new[] { withFuture, without })
        {
            state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));
            state.Snapshot(T0);
            state.Apply(Trade(EventKind.Buy, 2m, 1m, "A", 5));
        }

        withFuture.Apply(Trade(EventKind.Buy, 100m, 1m, "A", 15)); // applied before the 10 s snapshot

        FeatureSnapshot leaked = withFuture.Snapshot(T0.AddSeconds(10));
        FeatureSnapshot expected = without.Snapshot(T0.AddSeconds(10));

        Assert.Equal(expected.Ema, leaked.Ema);
        Assert.Equal(expected.EmaSlope!.Value, leaked.EmaSlope!.Value, 12);
        Assert.Equal(expected.ReturnShort, leaked.ReturnShort);
    }

    [Fact]
    public void Price_history_tells_a_young_token_apart_from_a_flat_price()
    {
        FeatureSnapshot young = Curve(_ => 1, _ => 1m, 15).Snapshot(T0.AddSeconds(15));
        FeatureSnapshot flat = Curve(_ => 1, _ => 1m, 90).Snapshot(T0.AddSeconds(90));
        FeatureSnapshot none = new TokenState("M", 300, 20, 14, T0).Snapshot(T0);

        Assert.Null(young.ReturnLong);
        Assert.Equal(15, young.PriceHistorySec);
        Assert.Equal(0, flat.ReturnLong!.Value, 12);
        Assert.Equal(60, flat.PriceHistorySec); // capped at the long window
        Assert.Equal(0, none.PriceHistorySec);
    }

    [Fact]
    public void Volume_momentum_ignores_trades_in_another_quote_currency()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        for (int t = 0; t <= 50; t++)
        {
            state.Apply(Quoted(EventKind.Buy, 1m, 1m, Wsol, t));
        }

        for (int t = 51; t <= 60; t++)
        {
            state.Apply(Quoted(EventKind.Buy, 7m, 1m, Usdc, t)); // only USDC trades since 51 s
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(60));

        Assert.Equal(-0.1, s.VolumeAcceleration!.Value, 12); // 0 SOL in (50, 60] vs 10 in (40, 50]
        Assert.Equal(0, s.VolumeBurstRatio, 12);             // no SOL traded in the short window
        Assert.Equal(0, s.ReturnLong!.Value, 12);            // the SOL price never moved
        Assert.Equal(60, s.PriceHistorySec);                 // and its history wasn't restarted
    }

    [Fact]
    public void Momentum_windows_are_configurable()
    {
        FeatureSnapshot s = Curve(t => 1 + (0.1 * t), _ => 1m, 60, new MomentumWindows(5, 10, 20))
            .Snapshot(T0.AddSeconds(60));

        Assert.Equal((7.0 / 6.5) - 1, s.ReturnShort!.Value, 10);
        Assert.Equal((7.0 / 6.0) - 1, s.ReturnMedium!.Value, 10);
        Assert.Equal((7.0 / 5.0) - 1, s.ReturnLong!.Value, 10);
        Assert.Equal((7.0 / 6.5) - (6.5 / 6.0), s.PriceAcceleration!.Value, 10);
    }

    [Fact]
    public void Ema_slope_is_the_relative_per_second_ema_change()
    {
        var state = new TokenState("M", 300, emaSpan: 1, rsiPeriod: 14, T0); // span 1: the EMA is the price
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));
        Assert.Null(state.Snapshot(T0).EmaSlope);

        state.Apply(Trade(EventKind.Buy, 2m, 1m, "A", 10));
        Assert.Equal(0.1, state.Snapshot(T0.AddSeconds(10)).EmaSlope!.Value, 10); // (2 - 1) / 10 s / 1

        Assert.Equal(0, state.Snapshot(T0.AddSeconds(20)).EmaSlope!.Value, 10); // flat since
    }

    [Fact]
    public void Another_quote_currency_trade_does_not_move_the_momentum_series()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        for (int t = 0; t <= 30; t++)
        {
            state.Apply(Quoted(EventKind.Buy, 1m, 1m, Wsol, t));
        }

        state.Snapshot(T0.AddSeconds(30));
        state.Apply(Quoted(EventKind.Buy, 0.00002m, 50m, Usdc, 31));
        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(31));

        Assert.Equal(0, s.ReturnShort!.Value, 12); // USDC prices aren't compared with SOL ones
        Assert.Equal(0, s.EmaSlope!.Value, 12);
    }

    // --- Holder-concentration proxy (txSigner) ---

    [Fact]
    public void Single_wallet_window_is_maximally_concentrated()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 2m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 3m, "A", 5));
        state.Apply(Trade(EventKind.Sell, 1m, 1m, "A", 8));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(9));

        Assert.Equal(1, s.UniqueTraders);
        Assert.Equal(1.0, s.TopTraderShare!.Value, 10);
        Assert.Equal(1.0, s.TraderHhi!.Value, 10);
    }

    [Fact]
    public void Uniform_wallets_spread_the_volume_evenly()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        string[] traders = ["A", "B", "C", "D"];
        for (int i = 0; i < traders.Length; i++)
        {
            state.Apply(Trade(EventKind.Buy, 1m, 5m, traders[i], i));
        }

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(4));

        Assert.Equal(4, s.UniqueTraders);
        Assert.Equal(0.25, s.TopTraderShare!.Value, 10);
        Assert.Equal(0.25, s.TraderHhi!.Value, 10); // 1/n
    }

    [Fact]
    public void Skewed_wallets_weight_shares_by_volume()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 2m, "A", 0));
        state.Apply(Trade(EventKind.Sell, 1m, 1m, "A", 1)); // buys and sells both count: A = 3
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 2));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(2, s.UniqueTraders);
        Assert.Equal(0.75, s.TopTraderShare!.Value, 10);
        Assert.Equal((0.75 * 0.75) + (0.25 * 0.25), s.TraderHhi!.Value, 10);
    }

    [Fact]
    public void Returning_traders_are_told_apart_from_new_ones()
    {
        var state = new TokenState("M", 300, 20, 14, T0); // short momentum window: 10 s
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 2));
        state.Apply(Trade(EventKind.Sell, 1m, 1m, "A", 25)); // returning: seen at 0
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "C", 26));  // new
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "C", 28));  // still new: its first trade is recent

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(30));

        Assert.Equal(3, s.UniqueTraders);
        Assert.Equal(0.5, s.NewTraderRatio!.Value, 10); // recent {A, C}, new {C}
    }

    [Fact]
    public void Returning_detection_tolerates_out_of_order_trades()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 25));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0)); // late: A's earlier trade

        Assert.Equal(0.0, state.Snapshot(T0.AddSeconds(30)).NewTraderRatio!.Value, 10);
    }

    [Fact]
    public void New_trader_ratio_is_null_without_a_recent_signed_trade()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(30)); // nothing in the last 10 s

        Assert.Null(s.NewTraderRatio);
        Assert.Equal(1, s.UniqueTraders);
    }

    [Fact]
    public void Only_new_traders_in_a_fresh_token_give_a_ratio_of_one()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 1));

        Assert.Equal(1.0, state.Snapshot(T0.AddSeconds(2)).NewTraderRatio!.Value, 10);
    }

    [Fact]
    public void Missing_signers_are_skipped_not_lumped_into_one_wallet()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 100m, "X", 0) with { Trader = null });
        state.Apply(Trade(EventKind.Buy, 1m, 100m, "X", 1) with { Trader = null });
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 2));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 3));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(4));

        Assert.Equal(4, s.BuyEvents); // still trades
        Assert.Equal(2, s.UniqueTraders);
        Assert.Equal(0.5, s.TopTraderShare!.Value, 10); // of the signed volume only
        Assert.Equal(0.5, s.TraderHhi!.Value, 10);
        Assert.Equal(1.0, s.NewTraderRatio!.Value, 10);
    }

    [Fact]
    public void Window_without_signers_leaves_concentration_undefined()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 5m, "X", 0) with { Trader = null });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(1));

        Assert.Equal(0, s.UniqueTraders);
        Assert.Null(s.TopTraderShare);
        Assert.Null(s.TraderHhi);
        Assert.Null(s.NewTraderRatio);
    }

    [Fact]
    public void Signed_trades_without_volume_count_traders_but_leave_shares_undefined()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 0m, "A", 0) with { QuoteAmount = null });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(1));

        Assert.Equal(1, s.UniqueTraders);
        Assert.Null(s.TopTraderShare);
        Assert.Null(s.TraderHhi);
    }

    [Fact]
    public void Per_wallet_volume_is_scoped_to_the_window_but_newness_to_the_token()
    {
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        state.Apply(Trade(EventKind.Buy, 1m, 100m, "WHALE", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 70));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "WHALE", 75));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(76)); // the whale's first buy has left the window

        Assert.Equal(2, s.UniqueTraders);
        Assert.Equal(0.5, s.TopTraderShare!.Value, 10); // its 100 no longer counts
        Assert.Equal(0.5, s.NewTraderRatio!.Value, 10); // recent {A, WHALE}: the whale still returns
        Assert.Equal(2, s.Holders); // the lifetime holders proxy is unaffected
    }

    [Fact]
    public void Concentration_shares_only_sum_SOL_amounts()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Quoted(EventKind.Buy, 1m, 1m, Wsol, 0) with { Trader = "A" });
        state.Apply(Quoted(EventKind.Buy, 1m, 3m, Wsol, 1) with { Trader = "B" });
        state.Apply(Quoted(EventKind.Buy, 1m, 1000m, Usdc, 2) with { Trader = "U" });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(2, s.UniqueTraders); // U's USDC trade isn't counted, like its volume
        Assert.Equal(0.75, s.TopTraderShare!.Value, 10); // B: 3 of 4 SOL
    }

    [Fact]
    public void Another_quote_currency_trade_keeps_unique_traders_consistent_with_the_hhi()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        for (int i = 0; i < 10; i++)
        {
            state.Apply(Quoted(EventKind.Buy, 1m, 1m, Wsol, i) with { Trader = $"W{i}" });
        }

        state.Apply(Quoted(EventKind.Buy, 1m, 1m, Usdc, 20) with { Trader = "U" });

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(21));

        Assert.Equal(10, s.UniqueTraders); // the SOL traders; U's USDC trade has no SOL amount
        Assert.Equal(0.1, s.TraderHhi!.Value, 10); // 1/n with n = 10
    }

    [Fact]
    public void Another_quote_currency_trade_still_makes_a_trader_returning()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Quoted(EventKind.Buy, 1m, 5m, Wsol, 0) with { Trader = "A" });
        state.Apply(Quoted(EventKind.Buy, 1m, 1m, Usdc, 25) with { Trader = "A" });

        Assert.Equal(0.0, state.Snapshot(T0.AddSeconds(30)).NewTraderRatio!.Value, 10);
    }

    [Fact]
    public void Newness_falls_back_to_the_window_once_holders_are_a_sketch()
    {
        var state = new TokenState("M", windowSeconds: 60, emaSpan: 20, rsiPeriod: 14, firstSeen: T0);
        for (int i = 0; i <= HolderCounter.ExactLimit; i++)
        {
            state.Apply(Trade(EventKind.Buy, 1m, 1m, $"W{i}", 0));
        }

        state.Apply(Trade(EventKind.Buy, 1m, 1m, "W0", 75)); // first-seen forgotten, t=0 pruned

        Assert.Equal(1.0, state.Snapshot(T0.AddSeconds(76)).NewTraderRatio!.Value, 10);
    }

    [Fact]
    public void Non_positive_amounts_carry_no_volume_and_keep_shares_in_range()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 5m, "A", 0));
        state.Apply(Trade(EventKind.Sell, 1m, -5m, "B", 1));
        state.Apply(Trade(EventKind.Buy, 1m, 0m, "C", 2));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(3, s.UniqueTraders);
        Assert.Equal(1.0, s.TopTraderShare!.Value, 10); // A: all of the 5, not 5 of 0
        Assert.Equal(1.0, s.TraderHhi!.Value, 10);
    }

    [Fact]
    public void Snapshots_reuse_state_without_leaking_between_them()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 3m, "A", 0));
        state.Snapshot(T0.AddSeconds(1));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "B", 2));

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(3));

        Assert.Equal(2, s.UniqueTraders);
        Assert.Equal(0.75, s.TopTraderShare!.Value, 10); // A 3 of 4, not double-counted
    }

    // --- FE-T5 (#114): burned liquidity promoted to a column ---

    private static RawEvent PoolCreated(string burnedLiquidity, int offsetSec) =>
        PumpApiEventParser.Parse(
            $$"""{"action":"createPool","mint":"M","pool":"raydium-cpmm","burnedLiquidity":"{{burnedLiquidity}}","timestamp":{{T0.AddSeconds(offsetSec).ToUnixTimeMilliseconds()}}}""")!;

    private static RawEvent PoolEvent(string poolId, string? burnedLiquidity, int offsetSec)
    {
        string burned = burnedLiquidity is null ? "" : $",\"burnedLiquidity\":\"{burnedLiquidity}\"";
        return PumpApiEventParser.Parse(
            $$"""{"action":"createPool","mint":"M","pool":"raydium-cpmm","poolId":"{{poolId}}"{{burned}},"timestamp":{{T0.AddSeconds(offsetSec).ToUnixTimeMilliseconds()}}}""")!;
    }

    [Fact]
    public void Burned_liquidity_is_unknown_until_a_pool_creation_reports_it()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 0));

        Assert.Null(state.Snapshot(T0.AddSeconds(1)).BurnedLiquidity);
    }

    [Fact]
    public void Burned_liquidity_comes_from_the_pool_creation_and_persists()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(PoolCreated("57.5%", 0));
        state.Apply(Trade(EventKind.Buy, 1m, 1m, "A", 1)); // later trades don't report it

        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(2));

        Assert.Equal(57.5, s.BurnedLiquidity);
        Assert.Equal(57.5, s.OnchainSignals["burnedLiquidityPercent"]); // still mirrored in the JSONB
    }

    [Theory]
    [InlineData("lots")]
    [InlineData("150%")]
    [InlineData("NaN%")]
    public void Malformed_burned_liquidity_is_unknown_not_zero(string raw)
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(PoolCreated(raw, 0));

        Assert.Null(state.Snapshot(T0.AddSeconds(1)).BurnedLiquidity);
    }

    [Fact]
    public void A_malformed_later_report_keeps_the_last_valid_burned_liquidity()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(PoolCreated("100%", 0));
        state.Apply(PoolCreated("garbage", 1));

        Assert.Equal(100.0, state.Snapshot(T0.AddSeconds(2)).BurnedLiquidity);
    }

    [Fact]
    public void Moving_to_another_pool_forgets_the_previous_pools_burned_liquidity()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(PoolEvent("P1", "100%", 0));
        state.Apply(PoolEvent("P2", null, 1)); // the new pool doesn't report its burn

        Assert.Null(state.Snapshot(T0.AddSeconds(2)).BurnedLiquidity);
    }

    [Fact]
    public void A_new_pools_own_report_replaces_the_previous_pools_burned_liquidity()
    {
        var state = new TokenState("M", 300, 20, 14, T0);
        state.Apply(PoolEvent("P1", "100%", 0));
        state.Apply(PoolEvent("P2", "0%", 1));
        state.Apply(PoolEvent("P2", null, 2)); // same pool, no report: keeps its reading

        Assert.Equal(0.0, state.Snapshot(T0.AddSeconds(3)).BurnedLiquidity);
    }
}
