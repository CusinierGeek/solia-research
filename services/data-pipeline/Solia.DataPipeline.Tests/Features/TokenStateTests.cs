using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;

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
    public void Quote_currency_change_restarts_the_price_series_and_skips_cross_currency_growth()
    {
        var state = new TokenState("M", windowSeconds: 300, emaSpan: 20, rsiPeriod: 2, firstSeen: T0);
        state.Apply(Quoted(EventKind.Buy, 0.0000001m, 1m, Wsol, 0));
        state.Apply(Quoted(EventKind.Buy, 0.0000001m, 1m, Wsol, 1));
        state.Snapshot(T0.AddSeconds(1));

        state.Apply(Quoted(EventKind.Buy, 0.00002m, 50m, Usdc, 10)); // migrated to a USDC pool
        FeatureSnapshot s = state.Snapshot(T0.AddSeconds(10));

        Assert.Equal(0.00002m, s.Price);
        Assert.Equal(0.00002, s.Ema!.Value, 15); // restarted, not blended with SOL prices
        Assert.Equal(0, s.Volatility);           // one USDC price: no fake 200x jump
        Assert.Null(s.Rsi);
        Assert.Equal(50m, s.Volume);             // SOL amounts aren't summed with USDC
        Assert.Equal(3, s.BuyEvents);            // counts still include every trade
        Assert.Null(s.VolumeGrowthRate);
        Assert.Null(s.PriceDerivative);
        Assert.Null(s.TrendingScore);
        Assert.Equal(Usdc, s.OnchainSignals["quoteMint"]);
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
}
