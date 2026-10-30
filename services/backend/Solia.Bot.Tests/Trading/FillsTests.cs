using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

/// <summary>The fill model on known numbers: liquidity gate per venue, pool cap, x·y=k impact and fees.</summary>
public class FillsTests
{
    private const decimal Mark = 0.0001m;
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static readonly BotOptions Defaults = PolicyFixtures.RealisticOptions();

    /// <summary>The defaults without fees, so the price impact is observable alone.</summary>
    private static BotOptions NoFees()
    {
        BotOptions options = PolicyFixtures.RealisticOptions();
        options.FeePct.Clear();
        return options;
    }

    private static TradeOpening Opened(decimal size, decimal entryPrice) => new(
        "mint1", T0, entryPrice, size, ExitLevels.Compute(PolicyFixtures.Buyable(), Mark, Defaults), PolicyFixtures.Buyable());

    [Theory]
    [InlineData("pumpswap", 1.0, true)]    // AMM at the 1 SOL default floor
    [InlineData("pumpswap", 0.99, false)]
    [InlineData("raydium", 0.03, false)]   // the dust pools of the 2026-10-30 trades
    [InlineData("RAYDIUM", 25.0, true)]    // venue keys are case-insensitive
    [InlineData(null, 0.5, false)]         // unknown venue: the default floor
    [InlineData(null, 5.0, true)]
    [InlineData("pumpfun", 0.0, true)]     // curve: real SOL ~0, prices off its virtual reserve
    [InlineData("bonk", 0.01, true)]
    public void The_liquidity_gate_applies_the_venue_floor(string? venue, double liquidity, bool tradeable)
    {
        Assert.Equal(tradeable, Fills.IsTradeable(new Pool(venue, (decimal)liquidity), Defaults));
    }

    [Theory]
    [InlineData("pumpfun")]
    [InlineData("pumpswap")]
    public void An_unknown_liquidity_is_never_tradeable(string venue)
    {
        Assert.False(Fills.IsTradeable(new Pool(venue, null), Defaults));
        Assert.Null(Fills.ReserveSol(new Pool(venue, null), Defaults));
        Assert.Equal(0m, Fills.MaxSize(new Pool(venue, null), Defaults));
    }

    [Fact]
    public void A_zero_floor_still_needs_a_positive_reserve()
    {
        BotOptions options = PolicyFixtures.RealisticOptions();
        options.LiquidityFloorSol["raydium"] = 0m;
        Assert.False(Fills.IsTradeable(new Pool("raydium", 0m), options)); // an empty AMM pool, no virtual SOL
        Assert.True(Fills.IsTradeable(new Pool("raydium", 0.001m), options));
    }

    [Fact]
    public void The_reserve_adds_the_curve_virtual_sol_only()
    {
        Assert.Equal(30.5m, Fills.ReserveSol(new Pool("pumpfun", 0.5m), Defaults));
        Assert.Equal(0.5m, Fills.ReserveSol(new Pool("pumpswap", 0.5m), Defaults));
        Assert.Equal(0.61m, Fills.MaxSize(new Pool("pumpfun", 0.5m), Defaults)); // 2 % of 30.5
        Assert.Equal(1m, Fills.MaxSize(new Pool("pumpswap", 50m), Defaults));
    }

    [Fact]
    public void A_buy_of_the_whole_reserve_pays_twice_the_mark()
    {
        // n/P · R/(R+n) tokens for n = R: half the tokens the mark promises.
        Assert.Equal(2 * Mark, Fills.BuyPrice(100m, Mark, new Pool("raydium", 100m), NoFees()));
    }

    [Fact]
    public void A_small_buy_barely_moves_the_price_and_a_large_one_moves_it_a_lot()
    {
        var pool = new Pool("raydium", 100m);
        decimal small = Fills.BuyPrice(0.01m, Mark, pool, NoFees());
        decimal large = Fills.BuyPrice(10m, Mark, pool, NoFees());

        Assert.Equal(Mark * 1.0001m, small); // impact = size / R
        Assert.Equal(Mark * 1.1m, large);
    }

    [Fact]
    public void A_sell_worth_the_whole_reserve_gets_half_its_mark_value()
    {
        // Entered at the mark (size 1, 10 000 tokens), sold at a mark where it's worth 100 SOL into a 100 SOL pool.
        TradeOpening entry = Opened(1m, Mark);
        Assert.Equal(Mark * 100m / 2m, Fills.SellPrice(entry, Mark * 100m, new Pool("raydium", 100m), NoFees()));
    }

    [Fact]
    public void A_small_sell_barely_moves_the_price_and_a_large_one_moves_it_a_lot()
    {
        var pool = new Pool("raydium", 100m);
        decimal small = Fills.SellPrice(Opened(0.01m, Mark), Mark, pool, NoFees());
        decimal large = Fills.SellPrice(Opened(10m, Mark), Mark, pool, NoFees());

        Assert.Equal(Mark * 100m / 100.01m, small);
        Assert.Equal(Mark * 100m / 110m, large);
    }

    [Fact]
    public void The_venue_fee_is_charged_on_each_side()
    {
        var deep = new Pool("pumpswap", 1_000_000_000m); // impact ~1e-9: the fee alone
        decimal buy = Fills.BuyPrice(1m, Mark, deep, Defaults);
        TradeOpening entry = Opened(1m, buy);
        decimal sell = Fills.SellPrice(entry, Mark, deep, Defaults);

        decimal proceeds = entry.ValueAt(sell);
        Assert.Equal(0.997m * 0.997m, Math.Round(proceeds, 6)); // PumpSwap 0.30 % per side
        Assert.Equal(0.0125, Fills.FeeOf("pumpfun", Defaults));
        Assert.Equal(0.0025, Fills.FeeOf("orca", Defaults));     // unlisted AMM: the default
        Assert.Equal(0.0025, Fills.FeeOf(null, Defaults));
    }

    [Fact]
    public void Without_a_default_an_unlisted_venue_has_no_floor_fee_or_virtual_sol()
    {
        BotOptions options = PolicyFixtures.RealisticOptions();
        options.LiquidityFloorSol.Remove(Fills.DefaultVenue);
        options.FeePct.Remove(Fills.DefaultVenue);

        Assert.Equal(0m, Fills.FloorOf("orca", options));
        Assert.Equal(0.0, Fills.FeeOf("orca", options));
        Assert.Equal(0m, Fills.VirtualSolOf("orca", options));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(null)]
    public void An_empty_or_unknown_pool_at_exit_pays_nothing(double? liquidity)
    {
        TradeOpening entry = Opened(2m, Mark);
        Assert.Equal(0m, Fills.SellPrice(entry, Mark * 7m, new Pool("raydium", (decimal?)liquidity), Defaults));
    }

    [Fact]
    public void The_proceeds_never_exceed_the_pool()
    {
        // A +656 % mark in a 0.03 SOL pool (the 2026-10-30 artifact): the pool can pay at most its 0.03 SOL.
        TradeOpening entry = Opened(2m, Mark);
        decimal proceeds = entry.ValueAt(Fills.SellPrice(entry, Mark * 7.56m, new Pool("raydium", 0.03m), NoFees()));
        Assert.True(proceeds < 0.03m, $"proceeds {proceeds}");
    }

    [Fact]
    public void A_buy_into_a_pool_without_reserve_throws()
    {
        Assert.Throws<ArgumentException>(() => Fills.BuyPrice(1m, Mark, new Pool("raydium", 0m), Defaults));
        Assert.Throws<ArgumentException>(() => Fills.BuyPrice(1m, Mark, default, Defaults));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fills.BuyPrice(0m, Mark, new Pool("raydium", 10m), Defaults));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fills.BuyPrice(1m, 0m, new Pool("raydium", 10m), Defaults));
    }
}
