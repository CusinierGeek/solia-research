using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Tests.Policy;

public class PositionSizerTests
{
    private static readonly BotOptions Opts = PolicyFixtures.Options(); // RiskFraction 0.5, cap 0.2, dust 0.001, TP 0.5, SL 0.3

    /// <summary>The defaults with the hard cap lifted, so the Kelly fraction itself is observable.</summary>
    private static readonly BotOptions Uncapped = new() { MaxPositionFraction = 1.0, RealisticFills = false };

    private static PortfolioState WithBalance(decimal balance) => new(
        Balance: balance,
        OpenPositionCount: 0,
        HoldsMint: false,
        MarketPrice: 0.0001m);

    /// <summary>A modest bet whose Kelly fraction sits below the cap: gain 0.1, rug 0.2, confidence 80.</summary>
    private static ModelPrediction Modest() =>
        PolicyFixtures.Buyable() with { ExpectedReturn = 0.1, RugProb = 0.2, Confidence = 80 };

    [Fact]
    public void Fraction_is_fractional_kelly_scaled_by_confidence()
    {
        // p=0.8, b=0.1, q=0.2, L=0.3 → edge = 0.08 − 0.06 = 0.02 ; f* = 0.02 / (0.1·0.3) = 2/3
        // fraction = 2/3 · 0.8 · 0.5 = 0.2667
        Assert.Equal(2.0 / 3.0 * 0.8 * 0.5, PositionSizer.Fraction(Modest(), Uncapped), 10);
    }

    [Fact]
    public void Risk_fraction_of_one_at_full_confidence_is_exactly_full_kelly()
    {
        var fullKelly = new BotOptions { MaxPositionFraction = 1.0, RiskFraction = 1.0, RealisticFills = false };
        ModelPrediction sure = Modest() with { Confidence = 100 };

        // f* = p/L − q/b = 0.8/0.3 − 0.2/0.1 = 2/3
        Assert.Equal((0.8 / 0.3) - (0.2 / 0.1), PositionSizer.Fraction(sure, fullKelly), 10);
    }

    [Fact]
    public void Gain_is_capped_at_the_take_profit()
    {
        // The bot exits at +TakeProfitPct, so an expected return beyond it can't be realized.
        double atTakeProfit = PositionSizer.Fraction(Modest() with { ExpectedReturn = Opts.TakeProfitPct }, Uncapped);
        double beyond = PositionSizer.Fraction(Modest() with { ExpectedReturn = 50.0 }, Uncapped);
        Assert.Equal(atTakeProfit, beyond);
    }

    [Fact]
    public void A_rug_costs_the_stop_loss_not_the_whole_stake()
    {
        // edge = 0.85·0.1 − 0.15·0.3 = +0.04 : positive EV once the stop caps the loss.
        ModelPrediction smallEdge = PolicyFixtures.Buyable() with { ExpectedReturn = 0.1, RugProb = 0.15 };
        Assert.True(PositionSizer.Fraction(smallEdge, Opts) > 0.0);
    }

    [Theory]
    [InlineData(0.0005)] // a sliver of upside with no rug would be full-cap Kelly
    [InlineData(0.049)]
    public void An_expected_return_below_the_take_profit_floor_sizes_to_zero(double expectedReturn)
    {
        ModelPrediction tiny = PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn, RugProb = 0.0 };

        Assert.Equal(0.0, PositionSizer.Gain(tiny, Opts));
        Assert.Equal(0.0, PositionSizer.Fraction(tiny, Opts));
    }

    [Theory]
    [InlineData(0.05, 0.05)] // at the floor: tradeable
    [InlineData(0.2, 0.2)]
    [InlineData(3.0, 0.5)]   // capped at TakeProfitPct
    [InlineData(double.NaN, 0.0)]
    public void Gain_is_the_expected_return_capped_and_floored(double expectedReturn, double gain)
    {
        Assert.Equal(gain, PositionSizer.Gain(PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn }, Opts), 12);
    }

    [Fact]
    public void A_zero_take_profit_floor_still_requires_a_positive_gain()
    {
        var noFloor = new BotOptions { MinTakeProfitPct = 0 };
        Assert.Equal(0.0, PositionSizer.Gain(PolicyFixtures.Buyable() with { ExpectedReturn = 0.0 }, noFloor));
        Assert.Equal(0.0005, PositionSizer.Gain(PolicyFixtures.Buyable() with { ExpectedReturn = 0.0005 }, noFloor), 12);
    }

    [Fact]
    public void Size_is_truncated_to_money_precision()
    {
        decimal size = PositionSizer.Size(Modest(), WithBalance(10m / 3m), Opts);

        Assert.True(size > 0m);
        Assert.Equal(Math.Round(size, Money.Decimals, MidpointRounding.ToZero), size);
        Assert.True(size <= (10m / 3m) * (decimal)PositionSizer.Fraction(Modest(), Opts));
    }

    [Theory]
    [InlineData(0.1, 0.12)] // more expected return → bigger
    public void Size_increases_with_expected_return(double low, double high)
    {
        Assert.True(PositionSizer.Fraction(Modest() with { ExpectedReturn = high }, Uncapped)
                    > PositionSizer.Fraction(Modest() with { ExpectedReturn = low }, Uncapped));
    }

    [Theory]
    [InlineData(0.2, 0.22)] // more rug risk → smaller
    public void Size_decreases_with_rug_probability(double lowRug, double highRug)
    {
        Assert.True(PositionSizer.Fraction(Modest() with { RugProb = lowRug }, Uncapped)
                    > PositionSizer.Fraction(Modest() with { RugProb = highRug }, Uncapped));
    }

    [Theory]
    [InlineData(70.0, 90.0)] // more confidence → bigger
    public void Size_increases_with_confidence(double low, double high)
    {
        Assert.True(PositionSizer.Fraction(Modest() with { Confidence = high }, Uncapped)
                    > PositionSizer.Fraction(Modest() with { Confidence = low }, Uncapped));
    }

    [Fact]
    public void Size_decreases_with_a_wider_stop_loss()
    {
        double tight = PositionSizer.Fraction(Modest(), new BotOptions { MaxPositionFraction = 1.0, StopLossPct = 0.25 });
        double wide = PositionSizer.Fraction(Modest(), new BotOptions { MaxPositionFraction = 1.0, StopLossPct = 0.3 });
        Assert.True(tight > wide);
    }

    [Fact]
    public void Size_is_balance_times_fraction()
    {
        decimal size = PositionSizer.Size(Modest(), WithBalance(10m), Uncapped);
        Assert.Equal(Money.Truncate(10m * (decimal)PositionSizer.Fraction(Modest(), Uncapped)), size);
    }

    [Fact]
    public void Size_is_capped_at_the_max_position_fraction()
    {
        // The default Buyable bet's Kelly fraction (1.12) blows past the 0.2 cap.
        Assert.Equal(Opts.MaxPositionFraction, PositionSizer.Fraction(PolicyFixtures.Buyable(), Opts));
        Assert.Equal(2m, PositionSizer.Size(PolicyFixtures.Buyable(), WithBalance(10m), Opts));
    }

    [Fact]
    public void A_computed_size_below_the_dust_floor_is_dropped_to_zero()
    {
        // Capped fraction 0.2 on a 0.004 balance = 0.0008 < the 0.001 floor.
        Assert.Equal(0m, PositionSizer.Size(PolicyFixtures.Buyable(), WithBalance(0.004m), Opts));
    }

    [Theory]
    [InlineData(0.1, 0.5)]  // edge = 0.5·0.1 − 0.5·0.3 < 0
    [InlineData(0.0, 0.2)]  // no expected gain
    [InlineData(-0.2, 0.1)] // expected loss
    public void A_non_positive_edge_sizes_to_zero(double expectedReturn, double rugProb)
    {
        ModelPrediction noEdge = PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn, RugProb = rugProb };
        Assert.Equal(0.0, PositionSizer.Fraction(noEdge, Opts));
        Assert.Equal(0m, PositionSizer.Size(noEdge, WithBalance(10m), Opts));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void A_non_positive_balance_sizes_to_zero(decimal balance)
    {
        Assert.Equal(0m, PositionSizer.Size(PolicyFixtures.Buyable(), WithBalance(balance), Opts));
    }

    [Theory]
    [MemberData(nameof(PolicyFixtures.Malformed), MemberType = typeof(PolicyFixtures))]
    public void A_malformed_prediction_sizes_to_zero_without_throwing(ModelPrediction malformed)
    {
        Assert.Equal(0.0, PositionSizer.Fraction(malformed, Opts));
        Assert.Equal(0m, PositionSizer.Size(malformed, WithBalance(10m), Opts));
    }

    private static PortfolioState InPool(decimal balance, string? venue, decimal? liquidity) =>
        WithBalance(balance) with { Pool = new Pool(venue, liquidity) };

    [Theory]
    [InlineData("pumpswap", 1000.0, 2.0)]   // 2 % of 1000 = 20 SOL: the balance cap (0.2 · 10) binds
    [InlineData("pumpswap", 50.0, 1.0)]     // 2 % of 50 SOL
    [InlineData("pumpfun", 0.5, 0.61)]      // curve: 2 % of (0.5 real + 30 virtual)
    [InlineData("raydium", 0.04, 0.0)]      // 0.0008 SOL < the 0.001 dust floor: skipped
    [InlineData("raydium", null, 0.0)]      // unknown pool: takes nothing
    public void With_realistic_fills_the_size_is_capped_by_the_pool(string venue, double? liquidity, double expected)
    {
        BotOptions realistic = PolicyFixtures.RealisticOptions();
        decimal size = PositionSizer.Size(PolicyFixtures.Buyable(), InPool(10m, venue, (decimal?)liquidity), realistic);
        Assert.Equal((decimal)expected, size);
    }

    [Fact]
    public void Idealized_fills_ignore_the_pool()
    {
        Assert.Equal(2m, PositionSizer.Size(PolicyFixtures.Buyable(), InPool(10m, "raydium", 0.01m), Opts));
    }

    [Fact]
    public void Size_never_exceeds_the_balance()
    {
        var allIn = new BotOptions { MaxPositionFraction = 1.0, RiskFraction = 1.0, RealisticFills = false };
        ModelPrediction greedy = PolicyFixtures.Buyable() with { ExpectedReturn = 100.0, Confidence = 100 };
        Assert.True(PositionSizer.Size(greedy, WithBalance(10m), allIn) <= 10m);
    }

    [Fact]
    public void Null_arguments_throw()
    {
        Assert.Throws<ArgumentNullException>(() => PositionSizer.Size(null!, WithBalance(10m), Opts));
        Assert.Throws<ArgumentNullException>(() => PositionSizer.Size(PolicyFixtures.Buyable(), WithBalance(10m), null!));
        Assert.Throws<ArgumentNullException>(() => PositionSizer.Fraction(null!, Opts));
    }
}
