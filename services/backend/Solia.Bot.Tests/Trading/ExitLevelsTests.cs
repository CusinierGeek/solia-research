using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Policy;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

public class ExitLevelsTests
{
    private static readonly BotOptions Opts = PolicyFixtures.Options(); // TP cap 0.5, SL 0.3, floor 0.05, ×3 σ

    private static ExitLevels Levels(ModelPrediction p, BotOptions? o = null) =>
        ExitLevels.Compute(p, 0.0001m, o ?? Opts);

    [Theory]
    [InlineData(0.2, 0.2)] // below the cap: TP = expected return
    [InlineData(0.5, 0.5)]
    [InlineData(2.0, 0.5)] // above the cap: clamped to TakeProfitPct
    public void Take_profit_is_the_expected_return_capped(double expectedReturn, double expectedTp)
    {
        ExitLevels levels = Levels(PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn });
        Assert.Equal(expectedTp, levels.TakeProfitPct, 12);
    }

    [Fact]
    public void Take_profit_equals_the_gain_the_sizer_bets_on()
    {
        ModelPrediction p = PolicyFixtures.Buyable() with { ExpectedReturn = 0.07 };
        Assert.Equal(PositionSizer.Gain(p, Opts), Levels(p).TakeProfitPct);
    }

    [Theory]
    [InlineData(-0.1)]   // expected loss
    [InlineData(0.0)]    // no expected gain
    [InlineData(0.0005)] // below the MinTakeProfitPct floor: TP would sit on the entry price
    public void A_prediction_without_a_tradeable_gain_has_no_levels(double expectedReturn)
    {
        Assert.Throws<ArgumentException>(() => Levels(PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn }));
    }

    [Theory]
    [InlineData(0.001)] // would floor at 0.05
    [InlineData(1.0)]   // would cap at 0.03
    public void A_floor_above_the_cap_does_not_throw_and_the_cap_wins(double volatility)
    {
        var unvalidated = new BotOptions { StopLossPct = 0.03 }; // MinStopLossPct stays 0.05 > 0.03

        Assert.Equal(0.03, Levels(PolicyFixtures.Buyable() with { Volatility = volatility }, unvalidated).StopLossPct, 12);
    }

    [Theory]
    [InlineData(0.04, 0.12)]  // 3 · σ
    [InlineData(0.001, 0.05)] // calm: floored at MinStopLossPct
    [InlineData(1.0, 0.3)]    // wild: capped at StopLossPct
    public void Stop_loss_scales_with_volatility_within_its_bounds(double volatility, double expectedSl)
    {
        ExitLevels levels = Levels(PolicyFixtures.Buyable() with { Volatility = volatility });
        Assert.Equal(expectedSl, levels.StopLossPct, 12);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0.0)]
    [InlineData(-0.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void Unknown_volatility_uses_the_fixed_stop_loss(double? volatility)
    {
        Assert.Equal(Opts.StopLossPct, Levels(PolicyFixtures.Buyable() with { Volatility = volatility }).StopLossPct);
    }

    [Fact]
    public void A_zero_multiple_disables_the_dynamic_stop()
    {
        var fixedStop = new BotOptions { StopLossVolatilityMultiple = 0 };
        Assert.Equal(fixedStop.StopLossPct, Levels(PolicyFixtures.Buyable() with { Volatility = 0.04 }, fixedStop).StopLossPct);
    }

    [Fact]
    public void Price_levels_are_derived_from_the_entry_price()
    {
        ExitLevels levels = Levels(PolicyFixtures.Buyable() with { ExpectedReturn = 0.5, Volatility = 0.04 });

        Assert.Equal(0.00015m, levels.TakeProfitPrice);
        Assert.Equal(0.000088m, levels.StopLossPrice);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void A_non_positive_entry_price_is_rejected(int entryPrice)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ExitLevels.Compute(PolicyFixtures.Buyable(), entryPrice, Opts));
    }
}
