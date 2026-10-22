using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Tests.Features;

public class IndicatorsTests
{
    [Fact]
    public void Ema_moves_toward_the_new_value()
    {
        Assert.Equal(0.2, Indicators.AlphaFromSpan(9), 10);
        Assert.Equal(12, Indicators.Ema(10, 20, 0.2), 10); // 10 + 0.2*(20-10)
    }

    [Theory]
    [InlineData(new double[] { 5 })]
    [InlineData(new double[] { 5, 5, 5 })]
    public void Volatility_is_zero_for_flat_or_single_price(double[] prices)
    {
        Assert.Equal(0, Indicators.Volatility(prices));
    }

    [Fact]
    public void Volatility_is_positive_for_varying_prices()
    {
        Assert.True(Indicators.Volatility(new double[] { 1, 2, 1, 2 }) > 0);
    }

    [Fact]
    public void Rsi_is_null_without_enough_history()
    {
        Assert.Null(Indicators.Rsi(new double[] { 1, 2, 3 }, period: 14));
    }

    [Fact]
    public void Rsi_is_100_when_only_rising()
    {
        Assert.Equal(100, Indicators.Rsi(new double[] { 1, 2, 3, 4, 5 }, period: 4));
    }

    [Fact]
    public void Rsi_is_zero_when_only_falling()
    {
        Assert.Equal(0, Indicators.Rsi(new double[] { 5, 4, 3, 2, 1 }, period: 4));
    }

    [Fact]
    public void Growth_rate_is_per_second_and_guards_zero_interval()
    {
        Assert.Equal(2, Indicators.GrowthRate(10, 20, 5), 10);
        Assert.Equal(0, Indicators.GrowthRate(10, 20, 0));
    }
}
