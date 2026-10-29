using Solia.DataPipeline.Features;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

public sealed class NumericBoundsTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 28, 0, 0, 0, TimeSpan.Zero);

    [Theory]
    [InlineData("0.000000001", true)]
    [InlineData("9999.999999", true)]
    [InlineData("-9999.999999", true)]
    [InlineData("10000", false)]
    [InlineData("-10000", false)]
    [InlineData("123456.78", false)]
    public void Price_fits_numeric_28_24(string value, bool fits) =>
        Assert.Equal(fits, NumericBounds.FitsPrice(decimal.Parse(value, System.Globalization.CultureInfo.InvariantCulture)));

    [Theory]
    [InlineData("9999999999999999.999999999999", true)] // max of NUMERIC(28,12)
    [InlineData("-9999999999999999", true)]
    [InlineData("10000000000000000", false)]
    [InlineData("-10000000000000000", false)]
    public void Amount_fits_numeric_28_12(string value, bool fits) =>
        Assert.Equal(fits, NumericBounds.FitsAmount(decimal.Parse(value, System.Globalization.CultureInfo.InvariantCulture)));

    [Fact]
    public void A_snapshot_that_fits_is_returned_unchanged()
    {
        FeatureSnapshot s = Snap() with { Price = 0.00001m, MarketCap = 30m, NetSolFlow = -2m };

        FeatureSnapshot bounded = NumericBounds.Bound(s, out IReadOnlyList<string> nulled);

        Assert.Same(s, bounded);
        Assert.Empty(nulled);
    }

    [Fact]
    public void Out_of_range_values_are_nulled_and_the_rest_is_kept()
    {
        FeatureSnapshot s = Snap() with
        {
            Price = 25_000m,
            Volume = 1e17m,
            MarketCap = 30m,
            Liquidity = -1e16m,
            NetSolFlow = 1e16m,
            AvgBuySol = 0.5m,
            MaxBuySol = 1e20m,
            Holders = 7,
            Rsi = 55,
        };

        FeatureSnapshot bounded = NumericBounds.Bound(s, out IReadOnlyList<string> nulled);

        Assert.Equal(["price", "volume", "liquidity", "net_sol_flow", "max_buy_sol"], nulled);
        Assert.Null(bounded.Price);
        Assert.Null(bounded.Volume);
        Assert.Null(bounded.Liquidity);
        Assert.Null(bounded.NetSolFlow);
        Assert.Null(bounded.MaxBuySol);
        Assert.Equal(30m, bounded.MarketCap);
        Assert.Equal(0.5m, bounded.AvgBuySol);
        Assert.Equal(7, bounded.Holders);
        Assert.Equal(55, bounded.Rsi);
    }

    [Fact]
    public void Null_values_stay_null_and_are_not_reported()
    {
        FeatureSnapshot s = Snap() with { Price = null, Volume = null };

        NumericBounds.Bound(s, out IReadOnlyList<string> nulled);

        Assert.Empty(nulled);
    }

    private static FeatureSnapshot Snap() => new() { Mint = "M1", Ts = T0, FirstSeenUtc = T0 };
}
