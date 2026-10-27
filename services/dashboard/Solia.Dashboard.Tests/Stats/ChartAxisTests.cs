using Solia.Dashboard.Stats;

namespace Solia.Dashboard.Tests.Stats;

public class ChartAxisTests
{
    [Theory]
    [InlineData(new[] { 0.0, 100.0 }, 1, 20, 0)]
    [InlineData(new[] { 100.0, 101.0 }, 10, 2, 1)]
    [InlineData(new[] { -0.03, 0.045 }, 100, 2, 2)]
    [InlineData(new[] { 0.0, 7.0 }, 1, 2, 0)]
    [InlineData(new[] { 0.0, 4.0 }, 10, 10, 1)]
    [InlineData(new[] { 0.0, 2000.0 }, 1, 500, 0)]
    public void Ticks_follow_a_one_two_five_step_on_whole_plotted_units(double[] values, double scale, int ticks, int digits)
    {
        Assert.Equal(new ChartAxis(scale, ticks, digits), ChartAxis.Fit(values));
    }

    [Fact]
    public void A_flat_series_gets_a_step_around_its_value()
    {
        Assert.Equal(new ChartAxis(1, 2, 0), ChartAxis.Fit([100.0, 100.0]));
        Assert.Equal(new ChartAxis(1000, 1, 3), ChartAxis.Fit([0.05]));
    }

    [Fact]
    public void No_values_only_zeros_or_non_finite_values_fall_back_to_a_unit_range()
    {
        ChartAxis expected = new(10, 2, 1); // a 0–1 range on five ticks
        Assert.Equal(expected, ChartAxis.Fit([]));
        Assert.Equal(expected, ChartAxis.Fit([0.0, 0.0]));
        Assert.Equal(expected, ChartAxis.Fit([double.NaN, double.PositiveInfinity]));
    }

    [Fact]
    public void Zero_is_included_on_request()
    {
        Assert.Equal(ChartAxis.Fit([0.0, 0.5]), ChartAxis.Fit([0.4, 0.5], includeZero: true));
        Assert.Equal(ChartAxis.Fit([-0.5, 0.0]), ChartAxis.Fit([-0.5, -0.4], includeZero: true));
    }

    [Fact]
    public void Tiny_ranges_cap_the_label_decimals()
    {
        ChartAxis axis = ChartAxis.Fit([0.0, 1e-20]);

        Assert.Equal(ChartAxis.MaxDigits, axis.Digits);
        Assert.Equal("0", axis.Label(axis.Plot(1e-20)));
    }

    [Theory]
    [InlineData(100, 2, 10023, "100.23")]
    [InlineData(10, 1, 1001.0000000001, "100.1")]
    [InlineData(1, 0, -20, "-20")]
    [InlineData(100, 2, 0, "0")]
    public void Labels_write_plotted_values_back_in_series_units_invariantly(double scale, int digits, double plotted, string label)
    {
        Assert.Equal(label, new ChartAxis(scale, 1, digits).Label(plotted));
    }

    [Fact]
    public void Plotting_multiplies_by_the_scale()
    {
        Assert.Equal(12.5, new ChartAxis(100, 5, 2).Plot(0.125), 9);
    }
}
