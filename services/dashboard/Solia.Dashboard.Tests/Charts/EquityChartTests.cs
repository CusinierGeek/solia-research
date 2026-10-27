using MudBlazor;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Charts;
using Solia.Dashboard.Stats;

namespace Solia.Dashboard.Tests.Charts;

/// <summary>
/// The wrappers render through the real library (no stub) for every curve shape a page can hand them. What the
/// library draws is not asserted — under bUnit it doesn't plot series anyway (it waits for a browser layout pass).
/// </summary>
public class EquityChartTests : MudTestContext
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 9, 0, 0, TimeSpan.Zero);

    public static TheoryData<int> PointCounts => [0, 1, 2, 1000];

    [Theory]
    [MemberData(nameof(PointCounts))]
    public void The_equity_chart_renders_both_panels_for_any_curve(int count)
    {
        EquityPoint[] points = Enumerable.Range(0, count)
            .Select(i => new EquityPoint(T0.AddMinutes(i), 100m + (i * 0.01m), 0.01m))
            .ToArray();
        var curve = EquityCurve.From(new PerformanceResponse(points, count * 0.01m), shown: 1000);

        IRenderedComponent<EquityChart> chart = Render<EquityChart>(p => p.Add(c => c.Curve, curve));

        Assert.NotNull(chart.Find(".solia-equity-chart__equity svg"));
        Assert.NotNull(chart.Find(".solia-equity-chart__pnl svg"));
    }

    [Fact]
    public void A_new_curve_re_renders_the_chart()
    {
        var first = Curve(1m, new EquityPoint(T0, 101m, 1m));
        IRenderedComponent<EquityChart> chart = Render<EquityChart>(p => p.Add(c => c.Curve, first));

        IReadOnlyList<ChartSeries<double>> drawn = EquitySeries(chart);

        var second = Curve(-1m, new EquityPoint(T0, 101m, 1m), new EquityPoint(T0.AddDays(3), 99m, -2m));
        chart.Render(p => p.Add(c => c.Curve, second));

        Assert.NotNull(chart.Find(".solia-equity-chart__equity svg"));
        Assert.NotSame(drawn, EquitySeries(chart));
    }

    [Fact]
    public void A_new_curve_with_the_same_data_keeps_the_drawn_series()
    {
        IRenderedComponent<EquityChart> chart = Render<EquityChart>(p => p.Add(c => c.Curve, Curve(1m, new EquityPoint(T0, 101m, 1m))));
        IReadOnlyList<ChartSeries<double>> drawn = EquitySeries(chart);

        chart.Render(p => p.Add(c => c.Curve, Curve(1m, new EquityPoint(T0, 101m, 1m))));

        Assert.Same(drawn, EquitySeries(chart));
    }

    [Fact]
    public void The_starting_balance_is_the_series_its_tooltip_is_picked_by()
    {
        IRenderedComponent<EquityChart> chart = Render<EquityChart>(p => p.Add(c => c.Curve, Curve(1m, new EquityPoint(T0, 101m, 1m))));

        Assert.Equal("Starting balance", EquitySeries(chart)[EquityChart.StartingBalanceSeries].Name);
    }

    private static EquityCurve Curve(decimal totalPnl, params EquityPoint[] points) =>
        EquityCurve.From(new PerformanceResponse(points, totalPnl), shown: 1000);

    private static IReadOnlyList<ChartSeries<double>> EquitySeries(IRenderedComponent<EquityChart> chart) =>
        chart.FindComponents<MudChart<double>>()[0].Instance.ChartSeries;

    [Theory]
    [InlineData(0, 0, 0)]
    [InlineData(290, 190, 20)]
    public void The_win_rate_donut_renders_for_any_breakdown(int wins, int losses, int breakEven)
    {
        var slices = WinRateSlices.From(new WinRateResponse(0.5, wins, losses, breakEven, "all"));

        IRenderedComponent<WinRateDonut> donut = Render<WinRateDonut>(p => p.Add(c => c.Slices, slices));

        Assert.NotNull(donut.Find(".solia-winrate-donut svg"));
    }
}
