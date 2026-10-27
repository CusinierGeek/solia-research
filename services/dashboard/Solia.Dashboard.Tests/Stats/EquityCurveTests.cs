using System.Globalization;
using Solia.Dashboard.Api;
using Solia.Dashboard.Formatting;
using Solia.Dashboard.Stats;

namespace Solia.Dashboard.Tests.Stats;

public class EquityCurveTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 9, 0, 0, TimeSpan.Zero);

    private static EquityCurve Curve(decimal totalPnl, params EquityPoint[] points) =>
        EquityCurve.From(new PerformanceResponse(points, totalPnl), shown: 1000);

    // Written the way the library writes a hovered point: its own culture, the formats the wrapper sets.
    private static string TooltipTime(DateTime time) => time.ToString(EquityCurve.TooltipTimeFormat, CultureInfo.CurrentCulture);

    private static string TooltipValue(double plotted) => plotted.ToString(EquityCurve.TooltipValueFormat, CultureInfo.CurrentCulture);

    private static (string Title, string Subtitle) Hover(EquityCurve curve, EquitySeriesKind series, (DateTime Time, double Value) point) =>
        curve.Tooltip(series, TooltipTime(point.Time), TooltipValue(point.Value));

    [Fact]
    public void An_empty_curve_has_no_points_no_starting_balance_and_empty_series()
    {
        EquityCurve curve = Curve(0m);

        Assert.True(curve.IsEmpty);
        Assert.True(curve.IsComplete);
        Assert.Null(curve.StartingBalance);
        Assert.Empty(curve.EquitySeries);
        Assert.Empty(curve.StartingBalanceSeries);
        Assert.Empty(curve.PnlSeries);
        Assert.Equal(TimeSpan.Zero, curve.Span);
        Assert.Equal(("0", TooltipTime(T0.UtcDateTime)), curve.Tooltip(EquitySeriesKind.Equity, TooltipTime(T0.UtcDateTime), "0"));
    }

    [Fact]
    public void Points_are_ordered_by_close_time_and_ties_keep_the_backend_order_plotted_just_apart()
    {
        EquityCurve curve = Curve(1m,
            new EquityPoint(T0.AddHours(2), 101m, 0.5m),
            new EquityPoint(T0, 100.2m, 0.2m),
            new EquityPoint(T0.AddHours(1), 100.4m, 0.2m),
            new EquityPoint(T0.AddHours(1), 100.5m, 0.1m));

        Assert.Equal([100.2m, 100.4m, 100.5m, 101m], curve.Points.Select(p => p.Equity));
        Assert.Equal(
            [T0.UtcDateTime, T0.AddHours(1).UtcDateTime, T0.AddHours(1).UtcDateTime + EquityCurve.MinPlotGap, T0.AddHours(2).UtcDateTime],
            curve.EquitySeries.Select(p => p.Time));
        Assert.All(curve.EquitySeries, p => Assert.Equal(DateTimeKind.Utc, p.Time.Kind));
    }

    [Fact]
    public void Times_are_plotted_in_utc_whatever_the_offset_received()
    {
        var local = new DateTimeOffset(2026, 10, 27, 11, 0, 0, TimeSpan.FromHours(2));

        EquityCurve curve = Curve(1m, new EquityPoint(local, 101m, 1m));

        Assert.Equal(new DateTime(2026, 10, 27, 9, 0, 0, DateTimeKind.Utc), curve.EquitySeries[0].Time);
    }

    [Fact]
    public void Decimals_stay_exact_on_the_points_and_the_series_carry_them_in_axis_units()
    {
        EquityCurve curve = Curve(1.123456789012m,
            new EquityPoint(T0, 101m, 1m),
            new EquityPoint(T0.AddHours(1), 101.123456789012m, 0.123456789012m));

        Assert.Equal(101.123456789012m, curve.Points[1].Equity);
        Assert.Equal(0.123456789012m, curve.Points[1].Pnl);
        Assert.Equal(101.123456789012 * curve.EquityAxis.Scale, curve.EquitySeries[1].Value, 6);
        Assert.Equal(0.123456789012 * curve.PnlAxis.Scale, curve.PnlSeries[1].Value, 6);
    }

    [Fact]
    public void The_starting_balance_is_the_latest_equity_minus_every_closed_trades_pnl()
    {
        // limit = 2 cut the first trades off: total_pnl still covers them.
        EquityCurve curve = Curve(5m,
            new EquityPoint(T0, 104m, -1m),
            new EquityPoint(T0.AddHours(1), 105m, 1m));

        Assert.Equal(100m, curve.StartingBalance);
        Assert.Equal(5m, curve.TotalPnl);
        Assert.All(curve.StartingBalanceSeries, p => Assert.Equal(100 * curve.EquityAxis.Scale, p.Value, 6));
        Assert.Equal(curve.EquitySeries.Select(p => p.Time), curve.StartingBalanceSeries.Select(p => p.Time));
    }

    [Fact]
    public void A_curve_as_long_as_shown_is_complete()
    {
        EquityPoint[] points = [new(T0, 101m, 1m), new(T0.AddHours(1), 102m, 1m)];

        EquityCurve curve = EquityCurve.From(new PerformanceResponse(points, 2m), shown: 2);

        Assert.True(curve.IsComplete);
        Assert.Equal(2, curve.Points.Count);
    }

    [Fact]
    public void One_point_more_than_shown_keeps_the_latest_and_marks_the_curve_incomplete()
    {
        EquityPoint[] points = [new(T0.AddHours(2), 103m, 1m), new(T0, 101m, 1m), new(T0.AddHours(1), 102m, 1m)];

        EquityCurve curve = EquityCurve.From(new PerformanceResponse(points, 3m), shown: 2);

        Assert.False(curve.IsComplete);
        Assert.Equal([102m, 103m], curve.Points.Select(p => p.Equity));
        Assert.Equal(100m, curve.StartingBalance);
    }

    [Fact]
    public void Showing_no_point_at_all_is_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EquityCurve.From(new PerformanceResponse([], 0m), shown: 0));
    }

    [Fact]
    public void The_equity_axis_spans_the_starting_balance_and_the_pnl_axis_spans_zero()
    {
        EquityCurve curve = Curve(0.3m,
            new EquityPoint(T0, 100.1m, 0.1m),
            new EquityPoint(T0.AddHours(1), 100.3m, 0.2m));

        Assert.Equal(ChartAxis.Fit([100.1, 100.3, 100]), curve.EquityAxis);
        Assert.Equal(ChartAxis.Fit([0.1, 0.2], includeZero: true), curve.PnlAxis);
    }

    [Theory]
    [InlineData(0, 1, "HH:mm")]
    [InlineData(30, 15, "HH:mm")]
    [InlineData(6 * 60, 3 * 60, "HH:mm")]
    [InlineData(24 * 60, 6 * 60, "HH:mm")]
    [InlineData(2 * 24 * 60, 12 * 60, "MM-dd HH:mm")]
    [InlineData(3 * 24 * 60, 24 * 60, "MM-dd")]
    [InlineData(10 * 24 * 60, 3 * 24 * 60, "MM-dd")]
    [InlineData(60 * 24 * 60, 30 * 24 * 60, "MM-dd")]
    [InlineData(365 * 24 * 60, 180 * 24 * 60, "MM-dd")]
    [InlineData(3 * 365 * 24 * 60, 365 * 24 * 60, "yyyy-MM")]
    [InlineData(10 * 365 * 24 * 60, 3 * 365 * 24 * 60, "yyyy-MM")]
    public void Time_labels_are_spaced_for_about_four_labels(int spanMinutes, int spacingMinutes, string format)
    {
        EquityCurve curve = Curve(2m,
            new EquityPoint(T0, 101m, 1m),
            new EquityPoint(T0.AddMinutes(spanMinutes), 102m, 1m));

        Assert.Equal(TimeSpan.FromMinutes(spanMinutes), curve.Span);
        Assert.Equal(TimeSpan.FromMinutes(spacingMinutes), curve.LabelSpacing);
        Assert.Equal(format, curve.LabelFormat);
    }

    [Fact]
    public void A_tooltip_finds_the_trade_by_its_round_tripped_time()
    {
        EquityCurve curve = Curve(1.4m,
            new EquityPoint(T0, 101.5m, 1.5m),
            new EquityPoint(T0.AddMinutes(1).AddTicks(1234), 101.4m, -0.1m));

        Assert.Equal(("Equity 101.4 · P&L -0.1", "2026-10-27 09:01:00 UTC"), Hover(curve, EquitySeriesKind.Equity, curve.EquitySeries[1]));
        Assert.Equal(("Equity 101.5 · P&L +1.5", "2026-10-27 09:00:00 UTC"), Hover(curve, EquitySeriesKind.Pnl, curve.PnlSeries[0]));
    }

    [Fact]
    public void Trades_closed_at_the_same_instant_each_get_their_own_tooltip()
    {
        EquityCurve curve = Curve(3m,
            new EquityPoint(T0, 101m, 1m),
            new EquityPoint(T0, 103m, 2m));

        Assert.Equal("Equity 101 · P&L +1", Hover(curve, EquitySeriesKind.Equity, curve.EquitySeries[0]).Title);
        Assert.Equal("Equity 103 · P&L +2", Hover(curve, EquitySeriesKind.Equity, curve.EquitySeries[1]).Title);
        Assert.Equal("Equity 103 · P&L +2", Hover(curve, EquitySeriesKind.Pnl, curve.PnlSeries[1]).Title);
        // The subtitle keeps the real close time, not the plotted one.
        Assert.Equal("2026-10-27 09:00:00 UTC", Hover(curve, EquitySeriesKind.Pnl, curve.PnlSeries[1]).Subtitle);
    }

    [Fact]
    public void A_time_written_slightly_off_still_finds_the_nearest_trade()
    {
        EquityCurve curve = Curve(2m, new EquityPoint(T0, 101m, 1m), new EquityPoint(T0.AddSeconds(1), 102m, 1m));

        string nearSecond = TooltipTime(T0.UtcDateTime.AddSeconds(1).AddTicks(-100));

        Assert.Equal("Equity 102 · P&L +1", curve.Tooltip(EquitySeriesKind.Equity, nearSecond, null).Title);
    }

    [Fact]
    public void The_starting_balance_line_has_its_own_tooltip()
    {
        EquityCurve curve = Curve(3.2m, new EquityPoint(T0, 101.5m, 1.5m), new EquityPoint(T0.AddHours(1), 103.2m, 1.7m));

        Assert.Equal(("Starting balance 100", "2026-10-27 10:00:00 UTC"),
            Hover(curve, EquitySeriesKind.StartingBalance, curve.StartingBalanceSeries[1]));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("09:00")]
    [InlineData("2026-10-27T10:00:00.0000000Z")]
    public void A_time_off_the_curve_falls_back_to_the_hovered_value_and_time(string? time)
    {
        EquityCurve curve = Curve(1.25m, new EquityPoint(T0, 101.25m, 1.25m));

        Assert.Equal(("101.25", time ?? ""), curve.Tooltip(EquitySeriesKind.Equity, time, TooltipValue(curve.EquitySeries[0].Value)));
        Assert.Equal(("1.25", time ?? ""), curve.Tooltip(EquitySeriesKind.Pnl, time, TooltipValue(curve.PnlSeries[0].Value)));
        Assert.Equal((DisplayFormat.Missing, time ?? ""), curve.Tooltip(EquitySeriesKind.Equity, time, null));
    }

    [Fact]
    public void A_refetched_curve_with_the_same_points_has_the_same_data()
    {
        EquityPoint[] points = [new(T0, 101m, 1m), new(T0.AddHours(1), 102m, 1m)];
        EquityCurve curve = Curve(2m, points);

        Assert.True(curve.HasSameData(Curve(2m, [.. points])));
        Assert.False(curve.HasSameData(null));
        Assert.False(curve.HasSameData(Curve(2.5m, points))); // another starting balance
        Assert.False(curve.HasSameData(Curve(3m, [.. points, new(T0.AddHours(2), 103m, 1m)])));
        Assert.False(curve.HasSameData(EquityCurve.From(new PerformanceResponse(points, 2m), shown: 1)));
    }
}
