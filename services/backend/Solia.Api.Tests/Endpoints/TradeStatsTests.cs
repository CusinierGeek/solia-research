using Solia.Api.Endpoints;

namespace Solia.Api.Tests.Endpoints;

public class TradeStatsTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    [Theory]
    [InlineData(0, 0, 0, 0.0)]
    [InlineData(3, 0, 0, 1.0)]
    [InlineData(0, 2, 0, 0.0)]
    [InlineData(29, 21, 0, 0.58)]
    [InlineData(5, 3, 2, 0.5)] // break-evens stay in the denominator, as in ReplayResult.WinRate
    [InlineData(0, 0, 4, 0.0)]
    public void Win_rate_is_wins_over_every_closed_trade(int wins, int losses, int breakEven, double expected) =>
        Assert.Equal(expected, TradeStats.WinRate(wins, losses, breakEven), precision: 12);

    [Fact]
    public void Win_rate_rejects_negative_counts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TradeStats.WinRate(-1, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => TradeStats.WinRate(0, -1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => TradeStats.WinRate(0, 0, -1));
    }

    [Fact]
    public void Win_rate_does_not_overflow_on_huge_counts() =>
        Assert.Equal(1.0 / 3.0, TradeStats.WinRate(int.MaxValue, int.MaxValue, int.MaxValue), precision: 12);

    [Fact]
    public void Equity_curve_accumulates_pnl_from_the_starting_balance_exactly()
    {
        PerformanceResponse curve = TradeStats.EquityCurve(10m,
        [
            (T0, 1.5m),
            (T0.AddMinutes(1), -0.5m),
            (T0.AddMinutes(2), 0m),
            (T0.AddMinutes(3), 0.000000000001m),
        ]);

        Assert.Equal(
            [
                new EquityPoint(T0, 11.5m, 1.5m),
                new EquityPoint(T0.AddMinutes(1), 11.0m, -0.5m),
                new EquityPoint(T0.AddMinutes(2), 11.0m, 0m),
                new EquityPoint(T0.AddMinutes(3), 11.000000000001m, 0.000000000001m),
            ],
            curve.Points);
        Assert.Equal(1.000000000001m, curve.TotalPnl);
    }

    [Fact]
    public void Equity_curve_of_the_latest_trades_carries_the_earlier_pnl()
    {
        PerformanceResponse curve = TradeStats.EquityCurve(10m, [(T0, 1m), (T0.AddMinutes(1), -0.25m)], pnlBefore: 2.5m);

        Assert.Equal(
            [new EquityPoint(T0, 13.5m, 1m), new EquityPoint(T0.AddMinutes(1), 13.25m, -0.25m)],
            curve.Points);
        Assert.Equal(3.25m, curve.TotalPnl);
    }

    [Fact]
    public void Equity_curve_of_no_trades_is_empty_with_zero_pnl()
    {
        PerformanceResponse curve = TradeStats.EquityCurve(10m, []);

        Assert.Empty(curve.Points);
        Assert.Equal(0m, curve.TotalPnl);
    }
}
