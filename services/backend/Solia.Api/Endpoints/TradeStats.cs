namespace Solia.Api.Endpoints;

/// <summary>
/// The trade statistics math, kept pure so it is unit-testable apart from the queries feeding it. Same
/// definitions as the bot's backtest (<c>ReplayResult</c>): a win is a closed trade with positive realized
/// P&amp;L, a loss a negative one, a break-even (0, or no recorded P&amp;L) neither — but every closed trade
/// counts in the win rate's denominator.
/// </summary>
public static class TradeStats
{
    /// <summary>
    /// <c>wins / (wins + losses + breakEven)</c> — the fraction of closed trades that were wins, in [0, 1]
    /// (<c>ReplayResult.WinRate</c>); 0 when nothing closed (nothing to rate yet).
    /// </summary>
    public static double WinRate(int wins, int losses, int breakEven)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(wins);
        ArgumentOutOfRangeException.ThrowIfNegative(losses);
        ArgumentOutOfRangeException.ThrowIfNegative(breakEven);
        long closed = (long)wins + losses + breakEven;
        return closed == 0 ? 0.0 : (double)wins / closed;
    }

    /// <summary>
    /// The equity curve: one point per closed trade, in the given (<c>closed_at</c>) order, with the equity
    /// after that trade — <paramref name="startingBalance"/> plus the realized P&amp;L so far.
    /// <paramref name="pnlBefore"/> is the realized P&amp;L of the earlier trades left off the curve (when only
    /// the latest points are returned), so the equity stays absolute and the total covers every trade. Money
    /// stays <see cref="decimal"/>, so the curve is exact.
    /// </summary>
    public static PerformanceResponse EquityCurve(
        decimal startingBalance, IEnumerable<(DateTimeOffset ClosedAt, decimal Pnl)> closedTrades, decimal pnlBefore = 0m)
    {
        ArgumentNullException.ThrowIfNull(closedTrades);
        var points = new List<EquityPoint>();
        decimal total = pnlBefore;
        foreach ((DateTimeOffset closedAt, decimal pnl) in closedTrades)
        {
            total += pnl;
            points.Add(new EquityPoint(closedAt, startingBalance + total, pnl));
        }

        return new PerformanceResponse(points, total);
    }
}
