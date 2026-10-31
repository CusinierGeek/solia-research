using Solia.Bot.Trading;

namespace Solia.Bot.Backtest;

/// <summary>
/// The outcome of a <see cref="ReplayHarness"/> run: every trade the engine closed, the positions it
/// left open, the final cash/equity and the summary stats a backtest reports. All values are read off
/// the same <see cref="PositionManager"/> the replay drove, so they reconcile by construction
/// (<see cref="FinalEquity"/> = <see cref="StartingBalance"/> + <see cref="TotalPnl"/>, and
/// <see cref="TotalPnl"/> = <see cref="RealizedPnl"/> once every position is closed). Derived stats
/// (<see cref="Wins"/>, <see cref="WinRate"/>) are computed once from <see cref="Trades"/>, never passed
/// in, so they can't drift from it.
///
/// <para><b>Equality.</b> Two results are equal when every scalar matches and <see cref="Trades"/> and
/// <see cref="OpenPositions"/> hold equal elements in the same order — value equality, so two identical
/// replays compare equal with <c>==</c>.</para>
/// </summary>
/// <param name="StartingBalance">Virtual capital the replay began with.</param>
/// <param name="FinalBalance">Uncommitted cash at the end — starting balance, minus open sizes, plus closed proceeds.</param>
/// <param name="FinalEquity">Final balance plus every still-open position marked at its last price.</param>
/// <param name="RealizedPnl">Sum of realized P&amp;L over <see cref="Trades"/>, less the network fees of failed entries.</param>
/// <param name="Trades">Closed trades in the order they closed.</param>
/// <param name="OpenPositions">Positions still held when the event stream ran out, in opening order.</param>
/// <param name="DroppedEventCount">
/// Events the engine dropped for arriving more than <c>MaxEventLagSeconds</c> behind its clock. Only an
/// <see cref="ReplayOrder.Arrival"/> replay can drop anything; a <see cref="ReplayOrder.Timestamp"/> one is
/// always 0.
/// </param>
/// <param name="FailedEntryCount">Entries that failed at execution (slippage tolerance, empty pool) — no trade, a fee paid.</param>
/// <param name="FailedExitCount">Exit attempts that breached the slippage tolerance and were retried.</param>
/// <param name="AbandonedEntryCount">Pending entries abandoned for want of a mark after their latency.</param>
public sealed record ReplayResult(
    decimal StartingBalance,
    decimal FinalBalance,
    decimal FinalEquity,
    decimal RealizedPnl,
    IReadOnlyList<ClosedTrade> Trades,
    IReadOnlyList<OpenPosition> OpenPositions,
    int DroppedEventCount,
    int FailedEntryCount = 0,
    int FailedExitCount = 0,
    int AbandonedEntryCount = 0)
{
    /// <summary>How many trades closed over the replay.</summary>
    public int ClosedTradeCount => Trades.Count;

    /// <summary>Closed trades with positive realized P&amp;L. A break-even (0) is neither a win nor a loss.</summary>
    public int Wins { get; } = Trades.Count(t => t.Pnl > 0m);

    /// <summary>Closed trades with negative realized P&amp;L.</summary>
    public int Losses { get; } = Trades.Count(t => t.Pnl < 0m);

    /// <summary>Closed trades with exactly zero realized P&amp;L.</summary>
    public int BreakEven => Trades.Count - Wins - Losses;

    /// <summary>
    /// Fraction of closed trades that were wins, in [0, 1]; 0 when no trade closed (an empty backtest
    /// has no win rate to speak of, not a divide-by-zero).
    /// </summary>
    public double WinRate => Trades.Count == 0 ? 0.0 : (double)Wins / Trades.Count;

    /// <summary>
    /// Total P&amp;L — realized plus the unrealized P&amp;L of positions still open, marked at their last
    /// price: <see cref="FinalEquity"/> − <see cref="StartingBalance"/>. Equals <see cref="RealizedPnl"/>
    /// when nothing is left open.
    /// </summary>
    public decimal TotalPnl => FinalEquity - StartingBalance;

    public bool Equals(ReplayResult? other) =>
        other is not null
        && StartingBalance == other.StartingBalance
        && FinalBalance == other.FinalBalance
        && FinalEquity == other.FinalEquity
        && RealizedPnl == other.RealizedPnl
        && DroppedEventCount == other.DroppedEventCount
        && FailedEntryCount == other.FailedEntryCount
        && FailedExitCount == other.FailedExitCount
        && AbandonedEntryCount == other.AbandonedEntryCount
        && Trades.SequenceEqual(other.Trades)
        && OpenPositions.SequenceEqual(other.OpenPositions);

    public override int GetHashCode() =>
        HashCode.Combine(StartingBalance, FinalBalance, FinalEquity, RealizedPnl, DroppedEventCount, Trades.Count, OpenPositions.Count);
}
