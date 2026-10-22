namespace Solia.Bot.Trading;

/// <summary>
/// Where the <see cref="PositionManager"/> records trades — one <c>trades</c> row per position, inserted
/// on open and updated on close (docs/DATA-MODEL.md). <see cref="PostgresTradeStore"/> is the durable implementation;
/// <see cref="InMemoryTradeStore"/> backs unit tests and replays. The manager only mutates its own state
/// once a store call has succeeded, so a failing store never leaves the book and the record out of step.
/// </summary>
public interface ITradeStore
{
    /// <summary>Records a newly opened trade and returns its id.</summary>
    Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default);

    /// <summary>Records the close of the trade <see cref="ClosedTrade.TradeId"/>.</summary>
    Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default);
}
