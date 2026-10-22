namespace Solia.Bot.Trading;

/// <summary>
/// An <see cref="ITradeStore"/> held in memory — for unit tests and replays. Ids are sequential from 1.
/// Rejects closing an unknown or already-closed trade, as the database would leave such an update
/// matching no open row. Not thread-safe: the <see cref="PositionManager"/> consumes events one at a time.
/// </summary>
public sealed class InMemoryTradeStore : ITradeStore, IClosedTradeCounter
{
    private readonly Dictionary<long, TradeOpening> _open = [];
    private readonly List<ClosedTrade> _closed = [];
    private long _nextId = 1;

    /// <summary>Trades opened and not yet closed, by id.</summary>
    public IReadOnlyDictionary<long, TradeOpening> Open => _open;

    /// <summary>Closed trades, in close order.</summary>
    public IReadOnlyList<ClosedTrade> Closed => _closed;

    public Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(opening);
        cancellationToken.ThrowIfCancellationRequested();

        long id = _nextId++;
        _open.Add(id, opening);
        return Task.FromResult(id);
    }

    public Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(trade);
        cancellationToken.ThrowIfCancellationRequested();

        if (!_open.Remove(trade.TradeId))
        {
            throw new InvalidOperationException($"Trade {trade.TradeId} is not open.");
        }

        _closed.Add(trade);
        return Task.CompletedTask;
    }

    public Task<long> CountClosedSinceAsync(DateTimeOffset? since, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.FromResult((long)_closed.Count(t => since is null || t.ClosedAt > since));
    }
}
