using Microsoft.Extensions.Logging;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Logging;
using Solia.Bot.Policy;
using Solia.Shared.Logging;

namespace Solia.Bot.Trading;

/// <summary>
/// The paper-trading engine: consumes <see cref="BotEvent"/>s in timestamp order, opens positions on
/// accepted entries, marks them against the price feed, closes them on exit rules and accounts P&amp;L.
/// <strong>Paper only</strong> — a fill is the observed market price (no slippage), nothing is ever sent
/// on-chain. Deterministic: the only clock is the event timestamp, never wall time.
///
/// <para><b>Ordering.</b> The event clock never moves back. An event up to
/// <see cref="BotOptions.MaxEventLagSeconds"/> behind it is applied at the clock's time; an older one is
/// dropped and counted in <see cref="DroppedEventCount"/>. A tick older than its mint's last mark never
/// overwrites it, and a tick from before a position was opened never exits it.</para>
///
/// <para><b>Fresh marks.</b> A mark is the mint's market only while it is at most
/// <see cref="BotOptions.MaxMarkAgeSeconds"/> old. Entry needs a fresh mark (else the gate declines:
/// no market). A signal or timeout exit on a stale mark assumes the unobserved market hit the stop and
/// fills at <c>min(last mark, stop-loss price)</c> — a silent feed must not book a break-even.</para>
///
/// <para><b>Entry.</b> A prediction for a mint with no open position goes through
/// <see cref="EntryRule"/>; on accept it fills at the mint's fresh mark, locks the size from
/// <see cref="Balance"/> and fixes dynamic TP/SL (<see cref="ExitLevels"/>).</para>
///
/// <para><b>Exits, by precedence</b> (one reason per close, the first that applies):
/// <list type="number">
///   <item><c>timeout</c> — swept before every event (and by <see cref="AdvanceToAsync"/>): a position
///   whose deadline (<c>opened_at + PositionTimeoutSeconds</c>) is at or before the event's time closes
///   <em>at its deadline</em>, at the last price marked before it. An event at or after the deadline
///   therefore never acts on an expired position.</item>
///   <item><c>sl</c> — a tick at or below the stop-loss price;</item>
///   <item><c>tp</c> — a tick at or above the take-profit price (a single tick can't hit both: TP is
///   above entry, SL below);</item>
///   <item><c>signal</c> — a fresh prediction for the held mint classified <see cref="TradeSignal.Sell"/>,
///   filled at the last mark.</item>
/// </list>
/// Price exits fill at the tick that triggered them, not at the level: a tick that gaps through the
/// stop realizes the gap.</para>
///
/// <para><b>Accounting.</b> Close returns <c>Size · exit / entry</c> to the balance;
/// <c>pnl = proceeds − size</c>, <c>pnl_pct = pnl / size</c>. <see cref="Equity"/> marks open positions
/// at their last price. State changes only after the <see cref="ITradeStore"/> call succeeds.</para>
///
/// <para><b>Failures.</b> If <see cref="HandleAsync"/> throws (a store error), every close or open
/// already recorded stays applied and the clock stands at the last applied one; re-delivering the same
/// event resumes where it stopped and never applies anything twice. Not thread-safe — one event at a
/// time.</para>
/// </summary>
public sealed class PositionManager
{
    private readonly BotOptions _options;
    private readonly ITradeStore _store;
    private readonly BestEffortLogWriter _log;
    private readonly Dictionary<string, OpenPosition> _positions = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Mark> _marks = new(StringComparer.Ordinal);

    // Earliest deadline first; entries of positions closed by another rule are skipped lazily.
    private readonly PriorityQueue<OpenPosition, (DateTimeOffset Deadline, long TradeId)> _deadlines = new();

    /// <param name="logSink">
    /// Operational sink for position opens/closes (service = 'bot'). Defaults to the no-op
    /// <see cref="NullBotLogSink"/> so the engine stays testable without a database; writes go through
    /// a <see cref="BestEffortLogWriter"/> — bounded, never thrown, so they never alter trading.
    /// </param>
    /// <param name="logger">Where a failed operational write is reported; defaults to none.</param>
    public PositionManager(BotOptions options, ITradeStore store, IBotLogSink? logSink = null, ILogger<PositionManager>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(store);

        _options = options;
        _store = store;
        _log = new BestEffortLogWriter(logSink ?? NullBotLogSink.Instance, logger);
        Balance = options.StartingBalance;
    }

    /// <summary>Uncommitted cash — starting balance, minus open sizes, plus closed proceeds.</summary>
    public decimal Balance { get; private set; }

    /// <summary>Balance plus every open position marked at its mint's last price.</summary>
    public decimal Equity => Balance + _positions.Values.Sum(p => p.Entry.ValueAt(_marks[p.Mint].Price));

    /// <summary>Sum of realized P&amp;L over closed trades.</summary>
    public decimal RealizedPnl { get; private set; }

    /// <summary>Closed trades since start — feeds the retrain counter.</summary>
    public int ClosedTradeCount { get; private set; }

    /// <summary>Events dropped for arriving more than <see cref="BotOptions.MaxEventLagSeconds"/> late.</summary>
    public int DroppedEventCount { get; private set; }

    /// <summary>The event clock: time of the latest applied event, sweep or timeout; null before the first.</summary>
    public DateTimeOffset? Clock { get; private set; }

    /// <summary>Positions currently held, keyed by mint.</summary>
    public IReadOnlyDictionary<string, OpenPosition> OpenPositions => _positions;

    /// <summary>Last observed price for <paramref name="mint"/>, or null if none yet.</summary>
    public decimal? MarkOf(string mint) => _marks.TryGetValue(mint, out Mark mark) ? mark.Price : null;

    /// <summary>
    /// Applies one event: sweeps timeouts up to its time, then marks/exits (price) or enters/exits
    /// (prediction). A late event is applied at the clock's time, or dropped past the allowed lag.
    /// </summary>
    public async Task HandleAsync(BotEvent botEvent, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(botEvent);

        DateTimeOffset now = botEvent.Ts;
        if (Clock is DateTimeOffset clock && now < clock)
        {
            if (clock - now > TimeSpan.FromSeconds(_options.MaxEventLagSeconds))
            {
                DroppedEventCount++;
                return;
            }

            now = clock;
        }

        await AdvanceToAsync(now, cancellationToken);

        switch (botEvent)
        {
            case BotEvent.Price price:
                await OnPriceAsync(price.Value, now, cancellationToken);
                break;
            case BotEvent.Prediction prediction:
                await OnPredictionAsync(prediction.Value, now, cancellationToken);
                break;
        }
    }

    /// <summary>
    /// Moves the event clock to <paramref name="now"/>, closing every position whose deadline has been
    /// reached (earliest deadline first). Lets a host sweep timeouts when no event arrives. A time before
    /// the clock is a no-op — the clock never moves back.
    /// </summary>
    public async Task AdvanceToAsync(DateTimeOffset now, CancellationToken cancellationToken = default)
    {
        if (now < Clock)
        {
            return;
        }

        while (_deadlines.TryPeek(out OpenPosition? position, out var key) && key.Deadline <= now)
        {
            if (_positions.TryGetValue(position.Mint, out OpenPosition? held) && held.TradeId == position.TradeId)
            {
                await CloseAsync(position, position.Deadline, StaleSafeExitPrice(position, position.Deadline),
                    ExitReason.Timeout, cancellationToken);
            }

            _deadlines.Dequeue();
        }

        Clock = now;
    }

    private async Task OnPriceAsync(PriceTick tick, DateTimeOffset now, CancellationToken cancellationToken)
    {
        // A non-positive price is a broken feed, not a market — don't mark (or stop out) on it.
        if (tick.Price <= 0m)
        {
            return;
        }

        // Never let an older tick overwrite a newer mark (equal times: the later arrival wins).
        if (_marks.TryGetValue(tick.Mint, out Mark last) && tick.Ts < last.Ts)
        {
            return;
        }

        _marks[tick.Mint] = new Mark(tick.Price, tick.Ts);

        // A price from before the fill says nothing about the position's exit.
        if (!_positions.TryGetValue(tick.Mint, out OpenPosition? position) || tick.Ts < position.Entry.OpenedAt)
        {
            return;
        }

        ExitLevels levels = position.Entry.Levels;
        if (tick.Price <= levels.StopLossPrice)
        {
            await CloseAsync(position, now, tick.Price, ExitReason.StopLoss, cancellationToken);
        }
        else if (tick.Price >= levels.TakeProfitPrice)
        {
            await CloseAsync(position, now, tick.Price, ExitReason.TakeProfit, cancellationToken);
        }
    }

    private async Task OnPredictionAsync(ModelPrediction prediction, DateTimeOffset now, CancellationToken cancellationToken)
    {
        if (_positions.TryGetValue(prediction.Mint, out OpenPosition? held))
        {
            if (SignalClassifier.Classify(prediction, _options) == TradeSignal.Sell)
            {
                await CloseAsync(held, now, StaleSafeExitPrice(held, now), ExitReason.Signal, cancellationToken);
            }

            return;
        }

        decimal? market = _marks.TryGetValue(prediction.Mint, out Mark mark) && IsFresh(mark, now) ? mark.Price : null;
        var portfolio = new PortfolioState(Balance, _positions.Count, HoldsMint: false, market);
        EntryDecision decision = EntryRule.Evaluate(prediction, portfolio, _options);
        if (!decision.Enter)
        {
            return;
        }

        // The gate only accepts with a positive market price.
        decimal entryPrice = market!.Value;
        var opening = new TradeOpening(
            prediction.Mint,
            now,
            entryPrice,
            decision.Size,
            ExitLevels.Compute(prediction, entryPrice, _options),
            prediction);

        long tradeId = await _store.OpenAsync(opening, cancellationToken);

        var position = new OpenPosition(tradeId, opening, opening.OpenedAt.AddSeconds(_options.PositionTimeoutSeconds));
        Balance -= opening.Size;
        _positions.Add(opening.Mint, position);
        _deadlines.Enqueue(position, (position.Deadline, position.TradeId));

        await _log.WriteAsync(LogSeverity.Info, "position opened",
            new { tradeId, mint = opening.Mint, entryPrice = opening.EntryPrice, size = opening.Size }, cancellationToken);
    }

    private bool IsFresh(Mark mark, DateTimeOffset at) =>
        at - mark.Ts <= TimeSpan.FromSeconds(_options.MaxMarkAgeSeconds);

    /// <summary>The last mark if fresh at <paramref name="at"/>; else the worse of it and the stop.</summary>
    private decimal StaleSafeExitPrice(OpenPosition position, DateTimeOffset at)
    {
        Mark mark = _marks[position.Mint];
        return IsFresh(mark, at) ? mark.Price : Math.Min(mark.Price, position.Entry.Levels.StopLossPrice);
    }

    private async Task CloseAsync(
        OpenPosition position, DateTimeOffset closedAt, decimal exitPrice, ExitReason reason, CancellationToken cancellationToken)
    {
        ClosedTrade trade = ClosedTrade.Settle(position, closedAt, exitPrice, reason);

        await _store.CloseAsync(trade, cancellationToken);

        _positions.Remove(position.Mint);
        Balance += trade.Proceeds;
        RealizedPnl += trade.Pnl;
        ClosedTradeCount++;
        if (Clock is null || closedAt > Clock)
        {
            Clock = closedAt;
        }

        await _log.WriteAsync(LogSeverity.Info, "position closed",
            new
            {
                tradeId = trade.TradeId,
                mint = position.Mint,
                reason = reason.ToCode(),
                exitPrice = trade.ExitPrice,
                pnl = trade.Pnl,
                pnlPct = trade.PnlPct,
            },
            cancellationToken);
    }

    private readonly record struct Mark(decimal Price, DateTimeOffset Ts);
}
