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
/// <strong>Paper only</strong> — fills are simulated, nothing is ever sent on-chain. Deterministic: the only clock
/// is the event timestamp, never wall time.
///
/// <para><b>Fills.</b> With <see cref="BotOptions.RealisticFills"/> (the default) an order trades against the pool
/// of the mark it fills at (<see cref="Fills"/>): entry needs that pool to clear the liquidity gate, the size is
/// capped by it, and both sides pay constant-product price impact and the venue fee — the entry price recorded is
/// the all-in price per token, the exit proceeds never exceed what the pool holds (an empty one pays nothing). Exit
/// triggers (TP/SL, dead-feed stop) read the observed mark; the fill applies on top. Without it, every fill is
/// the mark itself (the idealized simulator).</para>
///
/// <para><b>Ordering.</b> The event clock never moves back. An event up to
/// <see cref="BotOptions.MaxEventLagSeconds"/> behind it is applied at the clock's time; an older one is
/// dropped and counted in <see cref="DroppedEventCount"/>. A tick older than its mint's last mark never
/// overwrites it, and a tick from before a position was opened never exits it.</para>
///
/// <para><b>Fresh marks.</b> A mark is the mint's market only while it is at most
/// <see cref="BotOptions.MaxMarkAgeSeconds"/> old. Entry needs a fresh mark (else the gate declines:
/// no market). A signal or timeout exit on a stale mark fills at that last mark while the feed is live — some
/// mint was marked at most <see cref="BotOptions.MaxFeedSilenceSeconds"/> before the exit, by event time: the
/// mint just stopped trading, and a bonding curve's price only moves on trades. Once the feed itself is silent
/// it assumes the unobserved market hit the stop and fills at <c>min(last mark, stop-loss price)</c> — a dead
/// feed must not book a break-even.</para>
///
/// <para><b>Entry.</b> A prediction for a mint with no open position or pending entry goes through
/// <see cref="EntryRule"/> (gates, sizing and the pool cap read at decision time, against the balance net of a round
/// trip's fixed costs); on accept it fills at the mint's fresh mark (see <b>Fills</b>), locks the size from
/// <see cref="Balance"/> and fixes dynamic TP/SL from the mark (<see cref="ExitLevels"/>).</para>
///
/// <para><b>Execution latency</b> (<see cref="BotOptions.EntryLatencyMs"/> / <see cref="BotOptions.ExitLatencyMs"/>, on
/// the event clock). With a latency an accepted entry becomes <em>pending</em>: it holds its size and a slot (it counts
/// against <see cref="BotOptions.MaxConcurrentPositions"/>, and the mint takes no other entry) and fills at the mint's
/// first mark at or after <c>decision + latency</c>, against that mark's pool, with TP/SL from that fill mark. With no
/// such mark by <see cref="BotOptions.MaxMarkAgeSeconds"/> after it, the entry is abandoned (no trade, the size
/// released). An exit trigger likewise makes a <em>pending exit</em>: the position stays open, further triggers are
/// ignored, and it fills at the first mark at or after <c>trigger + latency</c> — or, with none by
/// <see cref="BotOptions.MaxMarkAgeSeconds"/> after it, at the stale-mark rule above. Zero latency fills at once, at the
/// decision mark.</para>
///
/// <para><b>Failed fills</b> (<see cref="BotOptions.MaxSlippagePct"/>). Each order is quoted at decision time (the fill
/// it would get at the decision mark). An entry filling more than the tolerance above its quote — or into a pool with
/// no SOL reserve — fails: no position, the size released, the network fee charged
/// (<see cref="FailedEntryCount"/>). An exit filling more than the tolerance below its quote fails, pays the fee and is
/// re-sent with its fill as the new quote, due one latency later (<see cref="FailedExitCount"/>); after
/// <see cref="BotOptions.MaxExitRetries"/> failures it fills at any price, so an exit is never dropped.</para>
///
/// <para><b>Network costs.</b> Every transaction pays <see cref="BotOptions.NetworkFeeSol"/>, failed or not; an entry
/// also pays <see cref="BotOptions.AccountRentSol"/> (<see cref="TradeOpening.EntryCost"/>,
/// <see cref="ClosedTrade.ExitCost"/>, both in <c>pnl</c>). A failed entry's fee is booked in
/// <see cref="RealizedPnl"/> directly.</para>
///
/// <para><b>Exits, by precedence</b> (one reason per close, the first that applies; with an exit latency, the
/// reason of the trigger the pending exit came from, and the close time is the fill's):
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
/// <para><b>Accounting</b> (SOL). Open takes <c>size + entry cost</c> from the balance; close returns
/// <c>Size · exit / entry − exit cost</c> (fill prices, so venue fees and impact are in it);
/// <c>pnl = Size · exit / entry − size − entry cost − exit cost</c>, <c>pnl_pct = pnl / size</c>.
/// <see cref="Equity"/> marks open positions at their last price (no exit impact) and counts pending entries at
/// their size. State changes only after the <see cref="ITradeStore"/> call succeeds.</para>
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
    private readonly ILogger? _logger;
    private readonly Dictionary<string, OpenPosition> _positions = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Mark> _marks = new(StringComparer.Ordinal);
    private readonly Dictionary<string, PendingEntry> _pendingEntries = new(StringComparer.Ordinal);
    private readonly Dictionary<string, PendingExit> _pendingExits = new(StringComparer.Ordinal);

    // Time of the newest mark on any mint: the price feed's last sign of life.
    private DateTimeOffset? _lastFeedTs;

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
        _logger = logger;
        Balance = options.StartingBalance;
    }

    /// <summary>Uncommitted cash — starting balance, minus open and pending sizes and the costs paid, plus closed proceeds.</summary>
    public decimal Balance { get; private set; }

    /// <summary>Balance plus every pending entry at its size and every open position marked at its mint's last price.</summary>
    public decimal Equity =>
        Balance + _pendingEntries.Values.Sum(e => e.Size) + _positions.Values.Sum(p => p.Entry.ValueAt(_marks[p.Mint].Price));

    /// <summary>Sum of realized P&amp;L over closed trades, less the network fees of failed entries.</summary>
    public decimal RealizedPnl { get; private set; }

    /// <summary>Closed trades since start — feeds the retrain counter.</summary>
    public int ClosedTradeCount { get; private set; }

    /// <summary>Events dropped for arriving more than <see cref="BotOptions.MaxEventLagSeconds"/> late.</summary>
    public int DroppedEventCount { get; private set; }

    /// <summary>Entries that failed at execution: slippage tolerance breached, or a pool with no SOL reserve.</summary>
    public int FailedEntryCount { get; private set; }

    /// <summary>Exit attempts that breached the slippage tolerance (each was retried).</summary>
    public int FailedExitCount { get; private set; }

    /// <summary>Pending entries abandoned for want of a mark after their latency.</summary>
    public int AbandonedEntryCount { get; private set; }

    /// <summary>The event clock: time of the latest applied event, sweep or timeout; null before the first.</summary>
    public DateTimeOffset? Clock { get; private set; }

    /// <summary>Positions currently held, keyed by mint — those with an exit in flight included.</summary>
    public IReadOnlyDictionary<string, OpenPosition> OpenPositions => _positions;

    /// <summary>Entries decided but not filled yet (see <see cref="BotOptions.EntryLatencyMs"/>).</summary>
    public int PendingEntryCount => _pendingEntries.Count;

    /// <summary>Whether <paramref name="mint"/>'s position has an exit in flight.</summary>
    public bool HasPendingExit(string mint) => _pendingExits.ContainsKey(mint);

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
    /// Moves the event clock to <paramref name="now"/>, in time order: every position whose deadline has been
    /// reached times out (closes, or sends its exit with an exit latency), and every pending order with no mark
    /// within <see cref="BotOptions.MaxMarkAgeSeconds"/> after its due time expires (an entry is abandoned, an exit
    /// fills at the stale-mark rule). Lets a host sweep when no event arrives. A time before the clock is a no-op —
    /// the clock never moves back.
    /// </summary>
    public async Task AdvanceToAsync(DateTimeOffset now, CancellationToken cancellationToken = default)
    {
        if (now < Clock)
        {
            return;
        }

        while (true)
        {
            while (_deadlines.TryPeek(out OpenPosition? queued, out _) && !IsHeld(queued))
            {
                _deadlines.Dequeue();
            }

            OpenPosition? expired = _deadlines.TryPeek(out OpenPosition? first, out var key) && key.Deadline <= now ? first : null;
            Expiry? expiry = NextExpiry() is { } next && next.At < now ? next : null;

            if (expired is not null && (expiry is null || expired.Deadline <= expiry.At))
            {
                if (!_pendingExits.ContainsKey(expired.Mint))
                {
                    await TriggerExitAsync(expired, expired.Deadline, StaleSafeExitPrice(expired, expired.Deadline),
                        ExitReason.Timeout, cancellationToken);
                }

                _deadlines.Dequeue();
            }
            else if (expiry is { IsExit: true })
            {
                OpenPosition held = _positions[expiry.Mint];
                PendingExit exit = _pendingExits[expiry.Mint];
                await CloseAsync(held, expiry.At, SellFill(held, StaleSafeExitPrice(held, expiry.At), _marks[held.Mint].Pool),
                    exit.Reason, ExitCostOf(exit), cancellationToken);
            }
            else if (expiry is not null)
            {
                await AbandonEntryAsync(expiry.Mint, expiry.At, cancellationToken);
            }
            else
            {
                break;
            }
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

        var mark = new Mark(tick.Price, tick.Ts, tick.Pool);
        _marks[tick.Mint] = mark;
        if (_lastFeedTs is null || tick.Ts > _lastFeedTs)
        {
            _lastFeedTs = tick.Ts;
        }

        // A pending entry fills at the first mark at or after its due time; the mint holds no position meanwhile.
        if (_pendingEntries.TryGetValue(tick.Mint, out PendingEntry? entry))
        {
            if (tick.Ts >= entry.Due)
            {
                await FillEntryAsync(tick.Mint, entry, mark, now, reserved: true, cancellationToken);
            }

            return;
        }

        // A price from before the fill says nothing about the position's exit.
        if (!_positions.TryGetValue(tick.Mint, out OpenPosition? position) || tick.Ts < position.Entry.OpenedAt)
        {
            return;
        }

        // An exit in flight: no second trigger, it fills at the first mark at or after its due time.
        if (_pendingExits.TryGetValue(tick.Mint, out PendingExit? exit))
        {
            if (tick.Ts >= exit.Due)
            {
                await FillPendingExitAsync(position, exit, mark, now, cancellationToken);
            }

            return;
        }

        ExitLevels levels = position.Entry.Levels;
        if (tick.Price <= levels.StopLossPrice)
        {
            await TriggerExitAsync(position, now, tick.Price, ExitReason.StopLoss, cancellationToken);
        }
        else if (tick.Price >= levels.TakeProfitPrice)
        {
            await TriggerExitAsync(position, now, tick.Price, ExitReason.TakeProfit, cancellationToken);
        }
    }

    private async Task OnPredictionAsync(ModelPrediction prediction, DateTimeOffset now, CancellationToken cancellationToken)
    {
        if (_positions.TryGetValue(prediction.Mint, out OpenPosition? held))
        {
            if (!_pendingExits.ContainsKey(prediction.Mint) && SignalClassifier.Classify(prediction, _options) == TradeSignal.Sell)
            {
                await TriggerExitAsync(held, now, StaleSafeExitPrice(held, now), ExitReason.Signal, cancellationToken);
            }

            return;
        }

        // An entry in flight is already sent: a newer prediction can't change it.
        if (_pendingEntries.ContainsKey(prediction.Mint))
        {
            return;
        }

        Mark? fresh = _marks.TryGetValue(prediction.Mint, out Mark mark) && IsFresh(mark, now) ? mark : null;
        // Sized against what is left once a round trip's fixed costs are set aside, so they can always be paid.
        decimal available = Math.Max(0m, Balance - _options.RoundTripFixedCostSol());
        var portfolio = new PortfolioState(
            available, _positions.Count + _pendingEntries.Count, HoldsMint: false, fresh?.Price, fresh?.Pool ?? default);
        EntryDecision decision = EntryRule.Evaluate(prediction, portfolio, _options);
        if (!decision.Enter)
        {
            LogDeclinedEntry(prediction, decision.Reason);
            return;
        }

        // The gate only accepts with a positive market price (and, with realistic fills, a tradeable pool).
        Mark market = fresh!.Value;
        var pending = new PendingEntry(
            prediction, now.AddMilliseconds(_options.EntryLatencyMs), decision.Size, BuyFill(decision.Size, market));

        if (_options.EntryLatencyMs == 0)
        {
            await FillEntryAsync(prediction.Mint, pending, market, now, reserved: false, cancellationToken);
            return;
        }

        Balance -= pending.Size;
        _pendingEntries.Add(prediction.Mint, pending);
    }

    /// <summary>
    /// Logs why a prediction that cleared the signal gate was not entered (#219). The many non-buy predictions are not
    /// logged: only the signals the bot lost after the model said buy.
    /// </summary>
    private void LogDeclinedEntry(ModelPrediction prediction, EntryRejectReason reason)
    {
        if (reason is EntryRejectReason.NotABuySignal or EntryRejectReason.None || _logger is null)
        {
            return;
        }

        _logger.LogInformation(
            "entry declined: {Reason} for {Mint} (pump {PumpProb:0.###}, rug {RugProb:0.###}, confidence {Confidence:0.#}, expected return {ExpectedReturn:0.###})",
            reason, prediction.Mint, prediction.PumpProb, prediction.RugProb, prediction.Confidence, prediction.ExpectedReturn);
    }

    /// <summary>
    /// Executes <paramref name="pending"/> at <paramref name="market"/>: opens the position, or fails it (slippage past
    /// the tolerance, or a pool with no SOL left) and charges the network fee. <paramref name="reserved"/>: its size is
    /// already held from the balance (a pending entry).
    /// </summary>
    private async Task FillEntryAsync(
        string mint, PendingEntry pending, Mark market, DateTimeOffset now, bool reserved, CancellationToken cancellationToken)
    {
        string? failure = null;
        decimal entryPrice = 0m;
        if (_options.RealisticFills && Fills.ReserveSol(market.Pool, _options) is not > 0m)
        {
            failure = "empty pool";
        }
        else
        {
            entryPrice = BuyFill(pending.Size, market);
            if (_options.MaxSlippagePct > 0 && entryPrice > pending.Quote * (1m + (decimal)_options.MaxSlippagePct))
            {
                failure = "slippage";
            }
        }

        if (failure is not null)
        {
            await FailEntryAsync(mint, pending, market, now, entryPrice, failure, reserved, cancellationToken);
            return;
        }

        var opening = new TradeOpening(
            mint,
            now,
            entryPrice,
            pending.Size,
            ExitLevels.Compute(pending.Prediction, market.Price, _options),
            pending.Prediction,
            _options.NetworkFeeSol + _options.AccountRentSol);

        long tradeId = await _store.OpenAsync(opening, cancellationToken);

        var position = new OpenPosition(tradeId, opening, opening.OpenedAt.AddSeconds(_options.PositionTimeoutSeconds));
        if (!reserved)
        {
            Balance -= opening.Size;
        }

        _pendingEntries.Remove(mint);
        Balance -= opening.EntryCost;
        _positions.Add(opening.Mint, position);
        _deadlines.Enqueue(position, (position.Deadline, position.TradeId));
        MoveClock(now);

        await _log.WriteAsync(LogSeverity.Info, "position opened",
            new
            {
                tradeId,
                mint = opening.Mint,
                entryPrice = opening.EntryPrice,
                markPrice = market.Price,
                size = opening.Size,
                entryCost = opening.EntryCost,
                venue = market.Pool.Venue,
                liquidity = market.Pool.LiquiditySol,
            },
            cancellationToken);
    }

    private async Task FailEntryAsync(
        string mint, PendingEntry pending, Mark market, DateTimeOffset now, decimal fillPrice, string failure, bool reserved,
        CancellationToken cancellationToken)
    {
        if (reserved)
        {
            _pendingEntries.Remove(mint);
            Balance += pending.Size;
        }

        // The transaction landed and failed: its fee is spent all the same.
        Balance -= _options.NetworkFeeSol;
        RealizedPnl -= _options.NetworkFeeSol;
        FailedEntryCount++;
        MoveClock(now);

        await _log.WriteAsync(LogSeverity.Warn, "entry failed",
            new
            {
                mint,
                reason = failure,
                quote = pending.Quote,
                fillPrice,
                markPrice = market.Price,
                size = pending.Size,
                networkFee = _options.NetworkFeeSol,
            },
            cancellationToken);
    }

    private async Task AbandonEntryAsync(string mint, DateTimeOffset at, CancellationToken cancellationToken)
    {
        PendingEntry pending = _pendingEntries[mint];
        _pendingEntries.Remove(mint);
        Balance += pending.Size;
        AbandonedEntryCount++;
        MoveClock(at);

        await _log.WriteAsync(LogSeverity.Info, "entry abandoned",
            new { mint, reason = "no mark", due = pending.Due, size = pending.Size }, cancellationToken);
    }

    /// <summary>
    /// An exit rule fired at <paramref name="at"/> with the market at <paramref name="markPrice"/>: closes at once
    /// without an exit latency, else sends the exit, quoted at that price, to fill one latency later.
    /// </summary>
    private async Task TriggerExitAsync(
        OpenPosition position, DateTimeOffset at, decimal markPrice, ExitReason reason, CancellationToken cancellationToken)
    {
        decimal quote = SellFill(position, markPrice, _marks[position.Mint].Pool);
        var pending = new PendingExit(reason, at.AddMilliseconds(_options.ExitLatencyMs), quote, Failures: 0);
        if (_options.ExitLatencyMs == 0)
        {
            await CloseAsync(position, at, quote, reason, ExitCostOf(pending), cancellationToken);
            return;
        }

        _pendingExits[position.Mint] = pending;
    }

    private async Task FillPendingExitAsync(
        OpenPosition position, PendingExit pending, Mark market, DateTimeOffset now, CancellationToken cancellationToken)
    {
        decimal exitPrice = SellFill(position, market.Price, market.Pool);
        bool breached = _options.MaxSlippagePct > 0
            && pending.Failures < _options.MaxExitRetries
            && exitPrice < pending.Quote * (1m - (decimal)_options.MaxSlippagePct);
        if (!breached)
        {
            await CloseAsync(position, now, exitPrice, pending.Reason, ExitCostOf(pending), cancellationToken);
            return;
        }

        // Re-sent, quoted at what the bot now sees; the failed attempt's fee is paid with the close.
        _pendingExits[position.Mint] = pending with
        {
            Due = now.AddMilliseconds(_options.ExitLatencyMs),
            Quote = exitPrice,
            Failures = pending.Failures + 1,
        };
        FailedExitCount++;
        MoveClock(now);

        await _log.WriteAsync(LogSeverity.Warn, "exit failed",
            new
            {
                tradeId = position.TradeId,
                mint = position.Mint,
                reason = pending.Reason.ToCode(),
                quote = pending.Quote,
                fillPrice = exitPrice,
                markPrice = market.Price,
                attempt = pending.Failures + 1,
            },
            cancellationToken);
    }

    /// <summary>One network fee per exit attempt, the failed ones included.</summary>
    private decimal ExitCostOf(PendingExit pending) => _options.NetworkFeeSol * (pending.Failures + 1);

    private decimal BuyFill(decimal size, Mark market) =>
        _options.RealisticFills ? Fills.BuyPrice(size, market.Price, market.Pool, _options) : market.Price;

    private decimal SellFill(OpenPosition position, decimal markPrice, Pool pool) =>
        _options.RealisticFills ? Fills.SellPrice(position.Entry, markPrice, pool, _options) : markPrice;

    private bool IsHeld(OpenPosition position) =>
        _positions.TryGetValue(position.Mint, out OpenPosition? held) && held.TradeId == position.TradeId;

    /// <summary>The pending order that runs out of marks first (due time + the mark age; mint, then entry first, as tie-breaks).</summary>
    private Expiry? NextExpiry()
    {
        TimeSpan window = TimeSpan.FromSeconds(_options.MaxMarkAgeSeconds);
        Expiry? next = null;
        foreach ((string mint, PendingEntry entry) in _pendingEntries)
        {
            next = Earlier(next, new Expiry(mint, entry.Due + window, IsExit: false));
        }

        foreach ((string mint, PendingExit exit) in _pendingExits)
        {
            next = Earlier(next, new Expiry(mint, exit.Due + window, IsExit: true));
        }

        return next;

        static Expiry Earlier(Expiry? current, Expiry candidate) =>
            current is null
            || candidate.At < current.At
            || (candidate.At == current.At && string.CompareOrdinal(candidate.Mint, current.Mint) < 0)
                ? candidate
                : current;
    }

    private void MoveClock(DateTimeOffset at)
    {
        if (Clock is null || at > Clock)
        {
            Clock = at;
        }
    }

    private bool IsFresh(Mark mark, DateTimeOffset at) =>
        at - mark.Ts <= TimeSpan.FromSeconds(_options.MaxMarkAgeSeconds);

    private bool FeedIsLive(DateTimeOffset at) =>
        _lastFeedTs is DateTimeOffset last && at - last <= TimeSpan.FromSeconds(_options.MaxFeedSilenceSeconds);

    /// <summary>
    /// The last mark if fresh at <paramref name="at"/> or the feed is still live; else the worse of it and the stop.
    /// </summary>
    private decimal StaleSafeExitPrice(OpenPosition position, DateTimeOffset at)
    {
        Mark mark = _marks[position.Mint];
        return IsFresh(mark, at) || FeedIsLive(at)
            ? mark.Price
            : Math.Min(mark.Price, position.Entry.Levels.StopLossPrice);
    }

    /// <summary>Closes <paramref name="position"/> at the fill price <paramref name="exitPrice"/>, paying <paramref name="exitCost"/>.</summary>
    private async Task CloseAsync(
        OpenPosition position, DateTimeOffset closedAt, decimal exitPrice, ExitReason reason, decimal exitCost,
        CancellationToken cancellationToken)
    {
        ClosedTrade trade = ClosedTrade.Settle(position, closedAt, exitPrice, reason, exitCost);

        await _store.CloseAsync(trade, cancellationToken);

        _positions.Remove(position.Mint);
        _pendingExits.Remove(position.Mint);
        Balance += trade.Proceeds;
        RealizedPnl += trade.Pnl;
        ClosedTradeCount++;
        MoveClock(closedAt);

        await _log.WriteAsync(LogSeverity.Info, "position closed",
            new
            {
                tradeId = trade.TradeId,
                mint = position.Mint,
                reason = reason.ToCode(),
                exitPrice = trade.ExitPrice,
                exitCost = trade.ExitCost,
                pnl = trade.Pnl,
                pnlPct = trade.PnlPct,
            },
            cancellationToken);
    }

    private readonly record struct Mark(decimal Price, DateTimeOffset Ts, Pool Pool);

    /// <param name="Quote">Fill price per token at the decision mark: the slippage tolerance's reference.</param>
    private sealed record PendingEntry(ModelPrediction Prediction, DateTimeOffset Due, decimal Size, decimal Quote);

    /// <param name="Quote">Fill price per token at the trigger's (or the last failed attempt's) mark.</param>
    /// <param name="Failures">Attempts that breached the slippage tolerance so far.</param>
    private sealed record PendingExit(ExitReason Reason, DateTimeOffset Due, decimal Quote, int Failures);

    private sealed record Expiry(string Mint, DateTimeOffset At, bool IsExit);
}
