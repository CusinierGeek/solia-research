using System.ComponentModel.DataAnnotations;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Trading;

namespace Solia.Bot.Backtest;

/// <summary>
/// Drives the <see cref="PositionManager"/> engine over a fixed <see cref="BotEvent"/> sequence and
/// reports the result — a reusable, deterministic backtest tool. It is the Sprint 3 Definition of Done:
/// given a replay of predictions and prices, the engine opens/closes positions and the resulting trades,
/// balance/equity and win-rate / total-P&amp;L are reproducible to the lamport.
///
/// <para><b>Determinism.</b> The only clock is the event timestamp — no wall time, no randomness, no
/// idle-timeout sweep (that is the live host's concern; see <c>PaperTradingBot</c>). Replaying the same
/// events with the same options and <see cref="ReplayOrder"/> always yields an equal <see cref="ReplayResult"/>.</para>
///
/// <para><b>Ordering.</b> <see cref="ReplayOrder.Arrival"/> (the default) feeds events as given, so a
/// recorded live stream replays exactly as the live bot saw it, late events and drops included.
/// <see cref="ReplayOrder.Timestamp"/> sorts into canonical event-time order with a fixed tie-break
/// (price before prediction at an equal timestamp), for data whose enumeration order is not meaningful.</para>
///
/// <para><b>Timeouts.</b> A position expires only when an event's timestamp crosses its deadline (the
/// engine sweeps timeouts up to each event's time). A position whose deadline falls after the last event
/// is left open and reported in <see cref="ReplayResult.OpenPositions"/> — the replay never invents a
/// clock move past the data to close it.</para>
///
/// <para><b>Paper only.</b> Trades are captured in an <see cref="InMemoryTradeStore"/>; nothing reaches a
/// database, a wallet or the chain.</para>
/// </summary>
public static class ReplayHarness
{
    /// <summary>
    /// Replays <paramref name="events"/> through a fresh engine configured with <paramref name="options"/>
    /// and returns the trades, final balance/equity and summary stats. The options are validated with the
    /// same rules the live host enforces at startup; the caller's sequence is left untouched.
    /// </summary>
    /// <exception cref="ValidationException">The options break a <see cref="BotOptions"/> rule.</exception>
    public static async Task<ReplayResult> RunAsync(
        BotOptions options,
        IEnumerable<BotEvent> events,
        ReplayOrder order = ReplayOrder.Arrival,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(events);
        Validator.ValidateObject(options, new ValidationContext(options), validateAllProperties: true);

        var store = new InMemoryTradeStore();
        var engine = new PositionManager(options, store);

        foreach (BotEvent botEvent in Sequence(events, order))
        {
            await engine.HandleAsync(botEvent, cancellationToken);
        }

        return new ReplayResult(
            options.StartingBalance,
            engine.Balance,
            engine.Equity,
            engine.RealizedPnl,
            store.Closed,
            [.. engine.OpenPositions.Values.OrderBy(p => p.Entry.OpenedAt).ThenBy(p => p.TradeId)],
            engine.DroppedEventCount);
    }

    private static IEnumerable<BotEvent> Sequence(IEnumerable<BotEvent> events, ReplayOrder order) => order switch
    {
        ReplayOrder.Arrival => events,
        // OrderBy/ThenBy are stable, so input order is the last tie-break.
        ReplayOrder.Timestamp => events.OrderBy(e => e.Ts).ThenBy(e => e is BotEvent.Price ? 0 : 1),
        _ => throw new ArgumentOutOfRangeException(nameof(order), order, "Unknown replay order."),
    };
}
