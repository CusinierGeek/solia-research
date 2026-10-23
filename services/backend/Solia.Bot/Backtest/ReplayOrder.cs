namespace Solia.Bot.Backtest;

/// <summary>How <see cref="ReplayHarness"/> sequences the events it is given before the engine sees them.</summary>
public enum ReplayOrder
{
    /// <summary>
    /// Apply events exactly as given — the order they arrived in. Use it to replay a recorded live
    /// stream faithfully: the engine's late-event rule runs as it did live (an event less than
    /// <c>MaxEventLagSeconds</c> behind the clock is applied at the clock's time, an older one is dropped
    /// and counted in <see cref="ReplayResult.DroppedEventCount"/>).
    /// </summary>
    Arrival,

    /// <summary>
    /// Sort into canonical event-time order: ascending timestamp, then a <see cref="Domain.BotEvent.Price"/>
    /// before a <see cref="Domain.BotEvent.Prediction"/> at the same instant (the mark is known when the
    /// decision is made), then input order. The result depends on the data alone, never on how a query or
    /// a feed merge enumerated it. No event is ever late in this order, so nothing is dropped.
    /// </summary>
    Timestamp,
}
