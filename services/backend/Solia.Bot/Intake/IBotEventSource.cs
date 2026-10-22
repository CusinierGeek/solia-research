using Solia.Bot.Domain;

namespace Solia.Bot.Intake;

/// <summary>
/// The feed the bot consumes: an async stream of <see cref="BotEvent"/>s (predictions and price ticks)
/// in logical-time order. An injected abstraction so the host can be driven by an in-memory replay
/// today and a DB/live feed in Sprint 4 without touching <see cref="Trading.PositionManager"/> or the
/// <c>PaperTradingBot</c>. The stream completes when the source is exhausted; a live source stays open
/// until the passed <see cref="CancellationToken"/> fires.
/// </summary>
public interface IBotEventSource
{
    IAsyncEnumerable<BotEvent> ReadAsync(CancellationToken cancellationToken = default);
}
