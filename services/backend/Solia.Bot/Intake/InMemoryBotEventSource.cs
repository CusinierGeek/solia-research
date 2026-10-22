using System.Runtime.CompilerServices;
using Solia.Bot.Domain;

namespace Solia.Bot.Intake;

/// <summary>
/// An <see cref="IBotEventSource"/> that replays a fixed, in-memory sequence of events — for unit tests,
/// the deterministic replay harness (T6) and any host that replays a fixed sequence (Sprint 4 wires a live feed).
/// The sequence is enumerated once per <see cref="ReadAsync"/> call and the stream completes when it is
/// exhausted; cancellation stops it between events.
/// </summary>
public sealed class InMemoryBotEventSource(IEnumerable<BotEvent> events) : IBotEventSource
{
    /// <summary>A source with no events — the stream completes immediately.</summary>
    public static InMemoryBotEventSource Empty { get; } = new([]);

    private readonly IEnumerable<BotEvent> _events = events ?? throw new ArgumentNullException(nameof(events));

    public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        foreach (BotEvent botEvent in _events)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return botEvent;
            await Task.Yield();
        }
    }
}
