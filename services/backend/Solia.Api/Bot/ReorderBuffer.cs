using Solia.Bot.Domain;

namespace Solia.Api.Bot;

/// <summary>
/// Holds the feed's events for up to <c>window</c> so the ones written slightly out of time order — across
/// polls, or a prediction scored after newer ticks — still reach the engine in time order, instead of
/// arriving behind its clock (where it drops anything later than <c>Bot:MaxEventLagSeconds</c>). An event is
/// released once an event <c>window</c> newer has been seen, or once it has waited <c>window</c> (a quiet
/// market still flows); releases are in (time, arrival) order. A zero window releases everything at once.
/// Not thread-safe: the feed serializes its calls.
/// </summary>
internal sealed class ReorderBuffer(TimeProvider time, TimeSpan window)
{
    private readonly List<(BotEvent Event, long Seq, long ArrivedAt)> _held = [];
    private long _seq;
    private DateTimeOffset? _newest;

    public int Count => _held.Count;

    public void Add(BotEvent botEvent)
    {
        _held.Add((botEvent, _seq++, time.GetTimestamp()));
        if (_newest is null || botEvent.Ts > _newest)
        {
            _newest = botEvent.Ts;
        }
    }

    /// <summary>Removes and returns the events due, in time order.</summary>
    public List<BotEvent> Release()
    {
        if (_held.Count == 0)
        {
            return [];
        }

        DateTimeOffset cutoff = _newest!.Value - window;
        foreach ((BotEvent botEvent, _, long arrivedAt) in _held)
        {
            if (botEvent.Ts > cutoff && time.GetElapsedTime(arrivedAt) >= window)
            {
                cutoff = botEvent.Ts;
            }
        }

        var due = _held.Where(h => h.Event.Ts <= cutoff).OrderBy(h => h.Event.Ts).ThenBy(h => h.Seq).ToList();
        _held.RemoveAll(h => h.Event.Ts <= cutoff);
        return due.ConvertAll(h => h.Event);
    }
}
