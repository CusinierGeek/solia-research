namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// The engine's snapshot clock, driven by event timestamps (so live and replay behave the same).
/// It only moves forward with plausible events: a single event dated more than the skew past the
/// clock is ignored (one bad timestamp can't fast-forward the clock and wipe the window), but
/// <c>confirmations</c> consecutive such events mean time really jumped — e.g. a missing replay
/// hour — and the clock follows. Symmetrically, that many consecutive events far behind mean the
/// clock itself was wrong (seeded by a bad first event) and it resets to them.
/// </summary>
internal sealed class EventClock(TimeSpan maxSkew, int confirmations)
{
    private int _aheadRun;
    private DateTimeOffset _aheadMin;
    private int _behindRun;
    private DateTimeOffset _behindMax;

    public DateTimeOffset? Now { get; private set; }

    /// <summary>Feeds one event timestamp; returns the (possibly unchanged) clock.</summary>
    public DateTimeOffset Observe(DateTimeOffset timestamp)
    {
        if (Now is not { } now)
        {
            Now = timestamp;
            return timestamp;
        }

        if (timestamp > now + maxSkew)
        {
            _behindRun = 0;
            _aheadMin = _aheadRun++ == 0 || timestamp < _aheadMin ? timestamp : _aheadMin;
            if (_aheadRun >= confirmations)
            {
                Now = _aheadMin; // the earliest of the run: conservative
                _aheadRun = 0;
            }
        }
        else if (timestamp < now - maxSkew)
        {
            _aheadRun = 0;
            _behindMax = _behindRun++ == 0 || timestamp > _behindMax ? timestamp : _behindMax;
            if (_behindRun >= confirmations)
            {
                Now = _behindMax;
                _behindRun = 0;
            }
        }
        else
        {
            _aheadRun = 0;
            _behindRun = 0;
            if (timestamp > now)
            {
                Now = timestamp;
            }
        }

        return Now.Value;
    }
}
