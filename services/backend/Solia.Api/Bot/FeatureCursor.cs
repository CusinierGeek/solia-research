namespace Solia.Api.Bot;

/// <summary>
/// Which <c>features</c> ids the feed has read. A BIGSERIAL id is assigned at INSERT but becomes visible at
/// COMMIT, so a lower id can show up after a higher one: a plain "highest id read" cursor would skip it for
/// good. Ids jumped over stay pending (<see cref="Gaps"/>) and are asked for again until they show up or
/// <c>maxCommitDelay</c> passes — a rolled-back insert or a sequence cache jump never shows up. A jump wider
/// than <see cref="MaxTrackedGap"/> ids (a <c>setval</c>, a bulk load) is not tracked id by id: it is skipped.
/// Not thread-safe: the feed's poll loop is its only user.
/// </summary>
internal sealed class FeatureCursor(TimeProvider time, TimeSpan maxCommitDelay, ILogger logger)
{
    internal const int MaxTrackedGap = 10_000;

    // Missing id -> monotonic timestamp it was first jumped over.
    private readonly Dictionary<long, long> _gaps = [];

    /// <summary>The highest id read so far; null before the first row.</summary>
    public long? MaxSeen { get; private set; }

    /// <summary>The ids below <see cref="MaxSeen"/> not read yet, still expected to commit.</summary>
    public IReadOnlyCollection<long> Gaps => _gaps.Keys;

    /// <summary>
    /// Starts the cursor after <paramref name="id"/> (the highest id committed when the first read found no fresh
    /// row), so the ids allocated after it are waited for from the start. No-op once a row was read.
    /// </summary>
    public void Seed(long id) => MaxSeen ??= id;

    /// <summary>
    /// Records that row <paramref name="id"/> was read (rows are accepted in ascending id order within a
    /// read). False when it was already read, so the caller drops the duplicate.
    /// </summary>
    public bool TryAccept(long id)
    {
        if (_gaps.Remove(id))
        {
            return true;
        }

        if (MaxSeen is long max)
        {
            if (id <= max)
            {
                return false;
            }

            long missing = id - max - 1;
            if (missing > MaxTrackedGap)
            {
                logger.LogWarning("bot feed: features ids jumped from {From} to {To}; the {Missing} ids between are not waited for",
                    max, id, missing);
            }
            else if (missing > 0)
            {
                long now = time.GetTimestamp();
                for (long gap = max + 1; gap < id; gap++)
                {
                    _gaps.TryAdd(gap, now);
                }
            }
        }

        MaxSeen = id;
        return true;
    }

    /// <summary>Stops waiting for the ids missing for longer than the commit delay.</summary>
    public void ExpireGaps()
    {
        foreach ((long id, long since) in _gaps)
        {
            if (time.GetElapsedTime(since) >= maxCommitDelay)
            {
                _gaps.Remove(id);
            }
        }
    }
}
