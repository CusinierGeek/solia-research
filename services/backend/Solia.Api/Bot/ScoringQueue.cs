namespace Solia.Api.Bot;

/// <summary>
/// The tokens waiting to be scored, one entry per token holding its newest row: a token that gets new rows
/// while it waits keeps its place and is scored on the newest one. Served oldest-waiting first, so every
/// token with fresh data gets its turn however many are active (none is dropped). Thread-safe: the poll
/// loop marks, the scoring loop takes.
/// </summary>
internal sealed class ScoringQueue
{
    private readonly OrderedDictionary<string, FeedRow> _pending = new(StringComparer.Ordinal);
    private readonly SemaphoreSlim _available = new(0);

    public int Count
    {
        get
        {
            lock (_pending)
            {
                return _pending.Count;
            }
        }
    }

    /// <summary>Queues <paramref name="row"/>'s token, or refreshes its waiting row if this one is newer.</summary>
    public void Mark(FeedRow row)
    {
        lock (_pending)
        {
            if (_pending.TryGetValue(row.Mint, out FeedRow? waiting))
            {
                if ((row.Ts, row.Id).CompareTo((waiting.Ts, waiting.Id)) > 0)
                {
                    _pending[row.Mint] = row;
                }

                return;
            }

            _pending.Add(row.Mint, row);
        }

        _available.Release();
    }

    /// <summary>Waits for and removes the token that has waited longest.</summary>
    public async Task<FeedRow> TakeAsync(CancellationToken cancellationToken)
    {
        await _available.WaitAsync(cancellationToken);
        lock (_pending)
        {
            FeedRow row = _pending.GetAt(0).Value;
            _pending.RemoveAt(0);
            return row;
        }
    }
}
