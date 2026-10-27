using Solia.Dashboard.Api;

namespace Solia.Dashboard.Logs;

/// <summary>
/// The log entries on screen, newest first (<c>ts</c>, then <c>id</c>, as <c>GET /logs/bot</c> orders them) and
/// never more than <see cref="Capacity"/>: a fetched snapshot <see cref="Reset"/>s it, a pushed entry is
/// <see cref="Add"/>ed in place (on top, unless it arrives late) and the oldest beyond the cap are dropped. An entry
/// already held (same id) is not added twice. Not thread-safe: use it from the renderer's dispatcher.
/// </summary>
public sealed class LogTail(int capacity = BotLogsFilter.DefaultLimit)
{
    private readonly List<LogDto> _rows = [];
    private readonly HashSet<long> _ids = [];

    public int Capacity { get; private set; } = capacity > 0 ? capacity : throw new ArgumentOutOfRangeException(nameof(capacity));

    /// <summary>The entries held, newest first.</summary>
    public IReadOnlyList<LogDto> Rows => _rows;

    /// <summary>Replaces the entries with <paramref name="snapshot"/> (any order), keeping the newest <paramref name="capacity"/>.</summary>
    public void Reset(IEnumerable<LogDto> snapshot, int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Capacity = capacity;
        _rows.Clear();
        _ids.Clear();
        // One sort rather than an Add (a scan for its place) per entry.
        foreach (LogDto log in snapshot.OrderByDescending(l => l.Ts).ThenByDescending(l => l.Id))
        {
            if (_rows.Count == capacity)
            {
                break;
            }
            if (_ids.Add(log.Id))
            {
                _rows.Add(log);
            }
        }
    }

    /// <summary>Whether the entry with that <paramref name="id"/> is held.</summary>
    public bool Contains(long id) => _ids.Contains(id);

    /// <summary>Adds a pushed entry; false when it was already held, or is older than every entry of a full tail.</summary>
    public bool Add(LogDto log)
    {
        if (_ids.Contains(log.Id))
        {
            return false;
        }

        int index = 0;
        while (index < _rows.Count && IsNewer(_rows[index], log))
        {
            index++;
        }
        if (index >= Capacity)
        {
            return false;
        }

        _rows.Insert(index, log);
        _ids.Add(log.Id);
        if (_rows.Count > Capacity)
        {
            _ids.Remove(_rows[^1].Id);
            _rows.RemoveAt(_rows.Count - 1);
        }
        return true;
    }

    private static bool IsNewer(LogDto a, LogDto b) => a.Ts != b.Ts ? a.Ts > b.Ts : a.Id > b.Id;
}
