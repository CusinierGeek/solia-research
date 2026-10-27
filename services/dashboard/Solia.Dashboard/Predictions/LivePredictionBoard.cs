using Solia.Dashboard.Api;

namespace Solia.Dashboard.Predictions;

/// <summary>
/// The live predictions view's rows: the latest prediction per (mint, horizon), as <c>GET /predictions/live</c>
/// returns them, kept current by <c>prediction_created</c> pushes. Bounded: past <see cref="Capacity"/> the row
/// with the oldest <c>ts</c> is dropped. Not thread-safe — the view only touches it on the renderer's dispatcher.
/// </summary>
public sealed class LivePredictionBoard
{
    private readonly Dictionary<(string Mint, string Horizon), PredictionRecordDto> _rows = [];

    // The same rows oldest first, so eviction past capacity is O(log n) instead of a scan.
    private readonly SortedSet<PredictionRecordDto> _byAge = new(AgeOrder.Instance);

    // The last Sorted result, reused until a row changes: renders (pushes, age ticks) far outnumber mutations.
    private (PredictionSort Sort, bool Descending, string? Horizon)? _viewKey;
    private IReadOnlyList<PredictionRecordDto> _view = [];

    public LivePredictionBoard(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
    }

    public int Capacity { get; }

    public int Count => _rows.Count;

    /// <summary>Replaces every row with <paramref name="snapshot"/> (a REST re-fetch).</summary>
    public void Reset(IEnumerable<PredictionRecordDto> snapshot)
    {
        _rows.Clear();
        _byAge.Clear();
        _viewKey = null;
        foreach (PredictionRecordDto prediction in snapshot)
        {
            Upsert(prediction);
        }
    }

    /// <summary>
    /// Makes <paramref name="prediction"/> its token's row at its horizon, unless that row is already newer (later
    /// <c>ts</c>, or same <c>ts</c> and a higher-or-equal <c>id</c>: a duplicate or late event). Returns whether the
    /// rows changed — including when another row was evicted to make room.
    /// </summary>
    public bool Upsert(PredictionRecordDto prediction)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        var key = (prediction.Mint, prediction.Horizon);
        if (_rows.TryGetValue(key, out PredictionRecordDto? current))
        {
            if (!IsNewer(prediction, current))
            {
                return false;
            }
            _byAge.Remove(current);
        }

        _rows[key] = prediction;
        _byAge.Add(prediction);
        _viewKey = null;
        if (_rows.Count <= Capacity)
        {
            return true;
        }

        PredictionRecordDto oldest = _byAge.Min!;
        _byAge.Remove(oldest);
        _rows.Remove((oldest.Mint, oldest.Horizon));
        return !ReferenceEquals(oldest, prediction);
    }

    /// <summary>
    /// The rows (only <paramref name="horizon"/>'s when given) ordered by <paramref name="sort"/>, missing values
    /// last whatever the direction; ties newest first. The same list is returned until a row changes.
    /// </summary>
    public IReadOnlyList<PredictionRecordDto> Sorted(PredictionSort sort, bool descending = true, string? horizon = null)
    {
        if (_viewKey == (sort, descending, horizon))
        {
            return _view;
        }

        IEnumerable<PredictionRecordDto> rows = horizon is null ? _rows.Values : _rows.Values.Where(p => p.Horizon == horizon);
        IOrderedEnumerable<PredictionRecordDto> ordered = sort switch
        {
            PredictionSort.PumpProb => ByValue(rows, p => p.PumpProb, descending),
            PredictionSort.RugProb => ByValue(rows, p => p.RugProb, descending),
            PredictionSort.Confidence => ByValue(rows, p => p.Confidence, descending),
            _ => descending ? rows.OrderByDescending(p => p.Ts) : rows.OrderBy(p => p.Ts),
        };
        _view = ordered.ThenByDescending(p => p.Ts).ThenByDescending(p => p.Id).ToList();
        _viewKey = (sort, descending, horizon);
        return _view;
    }

    private static bool IsNewer(PredictionRecordDto candidate, PredictionRecordDto current) =>
        candidate.Ts > current.Ts || (candidate.Ts == current.Ts && candidate.Id > current.Id);

    private static IOrderedEnumerable<PredictionRecordDto> ByValue(
        IEnumerable<PredictionRecordDto> rows, Func<PredictionRecordDto, double?> value, bool descending)
    {
        IOrderedEnumerable<PredictionRecordDto> present = rows.OrderBy(p => value(p) is null);
        return descending ? present.ThenByDescending(value) : present.ThenBy(value);
    }

    /// <summary>Oldest first: (ts, id), then the row key so two distinct rows never compare equal.</summary>
    private sealed class AgeOrder : IComparer<PredictionRecordDto>
    {
        public static readonly AgeOrder Instance = new();

        public int Compare(PredictionRecordDto? x, PredictionRecordDto? y)
        {
            if (ReferenceEquals(x, y))
            {
                return 0;
            }
            if (x is null || y is null)
            {
                return x is null ? -1 : 1;
            }

            int byTs = x.Ts.CompareTo(y.Ts);
            if (byTs != 0)
            {
                return byTs;
            }
            int byId = x.Id.CompareTo(y.Id);
            if (byId != 0)
            {
                return byId;
            }
            int byMint = string.CompareOrdinal(x.Mint, y.Mint);
            return byMint != 0 ? byMint : string.CompareOrdinal(x.Horizon, y.Horizon);
        }
    }
}
