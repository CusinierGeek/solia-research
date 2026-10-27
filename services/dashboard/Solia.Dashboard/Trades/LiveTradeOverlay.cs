using Solia.Dashboard.Api;

namespace Solia.Dashboard.Trades;

/// <summary>
/// The trades pushed live (<c>trade_opened</c> / <c>trade_closed</c>) since the newest page was fetched, laid over it
/// at render time: a pushed trade replaces its row, and an opened trade missing from the page goes on top (newest
/// first, the page trimmed back to its size). Only meaningful over the newest unfiltered page
/// (<see cref="TradeHistoryState.IsLive"/>). A fetch takes a <see cref="Mark"/> before it starts and
/// <see cref="Forget"/>s up to it once its page arrives — that page already reflects those trades — so a push that
/// lands while it is in flight still applies on top. A closed trade never goes back to open, whatever the order the
/// events and the fetch arrive in. Nothing is ever dropped (that would lose an opened trade from the total): once it
/// holds <paramref name="capacity"/> trades it is <see cref="IsFull"/>, and the host should re-fetch the page so the
/// next <see cref="Forget"/> empties it. The overlaid page is cached until the next change. Not thread-safe: use it
/// from the renderer's dispatcher.
/// </summary>
public sealed class LiveTradeOverlay(int capacity = LiveTradeOverlay.DefaultCapacity)
{
    public const int DefaultCapacity = 1000;

    private const string ClosedStatus = "closed";

    private readonly Dictionary<long, Entry> _entries = [];
    private long _sequence;
    private PagedResult<TradeDto>? _appliedTo;
    private PagedResult<TradeDto>? _applied;

    /// <summary>How many pushed trades are held.</summary>
    public int Count => _entries.Count;

    /// <summary>Whether it holds as many trades as it should: time to re-fetch the page.</summary>
    public bool IsFull => _entries.Count >= capacity;

    /// <summary>The position of the latest push: what a fetch starting now will reflect.</summary>
    public long Mark => _sequence;

    /// <summary>A <c>trade_opened</c>: the trade belongs on top of the newest page.</summary>
    public void Opened(TradeDto trade) => Put(trade, opened: true);

    /// <summary>A <c>trade_closed</c>: the trade's row (if shown) is updated in place.</summary>
    public void Closed(TradeDto trade) => Put(trade, opened: false);

    /// <summary>Drops the pushes up to <paramref name="mark"/>: a page fetched after them already shows them.</summary>
    public void Forget(long mark)
    {
        foreach (long id in _entries.Where(e => e.Value.Sequence <= mark).Select(e => e.Key).ToList())
        {
            _entries.Remove(id);
            _applied = null;
        }
    }

    public void Clear()
    {
        _entries.Clear();
        _applied = null;
    }

    /// <summary><paramref name="page"/> with the pushed trades laid over it.</summary>
    public PagedResult<TradeDto> Apply(PagedResult<TradeDto> page)
    {
        if (_entries.Count == 0)
        {
            return page;
        }
        if (_applied is not null && ReferenceEquals(page, _appliedTo))
        {
            return _applied;
        }

        var onPage = new HashSet<long>(page.Items.Select(t => t.Id));
        List<TradeDto> added = _entries.Values
            .Where(e => e.Opened && !onPage.Contains(e.Trade.Id))
            .Select(e => e.Trade)
            .OrderByDescending(t => t.OpenedAt)
            .ThenByDescending(t => t.Id)
            .ToList();
        IEnumerable<TradeDto> updated = page.Items.Select(t => _entries.TryGetValue(t.Id, out Entry? e) ? Latest(t, e.Trade) : t);
        TradeDto[] items = [.. added.Concat(updated).Take(Math.Max(page.PageSize, page.Items.Count))];
        _appliedTo = page;
        _applied = page with { Items = items, Total = page.Total + added.Count };
        return _applied;
    }

    private void Put(TradeDto trade, bool opened)
    {
        _entries.TryGetValue(trade.Id, out Entry? existing);
        _entries[trade.Id] = new Entry(
            existing is null ? trade : Latest(existing.Trade, trade),
            opened || existing?.Opened == true,
            ++_sequence);
        _applied = null;
    }

    // A closed trade is final: an "open" copy arriving after it is stale.
    private static TradeDto Latest(TradeDto current, TradeDto incoming) =>
        current.Status == ClosedStatus && incoming.Status != ClosedStatus ? current : incoming;

    private sealed record Entry(TradeDto Trade, bool Opened, long Sequence);
}
