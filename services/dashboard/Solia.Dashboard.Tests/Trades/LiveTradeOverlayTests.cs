using Solia.Dashboard.Api;
using Solia.Dashboard.Trades;

namespace Solia.Dashboard.Tests.Trades;

public class LiveTradeOverlayTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private static TradeDto Open(long id) =>
        new(id, $"mint{id}", null, T0.AddMinutes(id), null, 0.000001m, null, 0.5m, null, null, null, "buy", 70, null, "open");

    private static TradeDto Close(long id, decimal pnl = 0.1m) =>
        Open(id) with
        {
            ClosedAt = T0.AddMinutes(id + 1),
            ExitPrice = 0.0000012m,
            DurationSec = 60,
            Pnl = pnl,
            PnlPct = 0.2,
            ExitReason = "tp",
            Status = "closed",
        };

    private static PagedResult<TradeDto> Page(int pageSize, long total, params TradeDto[] items) => new(items, 1, pageSize, total);

    private static long[] Ids(PagedResult<TradeDto> page) => [.. page.Items.Select(t => t.Id)];

    [Fact]
    public void With_nothing_pushed_the_page_is_unchanged()
    {
        PagedResult<TradeDto> page = Page(50, 2, Open(2), Open(1));

        Assert.Same(page, new LiveTradeOverlay().Apply(page));
    }

    [Fact]
    public void Opened_trades_go_on_top_newest_first_and_count_in_the_total()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(3));
        overlay.Opened(Open(4));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 2, Open(2), Open(1)));

        Assert.Equal([4, 3, 2, 1], Ids(shown));
        Assert.Equal(4, shown.Total);
    }

    [Fact]
    public void The_page_is_trimmed_back_to_its_size()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(3));

        PagedResult<TradeDto> shown = overlay.Apply(Page(2, 5, Open(2), Open(1)));

        Assert.Equal([3, 2], Ids(shown));
        Assert.Equal(6, shown.Total);
    }

    [Fact]
    public void A_closed_trade_replaces_its_row_in_place()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Closed(Close(1, pnl: -0.2m));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 2, Open(2), Open(1)));

        Assert.Equal([2, 1], Ids(shown));
        Assert.Equal("closed", shown.Items[1].Status);
        Assert.Equal(-0.2m, shown.Items[1].Pnl);
        Assert.Equal(2, shown.Total);
    }

    [Fact]
    public void A_closed_trade_not_on_the_page_is_not_added()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Closed(Close(9));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 1, Open(1)));

        Assert.Equal([1], Ids(shown));
        Assert.Equal(1, shown.Total);
    }

    [Fact]
    public void A_trade_opened_then_closed_live_stays_on_top_closed()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(3));
        overlay.Closed(Close(3));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 1, Open(1)));

        Assert.Equal([3, 1], Ids(shown));
        Assert.Equal("closed", shown.Items[0].Status);
    }

    [Fact]
    public void A_closed_trade_never_goes_back_to_open()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Closed(Close(3));
        overlay.Opened(Open(3)); // a late, stale open

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 0));

        Assert.Equal("closed", Assert.Single(shown.Items).Status);

        // Nor does a page fetched before the close undo a pushed close.
        PagedResult<TradeDto> fetched = overlay.Apply(Page(50, 1, Open(3)));
        Assert.Equal("closed", Assert.Single(fetched.Items).Status);
    }

    [Fact]
    public void A_fetch_forgets_the_pushes_it_reflects_but_not_later_ones()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(3));
        long mark = overlay.Mark;   // the fetch starts here…
        overlay.Opened(Open(4));    // …and this push lands while it is in flight

        overlay.Forget(mark);

        Assert.Equal(1, overlay.Count);
        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 3, Open(3), Open(2), Open(1)));
        Assert.Equal([4, 3, 2, 1], Ids(shown));
        Assert.Equal(4, shown.Total);
    }

    [Fact]
    public void A_pushed_trade_already_on_the_fetched_page_is_not_duplicated()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(3));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 3, Open(3), Open(2), Open(1)));

        Assert.Equal([3, 2, 1], Ids(shown));
        Assert.Equal(3, shown.Total);
    }

    [Fact]
    public void Past_the_capacity_nothing_is_dropped_but_it_asks_for_a_refetch()
    {
        var overlay = new LiveTradeOverlay(capacity: 2);
        overlay.Opened(Open(1));
        Assert.False(overlay.IsFull);
        overlay.Opened(Open(2));
        Assert.True(overlay.IsFull);
        overlay.Opened(Open(3));

        PagedResult<TradeDto> shown = overlay.Apply(Page(50, 0));
        Assert.Equal([3, 2, 1], Ids(shown)); // every opened trade still counts
        Assert.Equal(3, shown.Total);

        overlay.Forget(overlay.Mark);       // the re-fetched page reflects them all
        Assert.False(overlay.IsFull);
    }

    [Fact]
    public void The_overlaid_page_is_reused_until_something_changes()
    {
        var overlay = new LiveTradeOverlay();
        PagedResult<TradeDto> page = Page(50, 1, Open(1));
        overlay.Opened(Open(2));

        PagedResult<TradeDto> first = overlay.Apply(page);
        Assert.Same(first, overlay.Apply(page));

        overlay.Closed(Close(2));
        PagedResult<TradeDto> afterClose = overlay.Apply(page);
        Assert.NotSame(first, afterClose);
        Assert.Equal("closed", afterClose.Items[0].Status);

        PagedResult<TradeDto> otherPage = Page(50, 1, Open(1));
        Assert.NotSame(afterClose, overlay.Apply(otherPage));

        overlay.Forget(overlay.Mark);
        Assert.Same(otherPage, overlay.Apply(otherPage));
    }

    [Fact]
    public void Clear_drops_everything()
    {
        var overlay = new LiveTradeOverlay();
        overlay.Opened(Open(1));

        overlay.Clear();

        Assert.Equal(0, overlay.Count);
    }
}
