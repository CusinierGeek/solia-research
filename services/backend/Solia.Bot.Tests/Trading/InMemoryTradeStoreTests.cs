using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

public class InMemoryTradeStoreTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static TradeOpening Opening() => new(
        "mint1", T0, 0.0001m, 2m,
        ExitLevels.Compute(PolicyFixtures.Buyable(), 0.0001m, PolicyFixtures.Options()),
        PolicyFixtures.Buyable());

    private static ClosedTrade Close(long id) =>
        ClosedTrade.Settle(new OpenPosition(id, Opening(), T0.AddSeconds(900)), T0.AddSeconds(1), 0.0001m, ExitReason.Signal);

    [Fact]
    public async Task Ids_are_sequential_and_a_close_moves_the_trade()
    {
        var store = new InMemoryTradeStore();

        Assert.Equal(1, await store.OpenAsync(Opening()));
        Assert.Equal(2, await store.OpenAsync(Opening()));

        await store.CloseAsync(Close(1));
        Assert.Equal([2L], store.Open.Keys);
        Assert.Equal(1, Assert.Single(store.Closed).TradeId);
    }

    [Fact]
    public async Task Closing_an_unknown_or_closed_trade_throws()
    {
        var store = new InMemoryTradeStore();
        long id = await store.OpenAsync(Opening());
        await store.CloseAsync(Close(id));

        await Assert.ThrowsAsync<InvalidOperationException>(() => store.CloseAsync(Close(id)));
        await Assert.ThrowsAsync<InvalidOperationException>(() => store.CloseAsync(Close(99)));
    }

    [Fact]
    public async Task A_cancelled_call_records_nothing()
    {
        var store = new InMemoryTradeStore();
        await Assert.ThrowsAsync<OperationCanceledException>(() => store.OpenAsync(Opening(), new CancellationToken(true)));
        Assert.Empty(store.Open);
    }

    [Fact]
    public async Task Closed_trades_are_counted_since_a_cutoff()
    {
        var store = new InMemoryTradeStore();
        long a = await store.OpenAsync(Opening());
        long b = await store.OpenAsync(Opening());
        await store.OpenAsync(Opening()); // still open: not counted
        await store.CloseAsync(Close(a));
        await store.CloseAsync(Close(b) with { ClosedAt = T0.AddSeconds(10) });

        Assert.Equal(2, await store.CountClosedSinceAsync(null));
        Assert.Equal(1, await store.CountClosedSinceAsync(T0.AddSeconds(1)));
        Assert.Equal(0, await store.CountClosedSinceAsync(T0.AddSeconds(10)));
    }
}
