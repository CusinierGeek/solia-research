using Solia.Api.Bot;

namespace Solia.Api.Tests.Bot;

/// <summary>The feed's scoring queue: every token with fresh data is scored, on its newest row, in turn.</summary>
public class ScoringQueueTests
{
    private static readonly DateTimeOffset T = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private static FeedRow Row(long id, string mint, double seconds) => new(id, mint, T.AddSeconds(seconds), 1m, true);

    [Fact]
    public async Task Tokens_are_served_oldest_waiting_first_and_none_is_dropped()
    {
        var queue = new ScoringQueue();
        for (int i = 0; i < 50; i++)
        {
            queue.Mark(Row(i, $"m{i}", i));
        }

        for (int i = 0; i < 50; i++)
        {
            Assert.Equal($"m{i}", (await queue.TakeAsync(CancellationToken.None)).Mint);
        }

        Assert.Equal(0, queue.Count);
    }

    [Fact]
    public async Task A_waiting_token_keeps_its_place_and_is_scored_on_its_newest_row()
    {
        var queue = new ScoringQueue();
        queue.Mark(Row(1, "a", 0));
        queue.Mark(Row(2, "b", 1));
        queue.Mark(Row(3, "a", 5));
        queue.Mark(Row(0, "a", -1)); // an older row (a late commit) doesn't replace a newer one

        Assert.Equal(2, queue.Count);
        Assert.Equal(Row(3, "a", 5), await queue.TakeAsync(CancellationToken.None));
        Assert.Equal(Row(2, "b", 1), await queue.TakeAsync(CancellationToken.None));
    }

    [Fact]
    public async Task Taking_waits_for_a_token_and_honours_cancellation()
    {
        var queue = new ScoringQueue();
        Task<FeedRow> take = queue.TakeAsync(CancellationToken.None);
        Assert.False(take.IsCompleted);

        queue.Mark(Row(1, "a", 0));
        Assert.Equal("a", (await take.WaitAsync(TimeSpan.FromSeconds(5))).Mint);

        using var cts = new CancellationTokenSource();
        Task<FeedRow> cancelled = queue.TakeAsync(cts.Token);
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
    }
}
