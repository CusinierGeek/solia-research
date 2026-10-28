using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Tests.Features;

public class HolderCounterTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    [Fact]
    public void Remembers_each_wallets_earliest_sighting_in_any_order()
    {
        var counter = new HolderCounter();
        counter.Add("A", T0.AddSeconds(10));
        counter.Add("A", T0); // late: earlier sighting wins
        counter.Add("A", T0.AddSeconds(20));

        Assert.True(counter.TryGetFirstSeen("A", out DateTimeOffset first));
        Assert.Equal(T0, first);
        Assert.False(counter.TryGetFirstSeen("B", out _));
    }

    [Fact]
    public void Forgets_first_sightings_once_it_is_a_sketch()
    {
        var counter = new HolderCounter();
        for (int i = 0; i <= HolderCounter.ExactLimit; i++)
        {
            counter.Add($"wallet-{i}", T0);
        }

        Assert.False(counter.TryGetFirstSeen("wallet-0", out _));
    }

    [Fact]
    public void Counts_exactly_up_to_the_exact_limit()
    {
        var counter = new HolderCounter();
        for (int i = 0; i < HolderCounter.ExactLimit; i++)
        {
            counter.Add($"wallet-{i}", T0);
            counter.Add($"wallet-{i}", T0); // duplicates never count twice
        }

        Assert.Equal(HolderCounter.ExactLimit, counter.Count);
    }

    [Theory]
    [InlineData(5_000)]
    [InlineData(150_000)] // past the old 100k hard cap: keeps counting
    public void Estimates_large_counts_within_five_percent(int wallets)
    {
        var counter = new HolderCounter();
        for (int i = 0; i < wallets; i++)
        {
            counter.Add($"wallet-{i}", T0);
        }

        Assert.InRange(counter.Count, wallets * 0.95, wallets * 1.05);
    }

    [Fact]
    public void Repeat_wallets_do_not_inflate_the_estimate()
    {
        var counter = new HolderCounter();
        for (int pass = 0; pass < 3; pass++)
        {
            for (int i = 0; i < 5_000; i++)
            {
                counter.Add($"wallet-{i}", T0);
            }
        }

        Assert.InRange(counter.Count, 4_750, 5_250);
    }
}
