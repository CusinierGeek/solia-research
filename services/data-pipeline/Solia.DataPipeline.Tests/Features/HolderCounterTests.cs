using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Tests.Features;

public class HolderCounterTests
{
    [Fact]
    public void Counts_exactly_up_to_the_exact_limit()
    {
        var counter = new HolderCounter();
        for (int i = 0; i < HolderCounter.ExactLimit; i++)
        {
            counter.Add($"wallet-{i}");
            counter.Add($"wallet-{i}"); // duplicates never count twice
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
            counter.Add($"wallet-{i}");
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
                counter.Add($"wallet-{i}");
            }
        }

        Assert.InRange(counter.Count, 4_750, 5_250);
    }
}
