using Solia.Dashboard.Api;
using Solia.Dashboard.Stats;

namespace Solia.Dashboard.Tests.Stats;

public class WinRateSlicesTests
{
    [Fact]
    public void Slices_are_wins_losses_and_break_even_in_that_order()
    {
        WinRateSlices slices = WinRateSlices.From(new WinRateResponse(0.58, 290, 190, 20, "all"));

        Assert.Equal(["Wins", "Losses", "Break-even"], slices.Labels);
        Assert.Equal([290.0, 190.0, 20.0], slices.Values);
        Assert.Equal(0.58, slices.Winrate);
        Assert.Equal(500, slices.Closed);
        Assert.False(slices.IsEmpty);
    }

    [Fact]
    public void Break_even_alone_is_not_empty()
    {
        Assert.False(WinRateSlices.From(new WinRateResponse(0, 0, 0, 3, "all")).IsEmpty);
    }

    [Fact]
    public void Nothing_closed_is_empty()
    {
        WinRateSlices slices = WinRateSlices.From(new WinRateResponse(0, 0, 0, 0, "all"));

        Assert.True(slices.IsEmpty);
        Assert.Equal(0, slices.Closed);
    }
}
