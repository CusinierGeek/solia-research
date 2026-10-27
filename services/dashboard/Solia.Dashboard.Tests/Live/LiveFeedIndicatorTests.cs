using AngleSharp.Dom;
using Microsoft.Extensions.DependencyInjection;
using Solia.Dashboard.Components.Layout;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

public class LiveFeedIndicatorTests : MudTestContext
{
    private readonly FakeLiveFeed _feed = new(LiveFeedState.Disconnected);

    public LiveFeedIndicatorTests() => Services.AddSingleton<ILiveFeed>(_feed);

    [Theory]
    [InlineData(LiveFeedState.Connected, "connected", "LIVE")]
    [InlineData(LiveFeedState.Reconnecting, "reconnecting", "RECONNECTING")]
    [InlineData(LiveFeedState.Disconnected, "disconnected", "OFFLINE")]
    public async Task Shows_the_current_state(LiveFeedState state, string dataState, string label)
    {
        await _feed.SetStateAsync(state);

        IElement chip = Render<LiveFeedIndicator>().Find(".solia-live-indicator");

        Assert.Equal(dataState, chip.GetAttribute("data-state"));
        Assert.Equal(label, chip.TextContent.Trim());
    }

    [Fact]
    public async Task Follows_state_changes()
    {
        IRenderedComponent<LiveFeedIndicator> indicator = Render<LiveFeedIndicator>();

        await _feed.SetStateAsync(LiveFeedState.Connected);
        indicator.WaitForAssertion(() => Assert.Equal("LIVE", indicator.Find(".solia-live-indicator").TextContent.Trim()));

        await _feed.SetStateAsync(LiveFeedState.Reconnecting);
        indicator.WaitForAssertion(() => Assert.Equal("RECONNECTING", indicator.Find(".solia-live-indicator").TextContent.Trim()));
    }

    [Fact]
    public async Task A_late_notification_does_not_override_the_current_state()
    {
        IRenderedComponent<LiveFeedIndicator> indicator = Render<LiveFeedIndicator>();
        await _feed.SetStateAsync(LiveFeedState.Connected);

        await _feed.NotifyAsync(LiveFeedState.Reconnecting);

        indicator.WaitForAssertion(() => Assert.Equal("LIVE", indicator.Find(".solia-live-indicator").TextContent.Trim()));
    }

    [Fact]
    public async Task Unsubscribes_when_disposed()
    {
        Render<LiveFeedIndicator>();
        Assert.Equal(1, _feed.StateSubscribers);

        await DisposeComponentsAsync();

        Assert.Equal(0, _feed.StateSubscribers);
        await _feed.SetStateAsync(LiveFeedState.Connected); // no render of a disposed component
    }

    [Fact]
    public void Keeps_the_host_classes()
    {
        IElement chip = Render<LiveFeedIndicator>(p => p.Add(c => c.Class, "ml-2")).Find(".solia-live-indicator");

        Assert.Contains("ml-2", chip.ClassList);
    }
}
