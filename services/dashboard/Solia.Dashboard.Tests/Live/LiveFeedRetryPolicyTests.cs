using Microsoft.AspNetCore.SignalR.Client;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

public class LiveFeedRetryPolicyTests
{
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 2)]
    [InlineData(2, 5)]
    [InlineData(3, 10)]
    [InlineData(4, 30)]
    [InlineData(1_000, 30)]
    [InlineData(long.MaxValue, 30)]
    public void Backs_off_then_retries_every_30_seconds_forever(long previousRetries, int expectedSeconds)
    {
        TimeSpan? delay = LiveFeedRetryPolicy.Instance.NextRetryDelay(new RetryContext { PreviousRetryCount = previousRetries });

        Assert.Equal(TimeSpan.FromSeconds(expectedSeconds), delay);
    }

    [Theory]
    [InlineData("http://backend:8080", "http://backend:8080/hubs/live")]
    [InlineData("http://backend:8080/", "http://backend:8080/hubs/live")]
    [InlineData("https://host/solia", "https://host/solia/hubs/live")]
    [InlineData("https://host/solia/", "https://host/solia/hubs/live")]
    public void Hub_url_is_under_the_backend_base_url(string baseUrl, string expected)
    {
        Assert.Equal(new Uri(expected), LiveFeedServiceCollectionExtensions.HubUrl(baseUrl));
    }
}
