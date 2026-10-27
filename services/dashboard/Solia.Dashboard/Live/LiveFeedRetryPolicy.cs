using Microsoft.AspNetCore.SignalR.Client;

namespace Solia.Dashboard.Live;

/// <summary>
/// The feed's backoff: 0 s, 2 s, 5 s, 10 s, then every 30 s — and it never gives up (SignalR's default stops after
/// four tries, which would leave the dashboard offline until restarted). Used both for automatic reconnect and for
/// retrying a start that failed.
/// </summary>
internal sealed class LiveFeedRetryPolicy : IRetryPolicy
{
    public static readonly LiveFeedRetryPolicy Instance = new();

    internal static readonly TimeSpan[] Delays =
        [TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(30)];

    public TimeSpan? NextRetryDelay(RetryContext retryContext) => Delay(retryContext.PreviousRetryCount);

    public static TimeSpan Delay(long previousRetryCount) =>
        Delays[(int)Math.Clamp(previousRetryCount, 0, Delays.Length - 1)];
}
