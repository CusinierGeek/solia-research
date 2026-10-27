using Microsoft.Extensions.Options;
using Solia.Dashboard.Configuration;

namespace Solia.Dashboard.Live;

/// <summary>Registers the app-wide <see cref="ILiveFeed"/>.</summary>
public static class LiveFeedServiceCollectionExtensions
{
    /// <summary>
    /// Registers one <see cref="ILiveFeed"/> for the whole app (see <see cref="LiveFeed"/> for why not one per
    /// circuit) on <c>{Backend:BaseUrl}/hubs/live</c>, started and stopped with the host. The connection is its own
    /// <see cref="ILiveConnection"/> service so host tests can swap it for one that never touches the network. Needs
    /// <see cref="DashboardServiceCollectionExtensions.AddDashboardOptions"/>.
    /// </summary>
    public static IServiceCollection AddLiveFeed(this IServiceCollection services)
    {
        services.AddSingleton<ILiveConnection>(sp =>
        {
            BackendOptions options = sp.GetRequiredService<IOptions<BackendOptions>>().Value;
            return HubLiveConnection.Create(HubUrl(options.BaseUrl), LiveFeedRetryPolicy.Instance, sp.GetRequiredService<ILoggerFactory>());
        });
        services.AddSingleton(sp => new LiveFeed(
            sp.GetRequiredService<ILiveConnection>(), LiveFeedRetryPolicy.Instance, sp.GetRequiredService<ILogger<LiveFeed>>()));
        services.AddSingleton<ILiveFeed>(sp => sp.GetRequiredService<LiveFeed>());
        services.AddHostedService(sp => sp.GetRequiredService<LiveFeed>());
        return services;
    }

    internal static Uri HubUrl(string baseUrl) => BackendOptions.Resolve(baseUrl, LiveFeedEvents.HubPath);
}
