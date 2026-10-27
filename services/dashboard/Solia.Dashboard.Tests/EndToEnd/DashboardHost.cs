using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Solia.Dashboard.Api;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.EndToEnd;

/// <summary>
/// The real dashboard host (its own DI, options, typed REST client with its resilience pipeline, and app-wide
/// SignalR live feed) bound to a <see cref="StubBackend"/>. Only the transport is swapped — both the REST client and
/// the hub connection reach the stub through its in-memory handler — plus a fast hub reconnect so an outage test
/// recovers in a fraction of a second rather than the production backoff's seconds.
/// </summary>
internal sealed class DashboardHost : IAsyncDisposable
{
    /// <summary>The hub's reconnect delay in these tests (production: 0 s, 2 s, 5 s, 10 s, then 30 s).</summary>
    public static readonly TimeSpan ReconnectDelay = TimeSpan.FromMilliseconds(100);

    private readonly WebApplicationFactory<Program> _factory;

    private DashboardHost(WebApplicationFactory<Program> factory) => _factory = factory;

    public IServiceProvider Services => _factory.Services;

    public ILiveFeed LiveFeed => Services.GetRequiredService<ILiveFeed>();

    /// <summary>An HTTP client to the dashboard itself (prerendered pages, health).</summary>
    public HttpClient CreateClient() => _factory.CreateClient();

    /// <summary>Boots the host and waits for its live feed to connect to the stub's hub.</summary>
    public static async Task<DashboardHost> StartAsync(StubBackend backend, TimeSpan timeout)
    {
        WebApplicationFactory<Program> factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment("Production");
            builder.UseSetting("BACKEND_URL", StubBackend.BaseUrl);
            builder.ConfigureTestServices(services =>
            {
                services.AddHttpClient<ISoliaApiClient, SoliaApiClient>().ConfigurePrimaryHttpMessageHandler(backend.CreateHandler);
                services.AddSingleton<ILiveConnection>(sp => HubLiveConnection.Create(
                    LiveFeedServiceCollectionExtensions.HubUrl(StubBackend.BaseUrl),
                    new FixedRetryPolicy(ReconnectDelay),
                    sp.GetRequiredService<ILoggerFactory>(),
                    http =>
                    {
                        http.HttpMessageHandlerFactory = _ => backend.CreateHandler();
                        http.Transports = HttpTransportType.LongPolling;
                    }));
            });
        });

        DashboardHost host = new(factory);
        await host.WaitForFeedAsync(LiveFeedState.Connected, timeout);
        return host;
    }

    public async Task WaitForFeedAsync(LiveFeedState state, TimeSpan timeout)
    {
        using CancellationTokenSource cts = new(timeout);
        while (LiveFeed.State != state)
        {
            await Task.Delay(20, cts.Token);
        }
    }

    public ValueTask DisposeAsync() => _factory.DisposeAsync();

    private sealed class FixedRetryPolicy(TimeSpan delay) : IRetryPolicy
    {
        public TimeSpan? NextRetryDelay(RetryContext retryContext) => delay;
    }
}
