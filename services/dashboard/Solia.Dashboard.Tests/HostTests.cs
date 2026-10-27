using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Solia.Dashboard.Configuration;
using Solia.Dashboard.Live;
using Solia.Dashboard.Tests.Live;

namespace Solia.Dashboard.Tests;

/// <summary>Boots the real dashboard host (no backend needed: the live feed runs on a fake connection).</summary>
public class HostTests
{
    private static WebApplicationFactory<Program> Factory(string environment, params (string Key, string Value)[] settings) =>
        Factory(new FakeLiveConnection(), environment, settings);

    private static WebApplicationFactory<Program> Factory(
        FakeLiveConnection connection, string environment, params (string Key, string Value)[] settings) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment(environment);
            foreach ((string key, string value) in settings)
            {
                builder.UseSetting(key, value);
            }

            builder.ConfigureTestServices(services => services.AddSingleton<ILiveConnection>(connection));
        });

    [Fact]
    public async Task The_live_feed_is_one_app_wide_hosted_singleton_started_with_the_host()
    {
        FakeLiveConnection connection = new();
        using WebApplicationFactory<Program> factory = Factory(connection, "Production", ("BACKEND_URL", "http://127.0.0.1:1"));
        IServiceProvider services = factory.Services;

        LiveFeed feed = services.GetRequiredService<LiveFeed>();

        Assert.Same(feed, services.GetRequiredService<ILiveFeed>());
        Assert.Contains(feed, services.GetServices<IHostedService>());
        using CancellationTokenSource timeout = new(TimeSpan.FromSeconds(5));
        while (feed.State != LiveFeedState.Connected)
        {
            await Task.Delay(10, timeout.Token);
        }

        Assert.Equal(1, connection.StartCalls);
    }

    [Fact]
    public async Task Health_live_is_ok_without_a_reachable_backend()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("BACKEND_URL", "http://127.0.0.1:1"));

        HttpResponseMessage response = await factory.CreateClient().GetAsync("/health/live");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Home_page_prerenders_the_shell_with_the_paper_trading_indicator()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("BACKEND_URL", "http://127.0.0.1:1"));

        string html = await factory.CreateClient().GetStringAsync("/");

        Assert.Contains("PAPER TRADING", html);
        Assert.Contains("Overview", html);
    }

    [Fact]
    public async Task Pages_are_interactive_but_the_error_page_is_rendered_statically()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("BACKEND_URL", "http://127.0.0.1:1"));
        HttpClient client = factory.CreateClient();

        string home = await client.GetStringAsync("/");
        string error = await client.GetStringAsync("/Error");

        Assert.Contains("\"type\":\"server\"", home);
        Assert.Contains("An error occurred while processing your request.", error);
        Assert.Contains("Request ID", error);
        Assert.DoesNotContain("\"type\":\"server\"", error);
    }

    [Fact]
    public async Task An_unknown_route_answers_404_with_the_not_found_page()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("BACKEND_URL", "http://127.0.0.1:1"));

        HttpResponseMessage response = await factory.CreateClient().GetAsync("/no-such-view");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        string html = await response.Content.ReadAsStringAsync();
        Assert.Contains("the page you are looking for does not exist", html);
        Assert.Contains("PAPER TRADING", html);
    }

    [Fact]
    public void Startup_fails_fast_without_a_backend_url()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");

        OptionsValidationException ex = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());

        Assert.Contains(DashboardServiceCollectionExtensions.InvalidBaseUrlMessage, ex.Failures);
    }

    [Fact]
    public void Startup_fails_fast_with_an_invalid_backend_url()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("Backend:BaseUrl", "backend:8080"));

        Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
    }
}
