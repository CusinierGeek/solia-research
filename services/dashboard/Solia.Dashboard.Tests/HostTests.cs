using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Options;
using Solia.Dashboard.Configuration;

namespace Solia.Dashboard.Tests;

/// <summary>Boots the real dashboard host (no backend needed).</summary>
public class HostTests
{
    private static WebApplicationFactory<Program> Factory(string environment, params (string Key, string Value)[] settings) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment(environment);
            foreach ((string key, string value) in settings)
            {
                builder.UseSetting(key, value);
            }
        });

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
