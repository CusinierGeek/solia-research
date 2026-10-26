using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Options;

namespace Solia.Api.Tests;

/// <summary>Boots the real host. No database is needed (see <see cref="TestApiFactory"/>).</summary>
public class HostSmokeTests
{
    private static WebApplicationFactory<Program> Factory(string environment, params (string Key, string Value)[] settings) =>
        TestApiFactory.Create(environment, settings);

    [Fact]
    public async Task Backend_url_supplied_by_a_late_config_source_reaches_the_openapi_document()
    {
        using WebApplicationFactory<Program> factory = Factory("Development", ("BACKEND_URL", "https://from-env.solia.example"));
        HttpClient client = factory.CreateClient();

        using JsonDocument doc = JsonDocument.Parse(await client.GetStringAsync("/openapi/v1.json"));

        string[] servers = doc.RootElement.GetProperty("servers").EnumerateArray()
            .Select(s => s.GetProperty("url").GetString()!).ToArray();
        // The serving host first (what the Swagger UI uses), then the configured base URL.
        Assert.Equal(["/", "https://from-env.solia.example"], servers);
    }

    [Fact]
    public async Task Malformed_db_connection_string_does_not_stop_the_host()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("SOLIA_DB_CONNECTION", "NotAKeyword=oops"));
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/health/live");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Wrong_method_returns_the_method_not_allowed_envelope_with_its_allow_header()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.PostAsync("/health", content: null);

        Assert.Equal(HttpStatusCode.MethodNotAllowed, response.StatusCode);
        Assert.Contains("GET", response.Content.Headers.Allow);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("method_not_allowed", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task Liveness_returns_200_without_any_dependency()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/health/live");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task OpenApi_document_is_served_in_development_with_the_configured_server()
    {
        using WebApplicationFactory<Program> factory = Factory("Development", ("Api:BaseUrl", "https://api.solia.example"));
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/openapi/v1.json");

        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        Assert.Contains("Solia backend API", json);
        Assert.Contains("https://api.solia.example", json);
    }

    [Fact]
    public async Task Unknown_route_returns_the_not_found_error_envelope()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/api/v1/does-not-exist");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("not_found", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task OpenApi_document_is_not_served_outside_development()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/openapi/v1.json");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task Root_redirects_to_the_swagger_ui_in_development()
    {
        using WebApplicationFactory<Program> factory = Factory("Development");
        HttpClient client = factory.CreateClient(new WebApplicationFactoryClientOptions { AllowAutoRedirect = false });

        HttpResponseMessage response = await client.GetAsync("/");

        Assert.Equal(HttpStatusCode.Redirect, response.StatusCode);
        Assert.Equal("/swagger", response.Headers.Location?.OriginalString);
    }

    [Fact]
    public async Task Root_is_the_not_found_envelope_outside_development()
    {
        using WebApplicationFactory<Program> factory = Factory("Production");
        HttpClient client = factory.CreateClient(new WebApplicationFactoryClientOptions { AllowAutoRedirect = false });

        HttpResponseMessage response = await client.GetAsync("/");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("not_found", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public void Invalid_base_url_fails_at_startup()
    {
        using WebApplicationFactory<Program> factory = Factory("Production", ("Api:BaseUrl", "not-a-url"));

        var ex = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
        Assert.Contains("BaseUrl", ex.Message);
    }
}
