using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Solia.Api.Tests.Endpoints;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Solia.Core.Ai;
using Solia.Core.Http;

namespace Solia.Api.Tests;

/// <summary>
/// The probes against the real host with an unreachable (or malformed) database and a stubbed ai-service. The
/// database-up paths run against a real Postgres in <c>EndToEndTests</c>.
/// </summary>
public class HealthEndpointsTests
{
    private readonly StubAiClient _ai = new();

    private WebApplicationFactory<Program> Factory(params (string Key, string Value)[] settings) =>
        TestApiFactory.Create("Production", settings,
            builder => builder.ConfigureTestServices(services => services.AddSingleton<ISoliaAiClient>(_ai)));

    [Fact]
    public async Task Liveness_is_healthy_and_checks_nothing_while_every_dependency_is_down()
    {
        _ai.Health = () => throw StubAiClient.Unreachable();
        using WebApplicationFactory<Program> factory = Factory();

        HttpResponseMessage response = await factory.CreateClient().GetAsync(ApiRoutes.HealthLive);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument doc = await ReadAsync(response);
        Assert.Equal("Healthy", doc.RootElement.GetProperty("status").GetString());
        Assert.Empty(doc.RootElement.GetProperty("entries").EnumerateObject());
    }

    [Theory]
    [InlineData(ApiRoutes.Health)]
    [InlineData(ApiRoutes.HealthReady)]
    public async Task An_unreachable_database_is_a_503_with_the_health_json(string path)
    {
        using WebApplicationFactory<Program> factory = Factory();

        HttpResponseMessage response = await factory.CreateClient().GetAsync(path);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        Assert.True(response.Headers.CacheControl?.NoStore);
        using JsonDocument doc = await ReadAsync(response);
        JsonElement root = doc.RootElement;
        Assert.Equal("Unhealthy", root.GetProperty("status").GetString());
        Assert.True(root.TryGetProperty("total_duration", out _));
        JsonElement db = root.GetProperty("entries").GetProperty("database");
        Assert.Equal("Unhealthy", db.GetProperty("status").GetString());
        Assert.Equal("database unreachable", db.GetProperty("description").GetString());
        Assert.Equal(["ready"], db.GetProperty("tags").EnumerateArray().Select(t => t.GetString()));
        Assert.Equal("Healthy", root.GetProperty("entries").GetProperty("ai_service").GetProperty("status").GetString());
    }

    [Fact]
    public async Task A_malformed_db_connection_string_reports_unhealthy_instead_of_failing_the_probe()
    {
        using WebApplicationFactory<Program> factory = Factory(("SOLIA_DB_CONNECTION", "NotAKeyword=oops"));

        HttpResponseMessage response = await factory.CreateClient().GetAsync(ApiRoutes.HealthReady);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        using JsonDocument doc = JsonDocument.Parse(body);
        Assert.Equal("database unreachable",
            doc.RootElement.GetProperty("entries").GetProperty("database").GetProperty("description").GetString());
        Assert.DoesNotContain("NotAKeyword", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task An_unreachable_ai_service_is_unhealthy_without_leaking_the_upstream_message()
    {
        _ai.Health = () => throw StubAiClient.Unreachable();
        using WebApplicationFactory<Program> factory = Factory();

        HttpResponseMessage response = await factory.CreateClient().GetAsync(ApiRoutes.HealthReady);

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement ai = doc.RootElement.GetProperty("entries").GetProperty("ai_service");
        Assert.Equal("Unhealthy", ai.GetProperty("status").GetString());
        Assert.Equal("ai-service unreachable", ai.GetProperty("description").GetString());
        Assert.DoesNotContain("10.0.0.7", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task An_ai_service_reporting_a_status_other_than_ok_is_unhealthy()
    {
        _ai.Health = () => new AiHealthResponse("starting");
        using WebApplicationFactory<Program> factory = Factory();

        HttpResponseMessage response = await factory.CreateClient().GetAsync(ApiRoutes.HealthReady);

        using JsonDocument doc = await ReadAsync(response);
        Assert.Equal("Unhealthy",
            doc.RootElement.GetProperty("entries").GetProperty("ai_service").GetProperty("status").GetString());
    }

    [Fact]
    public async Task A_check_that_throws_is_reported_without_its_exception_message()
    {
        using WebApplicationFactory<Program> factory = TestApiFactory.Create("Production", [],
            builder => builder.ConfigureTestServices(services =>
            {
                services.AddSingleton<ISoliaAiClient>(_ai);
                services.AddHealthChecks().AddCheck("throws", () => throw new InvalidOperationException("Host=10.0.0.7;Password=hunter2"));
            }));

        HttpResponseMessage response = await factory.CreateClient().GetAsync(ApiRoutes.Health);

        string body = await response.Content.ReadAsStringAsync();
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement entry = doc.RootElement.GetProperty("entries").GetProperty("throws");
        Assert.Equal("Unhealthy", entry.GetProperty("status").GetString());
        Assert.Equal("check failed", entry.GetProperty("description").GetString());
        Assert.DoesNotContain("10.0.0.7", body, StringComparison.Ordinal);
        Assert.DoesNotContain("hunter2", body, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(ApiRoutes.HealthReady)]
    [InlineData(ApiRoutes.HealthLive)]
    public async Task Probes_answer_get_only_with_the_method_not_allowed_envelope(string path)
    {
        using WebApplicationFactory<Program> factory = Factory();

        HttpResponseMessage response = await factory.CreateClient().PostAsync(path, content: null);

        Assert.Equal(HttpStatusCode.MethodNotAllowed, response.StatusCode);
        using JsonDocument doc = await ReadAsync(response);
        Assert.Equal("method_not_allowed", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    private static async Task<JsonDocument> ReadAsync(HttpResponseMessage response) =>
        JsonDocument.Parse(await response.Content.ReadAsStringAsync());
}
