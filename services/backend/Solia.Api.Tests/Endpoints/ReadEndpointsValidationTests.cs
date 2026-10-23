using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// Invalid query parameters are rejected before any query runs, so these need no database: each returns a
/// 400 with the documented error envelope — in Production (where the framework answers a bind failure
/// itself) and in Development (where it throws one).
/// </summary>
public class ReadEndpointsValidationTests
{
    public static TheoryData<string, string> InvalidRequests() => new()
    {
        { "Production", "/api/v1/tokens/history?outcome=moon" },
        { "Production", "/api/v1/tokens/history?from=2026-01-02T00:00:00Z&to=2026-01-01T00:00:00Z" },
        { "Production", "/api/v1/tokens/history?page=0" },
        { "Production", "/api/v1/tokens/history?pageSize=201" },
        { "Production", "/api/v1/tokens/history?pageSize=abc" },
        { "Development", "/api/v1/tokens/history?pageSize=abc" },
        { "Production", "/api/v1/tokens/history?from=not-a-date" },
        { "Production", "/api/v1/tokens/live?sort=price_asc" },
        { "Production", "/api/v1/tokens/live?limit=0" },
        { "Production", "/api/v1/tokens/live?limit=501" },
        { "Production", "/api/v1/trades/history?status=pending" },
        { "Production", "/api/v1/trades/history?from=2026-01-02T00:00:00Z&to=2026-01-01T00:00:00Z" },
        { "Production", "/api/v1/trades/history?page=-3" },
        { "Production", "/api/v1/trades/history?page=51&pageSize=200" },
        { "Production", "/api/v1/tokens/history?page=1000000&pageSize=200" },
        { "Production", "/api/v1/stats/winrate?window=1y" },
        { "Production", "/api/v1/stats/performance?limit=0" },
        { "Production", "/api/v1/stats/performance?limit=10001" },
        { "Production", "/api/v1/logs/bot?level=loud" },
        { "Production", "/api/v1/logs/bot?service=nope" },
        { "Production", "/api/v1/logs/bot?limit=1001" },
    };

    [Theory]
    [MemberData(nameof(InvalidRequests))]
    public async Task Invalid_query_returns_400_with_the_error_envelope(string environment, string url)
    {
        using WebApplicationFactory<Program> factory = TestApiFactory.Create(environment, []);
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync(url);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        JsonElement error = doc.RootElement.GetProperty("error");
        Assert.Equal("invalid_request", error.GetProperty("code").GetString());
        Assert.False(string.IsNullOrWhiteSpace(error.GetProperty("message").GetString()));
    }

    [Fact]
    public async Task Validation_message_names_the_offending_parameter()
    {
        using WebApplicationFactory<Program> factory = TestApiFactory.Create("Production", []);
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/api/v1/trades/history?status=pending");

        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("status must be one of: open, closed.",
            doc.RootElement.GetProperty("error").GetProperty("message").GetString());
    }

    [Fact]
    public async Task Window_choices_are_listed_in_documented_order()
    {
        using WebApplicationFactory<Program> factory = TestApiFactory.Create("Production", []);
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/api/v1/stats/winrate?window=1y");

        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("window must be one of: all, 24h, 7d, 30d.",
            doc.RootElement.GetProperty("error").GetProperty("message").GetString());
    }
}
