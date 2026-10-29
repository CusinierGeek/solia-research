using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Solia.Core.Ai;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// The AI-proxy endpoints' paths that never reach the database — body validation and the retrain proxy — with
/// <see cref="ISoliaAiClient"/> stubbed. The DB-backed paths (feature window, <c>/model/status</c> with its trade
/// count, including how it degrades when the ai-service fails) are in <see cref="AiProxyIntegrationTests"/>.
/// </summary>
public class AiProxyEndpointsTests
{
    private static (WebApplicationFactory<Program> Factory, HttpClient Client) Host(StubAiClient ai, string environment = "Production")
    {
        WebApplicationFactory<Program> factory = TestApiFactory.Create(environment, [],
            builder => builder.ConfigureTestServices(services => services.AddSingleton<ISoliaAiClient>(ai)));
        return (factory, factory.CreateClient());
    }

    private static StringContent Json(string body) => new(body, Encoding.UTF8, "application/json");

    private static async Task<JsonElement> ErrorAsync(HttpResponseMessage response)
    {
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return doc.RootElement.GetProperty("error").Clone();
    }

    public static TheoryData<string, string> InvalidPredictBodies() => new()
    {
        { "Production", """{ "horizons": ["1m"] }""" },
        { "Production", """{ "mint": "   " }""" },
        { "Production", $$"""{ "mint": "{{new string('x', 65)}}" }""" },
        { "Production", """{ "mint": "m1", "horizons": [] }""" },
        { "Production", """{ "mint": "m1", "horizons": ["1m", " "] }""" },
        { "Production", """{ "mint": "m1", "horizons": "1m" }""" },
        { "Production", """{ "mint": """ },
        { "Production", "" },
        { "Development", """{ "mint": """ },
    };

    [Theory]
    [MemberData(nameof(InvalidPredictBodies))]
    public async Task Invalid_predict_body_is_a_400_envelope_and_never_reaches_the_ai_service(string environment, string body)
    {
        var ai = new StubAiClient();
        (WebApplicationFactory<Program> factory, HttpClient client) = Host(ai, environment);
        using (factory)
        {
            HttpResponseMessage response = await client.PostAsync("/api/v1/predict", Json(body));

            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            Assert.Equal("invalid_request", (await ErrorAsync(response)).GetProperty("code").GetString());
            Assert.Equal(0, ai.Calls);
        }
    }

    [Fact]
    public async Task Retrain_returns_202_with_the_training_event_id()
    {
        var ai = new StubAiClient { Retrain = () => new AiRetrainResponse(1234) };
        (WebApplicationFactory<Program> factory, HttpClient client) = Host(ai);
        using (factory)
        {
            HttpResponseMessage response = await client.PostAsync("/api/v1/model/retrain", null);

            Assert.Equal(HttpStatusCode.Accepted, response.StatusCode);
            using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal(1234, doc.RootElement.GetProperty("training_event_id").GetInt64());
            Assert.Equal(1, ai.Calls);
        }
    }

    [Fact]
    public async Task Retrain_already_running_on_the_ai_service_is_a_409_conflict_envelope()
    {
        var ai = new StubAiClient
        {
            Retrain = () => throw new AiServiceException(
                AiServiceErrorKind.Busy, "a retrain is already running; try again later", 409, "retrain_in_progress"),
        };
        (WebApplicationFactory<Program> factory, HttpClient client) = Host(ai);
        using (factory)
        {
            HttpResponseMessage response = await client.PostAsync("/api/v1/model/retrain", null);

            Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
            JsonElement error = await ErrorAsync(response);
            Assert.Equal("conflict", error.GetProperty("code").GetString());
            Assert.Equal("a retrain is already running", error.GetProperty("message").GetString());
        }
    }

    [Fact]
    public async Task Ai_service_unavailable_on_retrain_is_a_503_envelope_without_upstream_details()
    {
        var ai = new StubAiClient { Retrain = () => throw StubAiClient.Unreachable() };
        (WebApplicationFactory<Program> factory, HttpClient client) = Host(ai);
        using (factory)
        {
            HttpResponseMessage response = await client.PostAsync("/api/v1/model/retrain", null);

            Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
            JsonElement error = await ErrorAsync(response);
            Assert.Equal("unavailable", error.GetProperty("code").GetString());
            Assert.Equal("the prediction service is unavailable", error.GetProperty("message").GetString());
            Assert.DoesNotContain("10.0.0.7", await response.Content.ReadAsStringAsync());
        }
    }

    [Fact]
    public async Task Ai_service_failure_on_retrain_is_a_502_internal_error_envelope()
    {
        var ai = new StubAiClient
        {
            Retrain = () => throw new AiServiceException(AiServiceErrorKind.UpstreamError, "Traceback: KeyError 'orchestrator'", 500),
        };
        (WebApplicationFactory<Program> factory, HttpClient client) = Host(ai);
        using (factory)
        {
            HttpResponseMessage response = await client.PostAsync("/api/v1/model/retrain", null);

            Assert.Equal(HttpStatusCode.BadGateway, response.StatusCode);
            Assert.Equal("internal_error", (await ErrorAsync(response)).GetProperty("code").GetString());
            Assert.DoesNotContain("Traceback", await response.Content.ReadAsStringAsync());
        }
    }
}
