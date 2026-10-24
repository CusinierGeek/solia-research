using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Solia.Core.Http;

namespace Solia.Api.Tests;

/// <summary>
/// Audits the whole public surface at once — every mapped operation, enumerated from the host's own routing
/// table so a new endpoint can't slip through: it is in the OpenAPI document with an id, a tag, a success
/// schema and the error envelope, and it fails with the one error shape. No database is needed.
/// </summary>
public class ApiSurfaceTests : IClassFixture<ApiSurfaceTests.DevelopmentHost>
{
    private readonly DevelopmentHost _host;

    public ApiSurfaceTests(DevelopmentHost host) => _host = host;

    [Fact]
    public void The_surface_is_the_documented_one()
    {
        string[] expected =
        [
            "GET /api/v1/logs/bot", "GET /api/v1/model/status", "GET /api/v1/predictions/history",
            "GET /api/v1/predictions/live", "GET /api/v1/stats/performance", "GET /api/v1/stats/winrate",
            "GET /api/v1/tokens/history", "GET /api/v1/tokens/live", "GET /api/v1/trades/history", "GET /health",
            "GET /health/live", "GET /health/ready", "POST /api/v1/model/retrain", "POST /api/v1/predict",
        ];

        Assert.Equal(expected, Operations().Select(o => $"{o.Method} {o.Path}").Order(StringComparer.Ordinal));
    }

    [Fact]
    public void Every_operation_is_documented_with_an_id_a_tag_and_a_success_schema()
    {
        foreach ((string method, string path) in Operations())
        {
            AssertDocumented(method, path);
        }
    }

    private void AssertDocumented(string method, string path)
    {
        JsonElement operation = _host.Operation(method, path);

        Assert.False(string.IsNullOrWhiteSpace(operation.GetProperty("operationId").GetString()));
        Assert.NotEmpty(operation.GetProperty("tags").EnumerateArray());
        Assert.False(string.IsNullOrWhiteSpace(operation.GetProperty("summary").GetString()));
        JsonElement success = operation.GetProperty("responses").EnumerateObject()
            .Single(r => r.Name is "200" or "202").Value;
        Assert.True(success.GetProperty("content").GetProperty("application/json").TryGetProperty("schema", out _));
    }

    [Fact]
    public void Every_api_operation_documents_the_500_error_envelope()
    {
        foreach ((string method, string path) in Operations().Where(o => o.Path.StartsWith("/api/v1", StringComparison.Ordinal)))
        {
            Assert.Equal("#/components/schemas/ApiErrorResponse", _host.ErrorSchemaRef(method, path, "500"));
        }
    }

    [Fact]
    public void The_error_envelope_schema_is_error_code_and_message()
    {
        JsonElement schemas = _host.Document.RootElement.GetProperty("components").GetProperty("schemas");
        JsonElement envelope = schemas.GetProperty("ApiErrorResponse");
        Assert.True(envelope.GetProperty("properties").TryGetProperty("error", out _));
        JsonElement error = schemas.GetProperty("ApiError").GetProperty("properties");
        Assert.True(error.TryGetProperty("code", out _));
        Assert.True(error.TryGetProperty("message", out _));
    }

    [Theory]
    [InlineData("POST", "/api/v1/predict", new[] { "400", "404", "415", "422", "502", "503" })]
    [InlineData("POST", "/api/v1/model/retrain", new[] { "400", "502", "503" })]
    [InlineData("GET", "/api/v1/tokens/live", new[] { "400" })]
    [InlineData("GET", "/api/v1/tokens/history", new[] { "400" })]
    [InlineData("GET", "/api/v1/trades/history", new[] { "400" })]
    [InlineData("GET", "/api/v1/predictions/live", new[] { "400" })]
    [InlineData("GET", "/api/v1/predictions/history", new[] { "400" })]
    [InlineData("GET", "/api/v1/stats/winrate", new[] { "400" })]
    [InlineData("GET", "/api/v1/stats/performance", new[] { "400" })]
    [InlineData("GET", "/api/v1/logs/bot", new[] { "400" })]
    public void Each_operation_documents_its_own_failures_with_the_envelope(string method, string path, string[] statuses)
    {
        foreach (string status in statuses)
        {
            Assert.Equal("#/components/schemas/ApiErrorResponse", _host.ErrorSchemaRef(method, path, status));
        }
    }

    [Fact]
    public void Predict_documents_its_request_body_and_response_fields()
    {
        JsonElement body = _host.Operation("POST", "/api/v1/predict").GetProperty("requestBody");
        Assert.True(body.GetProperty("required").GetBoolean());
        JsonElement request = _host.Resolve(body.GetProperty("content").GetProperty("application/json").GetProperty("schema"));
        Assert.True(request.GetProperty("properties").TryGetProperty("mint", out _));
        Assert.True(request.GetProperty("properties").TryGetProperty("horizons", out _));

        JsonElement response = _host.SuccessSchema("POST", "/api/v1/predict", "200");
        string[] fields = response.GetProperty("properties").EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(["mint", "model_version", "predictions"], fields);
    }

    [Theory]
    [InlineData("/api/v1/predictions/live")]
    [InlineData("/api/v1/predictions/history")]
    public void Prediction_reads_document_the_item_fields(string path)
    {
        JsonElement success = _host.SuccessSchema("GET", path, "200");
        JsonElement array = path.EndsWith("/live", StringComparison.Ordinal)
            ? success
            : _host.Resolve(success.GetProperty("properties").GetProperty("items"));
        JsonElement item = _host.Resolve(array.GetProperty("items"));
        string[] fields = item.GetProperty("properties").EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(
            ["confidence", "horizon", "id", "mint", "model_version", "position_size", "pump_prob", "rug_prob", "stop_loss",
             "strategy", "symbol", "take_profit", "ts"],
            fields);
    }

    [Fact]
    public void Retrain_documents_its_202_with_the_training_event_id()
    {
        JsonElement response = _host.SuccessSchema("POST", "/api/v1/model/retrain", "202");
        Assert.True(response.GetProperty("properties").TryGetProperty("training_event_id", out _));
    }

    [Theory]
    [InlineData(ApiRoutes.Health, new[] { "200", "503" })]
    [InlineData(ApiRoutes.HealthReady, new[] { "200", "503" })]
    [InlineData(ApiRoutes.HealthLive, new[] { "200" })]
    public void Probes_document_the_health_json_for_each_status_they_can_return(string path, string[] statuses)
    {
        string[] documented = _host.Operation("GET", path).GetProperty("responses").EnumerateObject().Select(r => r.Name).ToArray();
        Assert.Equal(statuses, documented.Where(s => s is "200" or "503").Order(StringComparer.Ordinal));

        foreach (string status in statuses)
        {
            JsonElement schema = _host.SuccessSchema("GET", path, status);
            string[] fields = schema.GetProperty("properties").EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal).ToArray();
            Assert.Equal(["entries", "status", "total_duration"], fields);
        }
    }

    [Fact]
    public async Task A_wrong_method_is_the_405_envelope_on_every_operation()
    {
        HttpClient client = _host.CreateClient();
        foreach ((string method, string path) in Operations())
        {
            HttpMethod wrong = method == "GET" ? HttpMethod.Delete : HttpMethod.Get;

            HttpResponseMessage response = await client.SendAsync(new HttpRequestMessage(wrong, path));

            await AssertEnvelopeAsync(response, HttpStatusCode.MethodNotAllowed, "method_not_allowed");
            Assert.Contains(method, response.Content.Headers.Allow);
        }
    }

    [Theory]
    [InlineData("/api/v1/tokens/does-not-exist")]
    [InlineData("/api/v1")]
    [InlineData("/health/unknown")]
    public async Task An_unknown_route_is_the_404_envelope(string path)
    {
        HttpResponseMessage response = await _host.CreateClient().GetAsync(path);

        await AssertEnvelopeAsync(response, HttpStatusCode.NotFound, "not_found");
    }

    [Fact]
    public async Task Predict_with_a_non_json_body_is_the_415_envelope()
    {
        using var content = new StringContent("mint=So1", Encoding.UTF8, "text/plain");

        HttpResponseMessage response = await _host.CreateClient().PostAsync("/api/v1/predict", content);

        await AssertEnvelopeAsync(response, HttpStatusCode.UnsupportedMediaType, "unsupported_media_type");
    }

    [Theory]
    [InlineData("/api/v1/tokens/live?limit=abc")]
    [InlineData("/api/v1/tokens/history?page=abc")]
    [InlineData("/api/v1/trades/history?from=yesterday")]
    [InlineData("/api/v1/stats/performance?limit=1.5")]
    [InlineData("/api/v1/logs/bot?limit=")]
    public async Task An_unbindable_query_value_is_the_400_envelope(string url)
    {
        HttpResponseMessage response = await _host.CreateClient().GetAsync(url);

        await AssertEnvelopeAsync(response, HttpStatusCode.BadRequest, "invalid_request");
    }

    private static async Task AssertEnvelopeAsync(HttpResponseMessage response, HttpStatusCode status, string code)
    {
        Assert.Equal(status, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        JsonElement error = doc.RootElement.GetProperty("error");
        Assert.Equal(code, error.GetProperty("code").GetString());
        Assert.False(string.IsNullOrWhiteSpace(error.GetProperty("message").GetString()));
    }

    /// <summary>
    /// The operations of the shared host — read from its routing table rather than as theory data, which would
    /// boot a second host during test discovery.
    /// </summary>
    private IEnumerable<(string Method, string Path)> Operations() => PublicOperations(_host.Services);

    /// <summary>Every public (method, path) the host maps: the versioned API and the probes.</summary>
    private static IEnumerable<(string Method, string Path)> PublicOperations(IServiceProvider services) =>
        services.GetRequiredService<EndpointDataSource>().Endpoints
            .OfType<RouteEndpoint>()
            .Select(e => (Endpoint: e, Path: "/" + e.RoutePattern.RawText!.Trim('/')))
            .Where(e => e.Path.StartsWith("/api/v1", StringComparison.Ordinal) || e.Path.StartsWith("/health", StringComparison.Ordinal))
            .SelectMany(e => e.Endpoint.Metadata.GetMetadata<IHttpMethodMetadata>()!.HttpMethods.Select(m => (m, e.Path)));

    /// <summary>The host in Development (where the OpenAPI document is served), with the document loaded once.</summary>
    public sealed class DevelopmentHost : IDisposable
    {
        private readonly WebApplicationFactory<Program> _factory = TestApiFactory.Create("Development", []);

        public DevelopmentHost() =>
            Document = JsonDocument.Parse(_factory.CreateClient().GetStringAsync("/openapi/v1.json").GetAwaiter().GetResult());

        public JsonDocument Document { get; }

        public IServiceProvider Services => _factory.Services;

        public HttpClient CreateClient() => _factory.CreateClient();

        public JsonElement Operation(string method, string path) =>
            Document.RootElement.GetProperty("paths").GetProperty(path).GetProperty(method.ToLowerInvariant());

        public string? ErrorSchemaRef(string method, string path, string status) =>
            Operation(method, path).GetProperty("responses").GetProperty(status)
                .GetProperty("content").GetProperty("application/json").GetProperty("schema").GetProperty("$ref").GetString();

        public JsonElement SuccessSchema(string method, string path, string status) =>
            Resolve(Operation(method, path).GetProperty("responses").GetProperty(status)
                .GetProperty("content").GetProperty("application/json").GetProperty("schema"));

        /// <summary>Follows a <c>#/components/schemas/…</c> reference (one level), else returns the inline schema.</summary>
        public JsonElement Resolve(JsonElement schema) =>
            schema.TryGetProperty("$ref", out JsonElement reference)
                ? Document.RootElement.GetProperty("components").GetProperty("schemas")
                    .GetProperty(reference.GetString()!["#/components/schemas/".Length..])
                : schema;

        public void Dispose()
        {
            Document.Dispose();
            _factory.Dispose();
        }
    }
}
