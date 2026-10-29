using System.Net;
using System.Text.Json.Nodes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Polly.CircuitBreaker;
using Polly.Timeout;
using Solia.Core.Ai;
using Solia.Core.Configuration;
using Solia.Core.Http;
using static Solia.Core.Tests.Ai.StubHandler;

namespace Solia.Core.Tests.Ai;

public class SoliaAiClientTests
{
    private static readonly Dictionary<string, string?> NoRetries = new() { ["AiService:MaxRetryAttempts"] = "0" };

    // ------------------------------------------------------------------ success paths

    [Fact]
    public async Task Predict_posts_snake_case_body_under_the_base_path_and_returns_predictions()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, AiClientTestHost.PredictJson));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        AiPredictResponse response = await provider.Client().PredictAsync(AiClientTestHost.SampleRequest());

        Assert.Equal("v3", response.ModelVersion);
        Assert.Equal(0.72, Assert.Single(response.Predictions).PumpProb);
        RecordedRequest sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Post, sent.Method);
        Assert.Equal("http://ai-service:8000/root/predict", sent.Uri!.ToString());
        JsonNode body = JsonNode.Parse(sent.Body!)!;
        Assert.Equal("So1abc", (string?)body["mint"]);
        Assert.Equal(1.5, (double?)body["features"]!["price"]);
    }

    [Fact]
    public async Task Model_status_gets_the_status_route()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK,
            """{"champion_version":"v7","created_at":null,"param_count":10,"metrics":{"auc":0.6}}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        AiModelStatus status = await provider.Client().GetModelStatusAsync();

        Assert.Equal("v7", status.ChampionVersion);
        Assert.Equal(10, status.ParamCount);
        RecordedRequest sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Get, sent.Method);
        Assert.Equal("http://ai-service:8000/root/model/status", sent.Uri!.ToString());
    }

    [Fact]
    public async Task Health_gets_the_health_route()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, """{"status":"ok"}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        AiHealthResponse health = await provider.Client().CheckHealthAsync();

        Assert.True(health.IsOk);
        RecordedRequest sent = Assert.Single(handler.Requests);
        Assert.Equal(HttpMethod.Get, sent.Method);
        Assert.Equal("http://ai-service:8000/root/health", sent.Uri!.ToString());
    }

    [Fact]
    public async Task Health_reporting_another_status_is_not_ok()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, """{"status":"starting"}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        Assert.False((await provider.Client().CheckHealthAsync()).IsOk);
    }

    [Fact]
    public async Task Health_of_an_unreachable_service_is_an_unavailable_failure()
    {
        var handler = new StubHandler(Throws(new HttpRequestException("connection refused")));
        using ServiceProvider provider = AiClientTestHost.Build(handler, NoRetries);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().CheckHealthAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
    }

    [Fact]
    public async Task Health_is_a_single_attempt_even_when_retries_are_configured()
    {
        var handler = new StubHandler(Throws(new HttpRequestException("connection refused")));
        using ServiceProvider provider = AiClientTestHost.Build(handler); // MaxRetryAttempts defaults to 2

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().CheckHealthAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task Failing_health_probes_never_open_a_circuit()
    {
        var handler = new StubHandler(
            Error(HttpStatusCode.BadGateway, "bad_gateway"),
            Error(HttpStatusCode.BadGateway, "bad_gateway"),
            Error(HttpStatusCode.BadGateway, "bad_gateway"),
            Error(HttpStatusCode.BadGateway, "bad_gateway"),
            Json(HttpStatusCode.OK, """{"status":"ok"}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:CircuitBreakerMinimumThroughput"] = "2",
        });
        ISoliaAiClient client = provider.Client();

        for (int i = 0; i < 4; i++)
        {
            var ex = await Assert.ThrowsAsync<AiServiceException>(() => client.CheckHealthAsync());
            Assert.IsNotAssignableFrom<BrokenCircuitException>(ex.InnerException); // never "circuit is open"
        }

        Assert.True((await client.CheckHealthAsync()).IsOk); // recovery is seen on the very next probe
        Assert.Equal(5, handler.Calls);
    }

    [Fact]
    public async Task A_hung_health_probe_ends_at_the_attempt_timeout()
    {
        var handler = new StubHandler(Hangs());
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:AttemptTimeout"] = "00:00:00.200",
        });

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().CheckHealthAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task Retrain_posts_a_manual_trigger_and_reads_the_202_training_event_id()
    {
        var handler = new StubHandler(Json(HttpStatusCode.Accepted, """{"training_event_id":17}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        AiRetrainResponse response = await provider.Client().TriggerRetrainAsync();

        Assert.Equal(17, response.TrainingEventId);
        RecordedRequest sent = Assert.Single(handler.Requests);
        Assert.Equal("http://ai-service:8000/root/model/retrain", sent.Uri!.ToString());
        Assert.Equal("""{"trigger":"manual"}""", sent.Body);
    }

    // ------------------------------------------------------------------ retries

    [Fact]
    public async Task Transient_failure_is_retried_then_succeeds()
    {
        var handler = new StubHandler(
            Error(HttpStatusCode.ServiceUnavailable, "champion_unavailable"),
            Throws(new HttpRequestException("connection refused")),
            Json(HttpStatusCode.OK, AiClientTestHost.PredictJson));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        AiPredictResponse response = await provider.Client().PredictAsync(AiClientTestHost.SampleRequest());

        Assert.Equal("v3", response.ModelVersion);
        Assert.Equal(3, handler.Calls);
        Assert.All(handler.Requests, r => Assert.NotNull(r.Body)); // the body is re-sent on every attempt
    }

    [Fact]
    public async Task Retries_exhausted_surface_the_last_error_as_a_typed_exception()
    {
        var handler = new StubHandler(Error(HttpStatusCode.GatewayTimeout, "gateway_timeout", "kaboom"));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(3, handler.Calls); // 1 attempt + MaxRetryAttempts (2)
        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Equal(504, ex.UpstreamStatusCode);
        Assert.Equal("gateway_timeout", ex.UpstreamCode);
        Assert.Equal("kaboom", ex.UpstreamMessage);
    }

    [Theory]
    [InlineData(HttpStatusCode.RequestTimeout)]
    [InlineData(HttpStatusCode.TooManyRequests)]
    [InlineData(HttpStatusCode.BadGateway)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    [InlineData(HttpStatusCode.GatewayTimeout)]
    public async Task Availability_statuses_are_retried(HttpStatusCode status)
    {
        var handler = new StubHandler(Error(status, "whatever"), Json(HttpStatusCode.OK, AiClientTestHost.PredictJson));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        await provider.Client().PredictAsync(AiClientTestHost.SampleRequest());

        Assert.Equal(2, handler.Calls);
    }

    [Fact]
    public async Task Internal_error_is_deterministic_and_not_retried()
    {
        var handler = new StubHandler(Error(HttpStatusCode.InternalServerError, "internal_error", "model bug"));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(1, handler.Calls);
        Assert.Equal(AiServiceErrorKind.UpstreamError, ex.Kind);
    }

    [Theory]
    [InlineData("no_champion")]
    [InlineData("champion_incompatible")]
    public async Task No_champion_is_neither_retried_nor_counted_by_the_breaker(string code)
    {
        var handler = new StubHandler(Error(HttpStatusCode.ServiceUnavailable, code));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:CircuitBreakerMinimumThroughput"] = "2",
        });
        ISoliaAiClient client = provider.Client();

        for (int i = 0; i < 4; i++)
        {
            var ex = await Assert.ThrowsAsync<AiServiceException>(() => client.PredictAsync(AiClientTestHost.SampleRequest()));
            Assert.Equal(AiServiceErrorKind.NoChampion, ex.Kind); // never "circuit is open"
        }

        Assert.Equal(4, handler.Calls); // one attempt per call
    }

    [Fact]
    public async Task Unreachable_service_after_retries_is_unavailable_not_a_raw_HttpRequestException()
    {
        var handler = new StubHandler(Throws(new HttpRequestException("no route to host")));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(3, handler.Calls);
        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Null(ex.UpstreamStatusCode);
        Assert.IsType<HttpRequestException>(ex.InnerException);
        Assert.Equal(503, ex.StatusCode);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, "invalid_features")]
    [InlineData(HttpStatusCode.BadRequest, "invalid_request")]
    [InlineData(HttpStatusCode.UnprocessableEntity, "invalid_request")]
    public async Task Invalid_request_is_never_retried(HttpStatusCode status, string code)
    {
        var handler = new StubHandler(Error(status, code));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task Retrain_is_not_retried_because_it_is_not_idempotent()
    {
        var handler = new StubHandler(Throws(new HttpRequestException("reset")), Json(HttpStatusCode.Accepted, """{"training_event_id":1}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().TriggerRetrainAsync());

        Assert.Equal(1, handler.Calls);
        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
    }

    [Fact]
    public async Task Retrain_already_running_is_a_busy_conflict_and_never_retried()
    {
        var handler = new StubHandler(
            Error(HttpStatusCode.Conflict, "retrain_in_progress", "a retrain is already running; try again later"));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().TriggerRetrainAsync());

        Assert.Equal(1, handler.Calls);
        Assert.Equal(AiServiceErrorKind.Busy, ex.Kind);
        Assert.Equal(409, ex.StatusCode);
        Assert.Equal("retrain_in_progress", ex.UpstreamCode);
    }

    // ------------------------------------------------------------------ error envelopes

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, "invalid_features", AiServiceErrorKind.InvalidRequest, 400, ApiErrorCodes.InvalidRequest)]
    [InlineData(HttpStatusCode.BadRequest, "invalid_request", AiServiceErrorKind.InvalidRequest, 400, ApiErrorCodes.InvalidRequest)]
    [InlineData(HttpStatusCode.UnprocessableEntity, "invalid_request", AiServiceErrorKind.InvalidRequest, 400, ApiErrorCodes.InvalidRequest)]
    [InlineData(HttpStatusCode.UnprocessableEntity, "window_too_short", AiServiceErrorKind.InsufficientHistory, 422, ApiErrorCodes.InsufficientHistory)]
    [InlineData(HttpStatusCode.ServiceUnavailable, "no_champion", AiServiceErrorKind.NoChampion, 503, ApiErrorCodes.Unavailable)]
    [InlineData(HttpStatusCode.ServiceUnavailable, "champion_incompatible", AiServiceErrorKind.NoChampion, 503, ApiErrorCodes.Unavailable)]
    [InlineData(HttpStatusCode.ServiceUnavailable, "champion_unavailable", AiServiceErrorKind.Unavailable, 503, ApiErrorCodes.Unavailable)]
    [InlineData(HttpStatusCode.Conflict, "retrain_in_progress", AiServiceErrorKind.Busy, 409, ApiErrorCodes.Conflict)]
    [InlineData(HttpStatusCode.ServiceUnavailable, "retrain_unavailable", AiServiceErrorKind.Unavailable, 503, ApiErrorCodes.Unavailable)]
    [InlineData(HttpStatusCode.InternalServerError, "internal_error", AiServiceErrorKind.UpstreamError, 502, ApiErrorCodes.InternalError)]
    [InlineData(HttpStatusCode.NotFound, "http_error", AiServiceErrorKind.UpstreamError, 502, ApiErrorCodes.InternalError)]
    public async Task Error_envelope_maps_to_a_typed_exception_and_the_public_error_shape(
        HttpStatusCode status, string code, AiServiceErrorKind kind, int publicStatus, string publicCode)
    {
        var handler = new StubHandler(Error(status, code, "upstream says no"));
        using ServiceProvider provider = AiClientTestHost.Build(handler, NoRetries);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(kind, ex.Kind);
        Assert.Equal((int)status, ex.UpstreamStatusCode);
        Assert.Equal(code, ex.UpstreamCode);
        Assert.Equal("upstream says no", ex.UpstreamMessage);
        Assert.Equal(publicStatus, ex.StatusCode);
        Assert.Equal(publicCode, ex.Code);
        Assert.DoesNotContain("upstream says no", ex.Message);
        Assert.IsAssignableFrom<ApiException>(ex); // the error-shape middleware renders it as-is
    }

    [Theory]
    [InlineData(AiServiceErrorKind.InvalidRequest)]
    [InlineData(AiServiceErrorKind.InsufficientHistory)]
    [InlineData(AiServiceErrorKind.NoChampion)]
    [InlineData(AiServiceErrorKind.Unavailable)]
    [InlineData(AiServiceErrorKind.UpstreamError)]
    public void The_upstream_message_is_never_exposed_publicly(AiServiceErrorKind kind)
    {
        var ex = new AiServiceException(kind, "secret: can't multiply sequence by non-int of type 'float'");

        Assert.DoesNotContain("secret", ex.Message);
        Assert.StartsWith("secret", ex.UpstreamMessage); // still available for logging
    }

    [Fact]
    public async Task Unserializable_request_is_an_invalid_request_and_never_sent()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, AiClientTestHost.PredictJson));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:CircuitBreakerMinimumThroughput"] = "2",
        });
        AiPredictRequest request = AiPredictRequest.ForVector("So1abc", new Dictionary<string, double?> { ["price"] = double.NaN });

        for (int i = 0; i < 3; i++)
        {
            var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(request));
            Assert.Equal(AiServiceErrorKind.InvalidRequest, ex.Kind);
            Assert.Equal(400, ex.StatusCode);
        }

        Assert.Equal(0, handler.Calls);
        await provider.Client().PredictAsync(AiClientTestHost.SampleRequest()); // the breaker did not open
    }

    [Fact]
    public async Task Non_envelope_error_body_is_classified_by_status()
    {
        var handler = new StubHandler(Json(HttpStatusCode.BadGateway, "<html>bad gateway</html>"));
        using ServiceProvider provider = AiClientTestHost.Build(handler, NoRetries);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Null(ex.UpstreamCode);
        Assert.Equal(502, ex.UpstreamStatusCode);
    }

    [Fact]
    public async Task Unreadable_success_body_is_an_upstream_error()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, "not json"));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(AiServiceErrorKind.UpstreamError, ex.Kind);
        Assert.Equal(1, handler.Calls);
    }

    [Theory]
    [InlineData("""{"mint":"x","predictions":[]}""")]
    [InlineData("""{"mint":"x","model_version":null,"predictions":[]}""")]
    [InlineData("""{"mint":"x","model_version":"v3"}""")]
    [InlineData("""{"mint":"x","model_version":"v3","predictions":[{"horizon":"1m","pump_prob":0.5}]}""")]
    public async Task Success_body_missing_a_required_field_is_an_upstream_error(string json)
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, json));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(AiServiceErrorKind.UpstreamError, ex.Kind);
    }

    [Fact]
    public async Task Status_without_champion_version_is_an_upstream_error()
    {
        var handler = new StubHandler(Json(HttpStatusCode.OK, """{"champion_version":null,"metrics":{}}"""));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(AiServiceErrorKind.UpstreamError, ex.Kind);
    }

    [Fact]
    public async Task Success_body_with_an_invalid_charset_is_an_upstream_error()
    {
        var handler = new StubHandler(JsonWithCharset(AiClientTestHost.PredictJson, "foo"));
        using ServiceProvider provider = AiClientTestHost.Build(handler);

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(AiServiceErrorKind.UpstreamError, ex.Kind);
    }

    // ------------------------------------------------------------------ timeouts / breaker / cancellation

    [Fact]
    public async Task Attempt_timeout_is_retried_and_then_surfaces_as_unavailable()
    {
        var handler = new StubHandler(Hangs());
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:AttemptTimeout"] = "00:00:00.050",
            ["AiService:MaxRetryAttempts"] = "1",
        });

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.IsType<TimeoutRejectedException>(ex.InnerException);
        Assert.Equal(2, handler.Calls);
    }

    [Fact]
    public async Task Retrain_waits_past_the_attempt_timeout_up_to_the_retrain_timeout()
    {
        // The ai-service answers once the run is recorded, which may take longer than an ordinary call.
        var handler = new StubHandler(async (request, ct) =>
        {
            await Task.Delay(TimeSpan.FromMilliseconds(300), ct);
            return await Json(HttpStatusCode.Accepted, """{"training_event_id":5}""")(request, ct);
        });
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:AttemptTimeout"] = "00:00:00.050",
            ["AiService:TotalTimeout"] = "00:00:00.200",
            ["AiService:MaxRetryAttempts"] = "0",
            ["AiService:RetrainTimeout"] = "00:00:05",
        });

        AiRetrainResponse response = await provider.Client().TriggerRetrainAsync();

        Assert.Equal(5, response.TrainingEventId);
    }

    [Fact]
    public async Task Retrain_timeout_bounds_the_retrain_call_without_a_retry()
    {
        var handler = new StubHandler(Hangs());
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:RetrainTimeout"] = "00:00:00.100",
        });

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().TriggerRetrainAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Equal(1, handler.Calls);
    }

    [Theory]
    [InlineData("/model/retrain", true)]
    [InlineData("/root/model/retrain", true)]
    [InlineData("/model/status", false)]
    [InlineData("/predict", false)]
    public void Only_the_retrain_route_gets_the_retrain_timeout(string path, bool retrain) =>
        Assert.Equal(retrain, AiClientServiceCollectionExtensions.IsRetrainPipeline(path));

    [Fact]
    public async Task Total_timeout_bounds_the_whole_call()
    {
        var handler = new StubHandler(Hangs());
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:TotalTimeout"] = "00:00:00.100",
            ["AiService:AttemptTimeout"] = "00:00:00.100",
            ["AiService:MaxRetryAttempts"] = "0",
        });

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().GetModelStatusAsync());

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task Stalled_response_body_times_the_attempt_out_and_is_retried()
    {
        var handler = new StubHandler(StalledBody(), Json(HttpStatusCode.OK, AiClientTestHost.PredictJson));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:AttemptTimeout"] = "00:00:00.100",
            ["AiService:MaxRetryAttempts"] = "1",
        });

        AiPredictResponse response = await provider.Client().PredictAsync(AiClientTestHost.SampleRequest());

        Assert.Equal("v3", response.ModelVersion);
        Assert.Equal(2, handler.Calls);
    }

    [Fact]
    public async Task Stalled_response_body_surfaces_as_unavailable()
    {
        var handler = new StubHandler(StalledBody());
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:AttemptTimeout"] = "00:00:00.050",
            ["AiService:MaxRetryAttempts"] = "0",
        });

        var ex = await Assert.ThrowsAsync<AiServiceException>(() => provider.Client().PredictAsync(AiClientTestHost.SampleRequest()));

        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.IsType<TimeoutRejectedException>(ex.InnerException);
    }

    [Fact]
    public async Task Open_circuit_fails_fast_without_calling_the_service()
    {
        var handler = new StubHandler(Error(HttpStatusCode.BadGateway, "bad_gateway"));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:MaxRetryAttempts"] = "0",
            ["AiService:CircuitBreakerMinimumThroughput"] = "2",
        });
        ISoliaAiClient client = provider.Client();

        await Assert.ThrowsAsync<AiServiceException>(() => client.GetModelStatusAsync());
        await Assert.ThrowsAsync<AiServiceException>(() => client.GetModelStatusAsync());
        var ex = await Assert.ThrowsAsync<AiServiceException>(() => client.GetModelStatusAsync());

        Assert.Equal(2, handler.Calls);
        Assert.Equal(AiServiceErrorKind.Unavailable, ex.Kind);
        Assert.IsAssignableFrom<BrokenCircuitException>(ex.InnerException);
    }

    [Fact]
    public async Task Each_route_has_its_own_circuit_breaker()
    {
        var handler = new StubHandler((request, ct) => request.RequestUri!.AbsolutePath.EndsWith(SoliaAiClient.PredictPath)
            ? Json(HttpStatusCode.OK, AiClientTestHost.PredictJson)(request, ct)
            : Error(HttpStatusCode.BadGateway, "bad_gateway")(request, ct));
        using ServiceProvider provider = AiClientTestHost.Build(handler, new Dictionary<string, string?>
        {
            ["AiService:MaxRetryAttempts"] = "0",
            ["AiService:CircuitBreakerMinimumThroughput"] = "2",
        });
        ISoliaAiClient client = provider.Client();

        await Assert.ThrowsAsync<AiServiceException>(() => client.TriggerRetrainAsync());
        await Assert.ThrowsAsync<AiServiceException>(() => client.TriggerRetrainAsync());
        var open = await Assert.ThrowsAsync<AiServiceException>(() => client.TriggerRetrainAsync());
        Assert.IsAssignableFrom<BrokenCircuitException>(open.InnerException); // retrain's circuit is open...

        AiPredictResponse response = await client.PredictAsync(AiClientTestHost.SampleRequest());

        Assert.Equal("v3", response.ModelVersion); // ...but /predict still goes through
        Assert.Equal(3, handler.Calls);
    }

    [Fact]
    public async Task Caller_cancellation_propagates_as_cancellation()
    {
        var handler = new StubHandler(Hangs());
        using ServiceProvider provider = AiClientTestHost.Build(handler);
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => provider.Client().GetModelStatusAsync(cts.Token));
    }

    // ------------------------------------------------------------------ options

    [Fact]
    public void Flat_AI_SERVICE_URL_overrides_the_section_value()
    {
        using ServiceProvider provider = AiClientTestHost.Build(new StubHandler(), new Dictionary<string, string?>
        {
            ["AI_SERVICE_URL"] = " http://ai-service:9000 ",
        });

        Assert.Equal("http://ai-service:9000", provider.GetRequiredService<IOptions<AiServiceOptions>>().Value.BaseUrl);
    }

    [Fact]
    public void Flat_RETRAIN_EVERY_N_TRADES_overrides_the_retrain_threshold()
    {
        using ServiceProvider provider = AiClientTestHost.Build(new StubHandler(), new Dictionary<string, string?>
        {
            ["AiService:RetrainThreshold"] = "100",
            ["RETRAIN_EVERY_N_TRADES"] = "250",
        });

        Assert.Equal(250, provider.GetRequiredService<IOptions<AiServiceOptions>>().Value.RetrainThreshold);
    }

    [Fact]
    public void BACKEND_URL_never_binds_onto_the_ai_service_options()
    {
        using ServiceProvider provider = AiClientTestHost.Build(new StubHandler(), new Dictionary<string, string?>
        {
            ["BACKEND_URL"] = "http://backend:5000",
        });

        Assert.Equal("http://ai-service:8000/root", provider.GetRequiredService<IOptions<AiServiceOptions>>().Value.BaseUrl);
    }

    [Theory]
    [InlineData("AiService:BaseUrl", "not-a-url")]
    [InlineData("AiService:BaseUrl", "ftp://ai-service")]
    [InlineData("AiService:MaxRetryAttempts", "-1")]
    [InlineData("AiService:AttemptTimeout", "00:00:00")]
    [InlineData("AiService:CircuitBreakerFailureRatio", "0")]
    [InlineData("AiService:AttemptTimeout", "00:00:10")] // = TotalTimeout: a retry could never run
    [InlineData("AiService:AttemptTimeout", "00:00:04")] // 3 attempts x 4s > 10s total
    [InlineData("AiService:MaxRetryAttempts", "5")] // 6 attempts x 2s > 10s total
    [InlineData("AiService:MaxFeatureWindowSize", "0")]
    [InlineData("AiService:MaxFeatureWindowSize", "4097")]
    [InlineData("AiService:RetrainThreshold", "0")]
    [InlineData("AiService:RetrainTimeout", "00:00:00")]
    [InlineData("RETRAIN_EVERY_N_TRADES", "-5")]
    public void Invalid_options_fail_validation(string key, string value)
    {
        using ServiceProvider provider = AiClientTestHost.Build(new StubHandler(), new Dictionary<string, string?> { [key] = value });

        Assert.Throws<OptionsValidationException>(() => provider.GetRequiredService<IOptions<AiServiceOptions>>().Value);
    }

    [Fact]
    public void Defaults_are_valid()
    {
        var options = new AiServiceOptions();

        Assert.True(options.HasValidBaseUrl());
        Assert.True(options.HasConsistentTimeouts());
        Assert.Equal(256, options.MaxFeatureWindowSize); // a cap above the model's seq_len (64), which the ai-service applies
        Assert.Equal(500, options.RetrainThreshold); // RETRAIN_EVERY_N_TRADES in .env.example
        // Above the ai-service's 20 s start timeout (DEFAULT_START_TIMEOUT_SECONDS), so it answers first.
        Assert.True(options.RetrainTimeout > TimeSpan.FromSeconds(20));
    }

    [Theory]
    [InlineData(0, 1000, 0, 1000, true)]
    [InlineData(2, 1000, 100, 3300, true)] // 3 x 1s + 100ms x (2^2 - 1)
    [InlineData(2, 1000, 100, 3299, false)]
    [InlineData(1, 1000, 0, 1000, false)]
    public void Timeout_budget_counts_every_attempt_and_the_nominal_backoff(
        int retries, int attemptMs, int baseDelayMs, int totalMs, bool consistent)
    {
        var options = new AiServiceOptions
        {
            MaxRetryAttempts = retries,
            AttemptTimeout = TimeSpan.FromMilliseconds(attemptMs),
            RetryBaseDelay = TimeSpan.FromMilliseconds(baseDelayMs),
            TotalTimeout = TimeSpan.FromMilliseconds(totalMs),
        };

        Assert.Equal(consistent, options.HasConsistentTimeouts());
    }
}
