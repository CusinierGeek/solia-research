using System.Net;
using System.Text.Json;
using Solia.Dashboard.Api;

namespace Solia.Dashboard.Tests.Api;

/// <summary>
/// The client against a stubbed handler: request paths/queries, the contract's sample payloads (fixtures taken from
/// docs/API-CONTRACTS.md, pinning the dashboard-owned DTOs) and the error mapping.
/// </summary>
public class SoliaApiClientTests
{
    private static readonly Uri BaseAddress = new("http://backend:8080/api/v1/");

    private static (SoliaApiClient Client, StubHttpHandler Handler) Create(StubHttpHandler handler) =>
        (new SoliaApiClient(new HttpClient(handler) { BaseAddress = BaseAddress }), handler);

    private static (SoliaApiClient Client, StubHttpHandler Handler) Create(string json, HttpStatusCode status = HttpStatusCode.OK) =>
        Create(StubHttpHandler.Json(json, status));

    private static string Requested(StubHttpHandler handler) => Assert.Single(handler.Requests).PathAndQuery;

    // --- Fixtures (docs/API-CONTRACTS.md) ---

    private const string LiveTokensJson = """
        [ { "mint": "So1abc", "symbol": "ABC", "name": "Alpha", "created_at": "2026-10-26T12:00:00Z",
            "last_price": 0.0000001234, "last_market_cap": 1000.5, "last_liquidity": 42.0, "holders": 311,
            "trending_score": 0.7, "venue": "pumpfun", "is_migrated": false, "updated_at": "2026-10-26T12:05:00+00:00" } ]
        """;

    private const string PredictionJson = """
        { "id": 42, "mint": "So1abc", "symbol": "ABC", "ts": "2026-10-26T12:00:00Z", "horizon": "1m", "pump_prob": 0.81,
          "rug_prob": 0.07, "strategy": "buy", "confidence": 74.0, "position_size": 0.5,
          "take_profit": 0.0000001500, "stop_loss": 0.0000001100, "model_version": "v3" }
        """;

    private const string TradeJson = """
        { "id": 7, "mint": "So1abc", "prediction_id": 42, "opened_at": "2026-10-26T12:00:00Z",
          "closed_at": "2026-10-26T12:01:30Z", "entry_price": 0.0000001234, "exit_price": 0.000000150123456789,
          "size": 0.5, "duration_sec": 90, "pnl": 0.0789, "pnl_pct": 0.2166, "strategy": "buy", "confidence": 74.0,
          "exit_reason": "take_profit", "status": "closed" }
        """;

    // --- Paths & queries ---

    [Fact]
    public async Task Live_tokens_hits_its_path_with_limit_and_sort()
    {
        var (client, handler) = Create(LiveTokensJson);

        await client.GetLiveTokensAsync(new LiveTokensQuery { Limit = 20, Sort = TokenSort.MarketCap });

        Assert.Equal("/api/v1/tokens/live?limit=20&sort=market_cap", Requested(handler));
    }

    [Fact]
    public async Task Token_history_sends_the_range_outcome_and_paging()
    {
        var (client, handler) = Create("""{ "items": [], "page": 3, "page_size": 25, "total": 0 }""");

        await client.GetTokenHistoryAsync(new TokenHistoryQuery
        {
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            To = new DateTimeOffset(2026, 10, 2, 0, 0, 0, TimeSpan.Zero),
            Outcome = TokenOutcome.Rug,
            Page = 3,
            PageSize = 25,
        });

        Assert.Equal(
            "/api/v1/tokens/history?from=2026-10-01T00%3A00%3A00Z&to=2026-10-02T00%3A00%3A00Z&outcome=rug&page=3&pageSize=25",
            Requested(handler));
    }

    [Fact]
    public async Task Live_predictions_sends_horizon_and_sort()
    {
        var (client, handler) = Create($"[{PredictionJson}]");

        await client.GetLivePredictionsAsync(new LivePredictionsQuery { Horizon = PredictionHorizon.Seconds30, Sort = PredictionSort.PumpProb });

        Assert.Equal("/api/v1/predictions/live?horizon=30s&sort=pump_prob", Requested(handler));
    }

    [Fact]
    public async Task Prediction_history_omits_a_blank_mint()
    {
        var (client, handler) = Create("""{ "items": [], "page": 1, "page_size": 50, "total": 0 }""");

        await client.GetPredictionHistoryAsync(new PredictionHistoryQuery { Mint = "  ", Strategy = TradeStrategy.Hold });

        Assert.Equal("/api/v1/predictions/history?strategy=hold", Requested(handler));
    }

    [Fact]
    public async Task Prediction_history_sends_every_filter()
    {
        var (client, handler) = Create("""{ "items": [], "page": 1, "page_size": 50, "total": 0 }""");

        await client.GetPredictionHistoryAsync(new PredictionHistoryQuery
        {
            Mint = " So1abc ",
            Horizon = PredictionHorizon.Hour1,
            Strategy = TradeStrategy.Buy,
            From = new DateTimeOffset(2026, 10, 1, 0, 0, 0, TimeSpan.Zero),
            Page = 2,
        });

        Assert.Equal("/api/v1/predictions/history?mint=So1abc&horizon=1h&strategy=buy&from=2026-10-01T00%3A00%3A00Z&page=2",
            Requested(handler));
    }

    [Fact]
    public async Task Trade_history_sends_status_and_paging()
    {
        var (client, handler) = Create("""{ "items": [], "page": 1, "page_size": 10, "total": 0 }""");

        await client.GetTradeHistoryAsync(new TradeHistoryQuery { Status = TradeStatus.Open, PageSize = 10 });

        Assert.Equal("/api/v1/trades/history?status=open&pageSize=10", Requested(handler));
    }

    [Fact]
    public async Task Winrate_sends_its_window()
    {
        var (client, handler) = Create("""{ "winrate": 0.58, "wins": 290, "losses": 190, "break_even": 20, "window": "7d" }""");

        await client.GetWinRateAsync(StatsWindow.Last7Days);

        Assert.Equal("/api/v1/stats/winrate?window=7d", Requested(handler));
    }

    [Fact]
    public async Task Performance_sends_its_limit()
    {
        var (client, handler) = Create("""{ "points": [], "total_pnl": 0 }""");

        await client.GetPerformanceAsync(limit: 500);

        Assert.Equal("/api/v1/stats/performance?limit=500", Requested(handler));
    }

    [Fact]
    public async Task Bot_logs_sends_level_service_and_limit()
    {
        var (client, handler) = Create("[]");

        await client.GetBotLogsAsync(new BotLogsQuery { Level = BotLogLevel.Warn, Service = LogService.DataPipeline, Limit = 200 });

        Assert.Equal("/api/v1/logs/bot?level=warn&service=data-pipeline&limit=200", Requested(handler));
    }

    [Fact]
    public async Task Calls_without_a_query_send_no_parameters()
    {
        var (client, handler) = Create(new StubHttpHandler(
            StubHttpHandler.Respond("[]"),
            StubHttpHandler.Respond("""{ "items": [], "page": 1, "page_size": 50, "total": 0 }"""),
            StubHttpHandler.Respond("""{ "winrate": 0, "wins": 0, "losses": 0, "break_even": 0, "window": "all" }"""),
            StubHttpHandler.Respond("""{ "champion_version": null, "created_at": null, "metrics": {}, "trades_since_last_train": 0, "retrain_threshold": 500 }""")));

        await client.GetLiveTokensAsync();
        await client.GetTradeHistoryAsync();
        await client.GetWinRateAsync();
        await client.GetModelStatusAsync();

        Assert.Equal(["/api/v1/tokens/live", "/api/v1/trades/history", "/api/v1/stats/winrate", "/api/v1/model/status"],
            handler.Requests.Select(u => u.PathAndQuery));
    }

    // --- Payloads ---

    [Fact]
    public async Task Live_tokens_deserialize_the_contract_sample()
    {
        var (client, _) = Create(LiveTokensJson);

        LiveTokenDto token = Assert.Single(await client.GetLiveTokensAsync());

        Assert.Equal("So1abc", token.Mint);
        Assert.Equal("ABC", token.Symbol);
        Assert.Equal(0.0000001234m, token.LastPrice);
        Assert.Equal(1000.5m, token.LastMarketCap);
        Assert.Equal(311, token.Holders);
        Assert.Equal(0.7, token.TrendingScore);
        Assert.False(token.IsMigrated);
        Assert.Equal(new DateTimeOffset(2026, 10, 26, 12, 5, 0, TimeSpan.Zero), token.UpdatedAt);
        Assert.Equal(TimeSpan.Zero, token.UpdatedAt.Offset);
    }

    [Fact]
    public async Task Token_history_deserializes_a_page()
    {
        var (client, _) = Create("""
            { "items": [ { "mint": "So1abc", "symbol": null, "name": null, "created_at": null,
                           "first_seen_at": "2026-10-26T12:00:00Z", "last_seen_at": null, "final_outcome": "pump" } ],
              "page": 2, "page_size": 1, "total": 1234 }
            """);

        PagedResult<TokenHistoryDto> page = await client.GetTokenHistoryAsync();

        Assert.Equal((2, 1, 1234L), (page.Page, page.PageSize, page.Total));
        TokenHistoryDto item = Assert.Single(page.Items);
        Assert.Equal("pump", item.FinalOutcome);
        Assert.Null(item.Symbol);
    }

    [Fact]
    public async Task Predictions_deserialize_with_exact_decimals_and_double_probabilities()
    {
        var (client, _) = Create($$"""{ "items": [{{PredictionJson}}], "page": 1, "page_size": 50, "total": 1 }""");

        PredictionRecordDto p = Assert.Single((await client.GetPredictionHistoryAsync()).Items);

        Assert.Equal(42, p.Id);
        Assert.Equal("1m", p.Horizon);
        Assert.Equal(0.81, p.PumpProb);
        Assert.Equal(0.07, p.RugProb);
        Assert.Equal(74.0, p.Confidence);
        Assert.Equal(0.5m, p.PositionSize);
        Assert.Equal(0.0000001500m, p.TakeProfit);
        Assert.Equal("0.0000001500", p.TakeProfit!.Value.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Assert.Equal(0.0000001100m, p.StopLoss);
        Assert.Equal("v3", p.ModelVersion);
    }

    [Fact]
    public async Task A_live_prediction_with_nulls_deserializes()
    {
        var (client, _) = Create("""
            [ { "id": 812, "mint": "So1abc", "symbol": "ABC", "ts": "2026-10-26T12:00:00Z", "horizon": "1m", "pump_prob": 0.82,
                "rug_prob": 0.07, "strategy": null, "confidence": 81.0, "position_size": null, "take_profit": null,
                "stop_loss": null, "model_version": null } ]
            """);

        PredictionRecordDto p = Assert.Single(await client.GetLivePredictionsAsync());

        Assert.Null(p.Strategy);
        Assert.Null(p.PositionSize);
        Assert.Null(p.ModelVersion);
    }

    [Fact]
    public async Task Trades_keep_full_decimal_precision()
    {
        var (client, _) = Create($$"""{ "items": [{{TradeJson}}], "page": 1, "page_size": 50, "total": 1 }""");

        TradeDto t = Assert.Single((await client.GetTradeHistoryAsync()).Items);

        Assert.Equal(42L, t.PredictionId);
        Assert.Equal(0.000000150123456789m, t.ExitPrice);
        Assert.Equal(0.0789m, t.Pnl);
        Assert.Equal(0.2166, t.PnlPct);
        Assert.Equal(90, t.DurationSec);
        Assert.Equal("take_profit", t.ExitReason);
        Assert.Equal("closed", t.Status);
        Assert.Equal(new DateTimeOffset(2026, 10, 26, 12, 1, 30, TimeSpan.Zero), t.ClosedAt);
    }

    [Fact]
    public async Task Winrate_deserializes_the_contract_sample()
    {
        var (client, _) = Create("""{ "winrate": 0.58, "wins": 290, "losses": 190, "break_even": 20, "window": "all" }""");

        Assert.Equal(new WinRateResponse(0.58, 290, 190, 20, "all"), await client.GetWinRateAsync());
    }

    [Fact]
    public async Task Performance_deserializes_the_contract_sample()
    {
        var (client, _) = Create("""{ "points": [ { "ts": "2026-10-26T12:00:00Z", "equity": 11.5, "pnl": 1.5 } ], "total_pnl": 3.2 }""");

        PerformanceResponse perf = await client.GetPerformanceAsync();

        Assert.Equal(3.2m, perf.TotalPnl);
        EquityPoint point = Assert.Single(perf.Points);
        Assert.Equal((11.5m, 1.5m), (point.Equity, point.Pnl));
    }

    [Fact]
    public async Task Bot_logs_keep_their_structured_context()
    {
        var (client, _) = Create("""
            [ { "id": 1, "ts": "2026-10-26T12:00:00Z", "service": "bot", "level": "warn", "message": "slow tick",
                "context": { "mint": "So1abc", "lag_ms": 1200 } },
              { "id": 2, "ts": "2026-10-26T11:59:00Z", "service": "backend", "level": "info", "message": "up", "context": null } ]
            """);

        IReadOnlyList<LogDto> logs = await client.GetBotLogsAsync();

        Assert.Equal(2, logs.Count);
        Assert.Equal(1200, logs[0].Context!.Value.GetProperty("lag_ms").GetInt32());
        Assert.Null(logs[1].Context);
    }

    [Fact]
    public async Task Model_status_deserializes_metrics_and_retrain_progress()
    {
        var (client, _) = Create("""
            { "champion_version": "v3", "created_at": "2026-10-20T08:00:00Z", "metrics": { "auc_pump": 0.71, "notes": "ok" },
              "trades_since_last_train": 142, "retrain_threshold": 500 }
            """);

        ModelStatusResponse status = await client.GetModelStatusAsync();

        Assert.Equal("v3", status.ChampionVersion);
        Assert.Equal(0.71, status.Metrics["auc_pump"].GetDouble());
        Assert.Equal(JsonValueKind.String, status.Metrics["notes"].ValueKind);
        Assert.Equal((142L, 500), (status.TradesSinceLastTrain, status.RetrainThreshold));
    }

    [Fact]
    public async Task Model_status_without_a_champion_deserializes()
    {
        var (client, _) = Create("""
            { "champion_version": null, "created_at": null, "metrics": {}, "trades_since_last_train": 3, "retrain_threshold": 500 }
            """);

        ModelStatusResponse status = await client.GetModelStatusAsync();

        Assert.Null(status.ChampionVersion);
        Assert.Empty(status.Metrics);
    }

    // --- Errors ---

    [Theory]
    [InlineData(HttpStatusCode.BadRequest, "invalid_request", "pageSize must be between 1 and 200.")]
    [InlineData(HttpStatusCode.NotFound, "not_found", "unknown mint")]
    [InlineData(HttpStatusCode.ServiceUnavailable, "unavailable", "the ai-service is unavailable")]
    public async Task An_error_envelope_becomes_a_typed_error(HttpStatusCode status, string code, string message)
    {
        var (client, _) = Create($$"""{ "error": { "code": "{{code}}", "message": "{{message}}" } }""", status);

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => client.GetTradeHistoryAsync());

        Assert.Equal((int)status, ex.StatusCode);
        Assert.Equal(code, ex.Code);
        Assert.Equal(message, ex.Message);
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound, "not_found")]
    [InlineData(HttpStatusCode.BadRequest, "invalid_request")]
    [InlineData(HttpStatusCode.InternalServerError, "internal_error")]
    public async Task A_non_envelope_error_is_classified_by_status(HttpStatusCode status, string code)
    {
        var (client, _) = Create(new StubHttpHandler(StubHttpHandler.Respond("<html>oops</html>", status, "text/html")));

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => client.GetModelStatusAsync());

        Assert.Equal((int)status, ex.StatusCode);
        Assert.Equal(code, ex.Code);
        Assert.DoesNotContain("oops", ex.Message);
    }

    [Theory]
    [InlineData(HttpStatusCode.RequestTimeout)]
    [InlineData(HttpStatusCode.BadGateway)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    [InlineData(HttpStatusCode.GatewayTimeout)]
    public async Task A_gateway_error_without_the_envelope_means_the_backend_is_unavailable(HttpStatusCode status)
    {
        var (client, _) = Create(new StubHttpHandler(StubHttpHandler.Respond("Bad gateway", status, "text/plain")));

        var ex = await Assert.ThrowsAsync<BackendUnavailableException>(() => client.GetLiveTokensAsync());

        Assert.Equal(BackendUnavailableException.DefaultMessage, ex.Message);
    }

    [Fact]
    public async Task A_network_failure_is_unavailable_without_leaking_its_text()
    {
        var (client, _) = Create(new StubHttpHandler(StubHttpHandler.Throw(new HttpRequestException("No such host is known. (backend:8080)"))));

        var ex = await Assert.ThrowsAsync<BackendUnavailableException>(() => client.GetBotLogsAsync());

        Assert.Equal(BackendUnavailableException.DefaultMessage, ex.Message);
        Assert.IsType<HttpRequestException>(ex.InnerException);
    }

    [Fact]
    public async Task An_http_client_timeout_is_unavailable()
    {
        var http = new HttpClient(new StubHttpHandler(StubHttpHandler.Hang()))
        {
            BaseAddress = BaseAddress,
            Timeout = TimeSpan.FromMilliseconds(50),
        };

        await Assert.ThrowsAsync<BackendUnavailableException>(() => new SoliaApiClient(http).GetWinRateAsync());
    }

    [Fact]
    public async Task The_callers_cancellation_is_not_a_backend_failure()
    {
        var (client, _) = Create(new StubHttpHandler(StubHttpHandler.Hang()));
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => client.GetPerformanceAsync(cancellationToken: cts.Token));
    }

    [Theory]
    [InlineData("{ not json")]
    [InlineData("null")]
    [InlineData("""{ "winrate": "high" }""")]
    [InlineData("{}")] // missing fields are contract drift, not zeros
    [InlineData("""{ "winrate": 0.5, "wins": 1, "losses": 1, "break_even": 0 }""")]
    [InlineData("""{ "winrate": 0.5, "wins": 1, "losses": 1, "break_even": 0, "window": null }""")]
    public async Task An_unreadable_success_body_is_an_invalid_response(string body)
    {
        var (client, _) = Create(body);

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => client.GetWinRateAsync());

        Assert.Equal(BackendErrorException.InvalidResponseCode, ex.Code);
        Assert.Equal(200, ex.StatusCode);
    }

    [Fact]
    public async Task A_null_where_the_contract_forbids_it_is_an_invalid_response()
    {
        var (client, _) = Create("""
            { "champion_version": null, "created_at": null, "metrics": null, "trades_since_last_train": 3, "retrain_threshold": 500 }
            """);

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => client.GetModelStatusAsync());

        Assert.Equal(BackendErrorException.InvalidResponseCode, ex.Code);
    }

    [Fact]
    public async Task A_partial_error_envelope_keeps_its_code()
    {
        var (client, _) = Create("""{ "error": { "code": "not_found" } }""", HttpStatusCode.NotFound);

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => client.GetTradeHistoryAsync());

        Assert.Equal(("not_found", "The backend answered 404."), (ex.Code, ex.Message));
    }
}
