using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Core.Ai;
using static Solia.Api.Tests.Endpoints.AiProxyFixture;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// A real Postgres (migrations applied) seeded through raw SQL, and the real host pointed at it with
/// <see cref="ISoliaAiClient"/> stubbed, a 3-row feature window cap and a retrain threshold of 300.
/// </summary>
public sealed class AiProxyFixture : PostgresFixture, IAsyncLifetime
{
    public static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    public const int WindowSize = 3;

    public const int ConfiguredThreshold = 300;

    private WebApplicationFactory<Program> _factory = null!;

    public StubAiClient Ai { get; } = new();

    public HttpClient Client { get; private set; } = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await SeedAsync();
        _factory = TestApiFactory.Create("Production",
            [
                ("SOLIA_DB_CONNECTION", ConnectionString),
                ("AiService:MaxFeatureWindowSize", WindowSize.ToString()),
                ("RETRAIN_EVERY_N_TRADES", ConfiguredThreshold.ToString()),
            ],
            builder => builder.ConfigureTestServices(services => services.AddSingleton<ISoliaAiClient>(Ai)));
        Client = _factory.CreateClient();
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        Client.Dispose();
        await _factory.DisposeAsync();
        await DisposeAsync();
    }

    private async Task SeedAsync()
    {
        // "a" has four snapshots (one minute apart, price 0.001 … 0.004); "b" is known but has none yet.
        await ExecAsync("INSERT INTO tokens_history (mint, first_seen_at) VALUES ('a', @t0), ('b', @t0)", ("t0", T0));
        for (int i = 0; i < 4; i++)
        {
            await ExecAsync(
                "INSERT INTO features (mint, ts, price, holders, buy_sell_ratio, net_sol_flow, large_buy_count, is_migrated) " +
                "VALUES ('a', @ts, @p, @h, 1.5, -0.25, 2, @mig)",
                ("ts", T0.AddMinutes(i)), ("p", 0.001m * (i + 1)), ("h", i == 2 ? DBNull.Value : 10 * (i + 1)), ("mig", i == 3));
        }

        // Three training runs. The latest *finished* one started at T0 (and was rejected: it still consumed the
        // trades). The one started at T0+90min never finished (crashed / in flight), so it must not reset the count.
        await ExecAsync(
            """
            INSERT INTO training_events (started_at, finished_at, trigger, decision) VALUES
                (@a, @a, 'initial', 'promoted'), (@b, @bEnd, 'n_trades', 'rejected'), (@c, NULL, 'n_trades', NULL)
            """,
            ("a", T0.AddHours(-3)), ("b", T0), ("bEnd", T0.AddMinutes(10)), ("c", T0.AddMinutes(90)));

        // Closed after T0 (one before, one after the unfinished run started): two counted. Not counted: closed before T0, closed exactly at T0, still open, closed
        // without a closed_at.
        const string closedSql = "INSERT INTO trades (mint, opened_at, closed_at, pnl, strategy, status) VALUES ('a', @o, @c, 1, 'buy', 'closed')";
        await ExecAsync(closedSql, ("o", T0.AddHours(-3)), ("c", T0.AddHours(-2)));
        await ExecAsync(closedSql, ("o", T0.AddHours(-1)), ("c", T0));
        await ExecAsync(closedSql, ("o", T0), ("c", T0.AddHours(1)));
        await ExecAsync(closedSql, ("o", T0), ("c", T0.AddHours(2)));
        await ExecAsync("INSERT INTO trades (mint, opened_at, strategy, status) VALUES ('a', @o, 'buy', 'open')", ("o", T0.AddHours(3)));
        await ExecAsync("INSERT INTO trades (mint, opened_at, pnl, strategy, status) VALUES ('a', @o, 1, 'buy', 'closed')", ("o", T0.AddHours(3)));
    }

    private async Task ExecAsync(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        await cmd.ExecuteNonQueryAsync();
    }
}

/// <summary>
/// The AI-proxy endpoints end to end over the <see cref="AiProxyFixture"/>: the feature window sent to the
/// ai-service, the documented response shapes, 404s and the trades-since-last-train count. Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class AiProxyIntegrationTests(AiProxyFixture api) : IClassFixture<AiProxyFixture>
{
    private static async Task<JsonElement> JsonAsync(HttpResponseMessage response)
    {
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        return doc.RootElement.Clone();
    }

    private AiPredictRequest LastRequest() => api.Ai.PredictRequests[^1];

    /// <summary>A healthy ai-service answer, for tests that only look at the request the backend sent.</summary>
    private void AnswerPredictions() =>
        api.Ai.Predict = r => new AiPredictResponse(r.Mint, "v3", [new AiPrediction("1m", 0.72, 0.08, 0.35, 81.5)]);

    [Fact]
    public async Task Predict_sends_the_latest_window_oldest_first_and_returns_the_documented_shape()
    {
        api.Ai.Predict = r => new AiPredictResponse(r.Mint, "v7",
            [new AiPrediction("1m", 0.72, 0.08, 0.35, 81.5), new AiPrediction("5m", 0.5, 0.2, -0.1, 40)]);

        HttpResponseMessage response = await api.Client.PostAsJsonAsync("/api/v1/predict", new { mint = " a ", horizons = new[] { "1m", "5m" } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await JsonAsync(response);
        Assert.Equal("a", body.GetProperty("mint").GetString());
        Assert.Equal("v7", body.GetProperty("model_version").GetString());
        JsonElement first = body.GetProperty("predictions")[0];
        Assert.Equal("1m", first.GetProperty("horizon").GetString());
        Assert.Equal(0.72, first.GetProperty("pump_prob").GetDouble());
        Assert.Equal(0.08, first.GetProperty("rug_prob").GetDouble());
        Assert.Equal(0.35, first.GetProperty("expected_return").GetDouble());
        Assert.Equal(81.5, first.GetProperty("confidence").GetDouble());
        Assert.Equal(2, body.GetProperty("predictions").GetArrayLength());
        // Predictions only: no trade decision leaks into the public shape.
        Assert.Equal(["mint", "model_version", "predictions"], body.EnumerateObject().Select(p => p.Name).ToArray());

        AiPredictRequest sent = LastRequest();
        Assert.Equal("a", sent.Mint);
        Assert.Equal(["1m", "5m"], sent.Horizons!);
        var window = Assert.IsAssignableFrom<IReadOnlyList<IReadOnlyDictionary<string, double?>>>(sent.Features);
        Assert.Equal(WindowSize, window.Count);
        Assert.Equal([0.002, 0.003, 0.004], window.Select(r => r["price"]!.Value).ToArray());
        Assert.Equal([20.0, null, 40.0], window.Select(r => r["holders"]).ToArray());
        Assert.Equal([0.0, 0.0, 1.0], window.Select(r => r["is_migrated"]!.Value).ToArray());
        Assert.Equal(1.5, window[0]["buy_sell_ratio"]);
        Assert.Null(window[0]["rsi"]);
        Assert.Equal(-0.25, window[0]["net_sol_flow"]);
        Assert.Equal(2.0, window[0]["large_buy_count"]);
        Assert.Null(window[0]["max_buy_sol"]);
        Assert.Equal(24, window[0].Count); // the ai-service's FEATURE_COLUMNS
    }

    [Fact]
    public async Task Predict_deduplicates_horizons_after_trimming()
    {
        AnswerPredictions();

        HttpResponseMessage response = await api.Client.PostAsJsonAsync(
            "/api/v1/predict", new { mint = "a", horizons = new[] { "1h", " 1h", "5m", "1h " } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(["1h", "5m"], LastRequest().Horizons!);
    }

    [Fact]
    public async Task Predict_on_a_window_too_short_for_the_champion_is_a_422_envelope()
    {
        api.Ai.Predict = _ => throw StubAiClient.WindowTooShort();

        HttpResponseMessage response = await api.Client.PostAsJsonAsync("/api/v1/predict", new { mint = "a" });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        string raw = await response.Content.ReadAsStringAsync();
        JsonElement error = (await JsonAsync(response)).GetProperty("error");
        Assert.Equal("insufficient_history", error.GetProperty("code").GetString());
        Assert.Equal("not enough feature history yet to score this token", error.GetProperty("message").GetString());
        Assert.DoesNotContain("champion needs", raw);
    }

    [Fact]
    public async Task Predict_without_horizons_asks_for_all_of_them()
    {
        AnswerPredictions();

        HttpResponseMessage response = await api.Client.PostAsJsonAsync("/api/v1/predict", new { mint = "a" });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Null(LastRequest().Horizons);
    }

    [Theory]
    [InlineData("nope", "unknown mint nope")]
    [InlineData("b", "no feature snapshot yet for mint b")]
    public async Task Predict_for_a_mint_without_features_is_a_404_envelope(string mint, string message)
    {
        int before = api.Ai.Calls;

        HttpResponseMessage response = await api.Client.PostAsJsonAsync("/api/v1/predict", new { mint });

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        JsonElement error = (await JsonAsync(response)).GetProperty("error");
        Assert.Equal("not_found", error.GetProperty("code").GetString());
        Assert.Equal(message, error.GetProperty("message").GetString());
        Assert.Equal(before, api.Ai.Calls);
    }

    [Fact]
    public async Task Predict_with_no_champion_is_a_503_envelope()
    {
        api.Ai.Predict = _ => throw StubAiClient.NoChampion();

        HttpResponseMessage response = await api.Client.PostAsJsonAsync("/api/v1/predict", new { mint = "a" });

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("unavailable", (await JsonAsync(response)).GetProperty("error").GetProperty("code").GetString());
    }

    [Fact]
    public async Task Model_status_combines_the_champion_with_the_local_trade_count()
    {
        DateTimeOffset created = T0.AddDays(-1);
        api.Ai.Status = () => new AiModelStatus("v3", created, 1234,
            new Dictionary<string, JsonElement> { ["val_auc"] = JsonSerializer.SerializeToElement(0.81) }, null, null);

        HttpResponseMessage response = await api.Client.GetAsync("/api/v1/model/status");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await JsonAsync(response);
        Assert.Equal("v3", body.GetProperty("champion_version").GetString());
        Assert.Equal(created, body.GetProperty("created_at").GetDateTimeOffset());
        Assert.Equal(0.81, body.GetProperty("metrics").GetProperty("val_auc").GetDouble());
        Assert.Equal(2, body.GetProperty("trades_since_last_train").GetInt64());
        Assert.Equal(ConfiguredThreshold, body.GetProperty("retrain_threshold").GetInt32());
    }

    [Fact]
    public async Task Model_status_takes_count_and_threshold_from_the_ai_service_when_it_reports_both()
    {
        api.Ai.Status = () => new AiModelStatus("v4", null, null, null, 9999, 777);

        JsonElement body = await JsonAsync(await api.Client.GetAsync("/api/v1/model/status"));

        Assert.Equal(9999, body.GetProperty("trades_since_last_train").GetInt64());
        Assert.Equal(777, body.GetProperty("retrain_threshold").GetInt32());
        Assert.Empty(body.GetProperty("metrics").EnumerateObject());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("created_at").ValueKind);
    }

    [Theory]
    [InlineData(null, 777)]
    [InlineData(9999, null)]
    public async Task Model_status_never_mixes_sources_when_the_ai_service_reports_only_half(int? count, int? threshold)
    {
        api.Ai.Status = () => new AiModelStatus("v4", null, null, null, count, threshold);

        JsonElement body = await JsonAsync(await api.Client.GetAsync("/api/v1/model/status"));

        Assert.Equal(2, body.GetProperty("trades_since_last_train").GetInt64());
        Assert.Equal(ConfiguredThreshold, body.GetProperty("retrain_threshold").GetInt32());
    }

    public static TheoryData<string> AiFailures() => ["unreachable", "no_champion", "upstream_error"];

    [Theory]
    [MemberData(nameof(AiFailures))]
    public async Task Model_status_still_reports_the_local_progress_when_the_ai_service_fails(string failure)
    {
        api.Ai.Status = () => throw (failure switch
        {
            "unreachable" => StubAiClient.Unreachable(),
            "no_champion" => StubAiClient.NoChampion(),
            _ => new AiServiceException(AiServiceErrorKind.UpstreamError, "Traceback: boom", 500),
        });

        HttpResponseMessage response = await api.Client.GetAsync("/api/v1/model/status");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string raw = await response.Content.ReadAsStringAsync();
        JsonElement body = await JsonAsync(response);
        Assert.Equal(JsonValueKind.Null, body.GetProperty("champion_version").ValueKind);
        Assert.Equal(JsonValueKind.Null, body.GetProperty("created_at").ValueKind);
        Assert.Empty(body.GetProperty("metrics").EnumerateObject());
        Assert.Equal(2, body.GetProperty("trades_since_last_train").GetInt64());
        Assert.Equal(ConfiguredThreshold, body.GetProperty("retrain_threshold").GetInt32());
        Assert.DoesNotContain("10.0.0.7", raw);
        Assert.DoesNotContain("Traceback", raw);
    }
}
