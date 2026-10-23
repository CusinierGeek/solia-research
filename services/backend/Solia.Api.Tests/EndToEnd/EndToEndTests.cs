using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Api.Tests.Endpoints;
using Solia.Core.Ai;
using Solia.Secrets;

namespace Solia.Api.Tests.EndToEnd;

/// <summary>
/// The Sprint-4 Definition of Done, proven against the production host (see <see cref="EndToEndFixture"/>):
/// every endpoint returns its documented shape, <c>/predict</c> and <c>/model/*</c> proxy the ai-service, the
/// hosted bot writes <c>trades</c>, secrets decrypt from <c>secrets.enc</c> with nothing secret logged, and the
/// health probes follow the database and the ai-service. Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class EndToEndTests(EndToEndFixture e2e) : IClassFixture<EndToEndFixture>
{
    private const string LiveTokenFields =
        "created_at,holders,is_migrated,last_liquidity,last_market_cap,last_price,mint,name,symbol,trending_score,updated_at,venue";

    private const string TokenHistoryFields = "created_at,final_outcome,first_seen_at,last_seen_at,mint,name,symbol";

    private const string TradeFields =
        "closed_at,confidence,duration_sec,entry_price,exit_price,exit_reason,id,mint,opened_at,pnl,pnl_pct,prediction_id,size,status,strategy";

    private const string PageFields = "items,page,page_size,total";

    // ------------------------------------------------------------------ documented shapes

    /// <summary>
    /// (url, root fields, path of the element list — "" for a root array, null when none —, element fields).
    /// Fields are sorted, comma-separated.
    /// </summary>
    public static TheoryData<string, string?, string?, string?> ReadShapes() => new()
    {
        { "/api/v1/tokens/live", null, "", LiveTokenFields },
        { "/api/v1/tokens/history", PageFields, "items", TokenHistoryFields },
        { "/api/v1/trades/history?status=closed", PageFields, "items", TradeFields },
        { "/api/v1/stats/winrate", "break_even,losses,window,winrate,wins", null, null },
        { "/api/v1/stats/performance", "points,total_pnl", "points", "equity,pnl,ts" },
        { "/api/v1/logs/bot", null, "", "context,id,level,message,service,ts" },
        { "/api/v1/model/status", "champion_version,created_at,metrics,retrain_threshold,trades_since_last_train", null, null },
    };

    [Theory]
    [MemberData(nameof(ReadShapes))]
    public async Task Every_read_endpoint_returns_its_documented_shape(string url, string? rootFields, string? listPath, string? elementFields)
    {
        HttpResponseMessage response = await e2e.Client.GetAsync(url);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        JsonElement root = doc.RootElement;
        if (rootFields is not null)
        {
            Assert.Equal(rootFields, Fields(root));
        }

        if (listPath is not null)
        {
            JsonElement list = listPath.Length == 0 ? root : root.GetProperty(listPath);
            Assert.Equal(JsonValueKind.Array, list.ValueKind);
            Assert.NotEmpty(list.EnumerateArray());
            Assert.All(list.EnumerateArray(), element => Assert.Equal(elementFields, Fields(element)));
        }
    }

    // ------------------------------------------------------------------ ai-service proxy

    [Fact]
    public async Task Predict_sends_the_tokens_feature_window_to_the_ai_service_and_returns_its_scores()
    {
        HttpResponseMessage response = await e2e.Client.PostAsJsonAsync("/api/v1/predict",
            new { mint = EndToEndFixture.PredictMint, horizons = new[] { "1m", " 1m" } });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("mint,model_version,predictions", Fields(doc.RootElement));
        Assert.Equal(EndToEndFixture.PredictMint, doc.RootElement.GetProperty("mint").GetString());
        Assert.Equal("v3", doc.RootElement.GetProperty("model_version").GetString());
        JsonElement prediction = Assert.Single(doc.RootElement.GetProperty("predictions").EnumerateArray());
        Assert.Equal("confidence,expected_return,horizon,pump_prob,rug_prob", Fields(prediction));
        Assert.Equal(0.72, prediction.GetProperty("pump_prob").GetDouble());

        AiPredictRequest sent = PredictRequestsFor(EndToEndFixture.PredictMint).Last();
        Assert.Equal(["1m"], sent.Horizons);
        var window = Assert.IsAssignableFrom<IEnumerable<IReadOnlyDictionary<string, double?>>>(sent.Features).ToList();
        Assert.Equal([1.0, 2.0, 3.0], window.Select(row => row["price"])); // the seeded rows, oldest first
    }

    [Fact]
    public async Task Predict_maps_ai_service_failures_to_the_envelope()
    {
        Func<AiPredictRequest, AiPredictResponse> original = e2e.Ai.Predict;
        e2e.Ai.Predict = r => r.Mint == EndToEndFixture.PredictMint ? throw StubAiClient.NoChampion() : original(r);
        try
        {
            HttpResponseMessage response = await e2e.Client.PostAsJsonAsync("/api/v1/predict", new { mint = EndToEndFixture.PredictMint });

            await AssertEnvelopeAsync(response, HttpStatusCode.ServiceUnavailable, "unavailable");
        }
        finally
        {
            e2e.Ai.Predict = original;
        }

        await AssertEnvelopeAsync(
            await e2e.Client.PostAsJsonAsync("/api/v1/predict", new { mint = "never-seen" }), HttpStatusCode.NotFound, "not_found");
    }

    [Fact]
    public async Task Model_status_and_retrain_proxy_the_ai_service()
    {
        e2e.Ai.Status = () => new AiModelStatus("v3", EndToEndFixture.T0, 1234,
            new Dictionary<string, JsonElement> { ["auc"] = JsonSerializer.SerializeToElement(0.61) }, null, null);

        JsonElement status = await e2e.Client.GetFromJsonAsync<JsonElement>("/api/v1/model/status");
        HttpResponseMessage retrain = await e2e.Client.PostAsync("/api/v1/model/retrain", content: null);

        Assert.Equal("v3", status.GetProperty("champion_version").GetString());
        Assert.Equal(0.61, status.GetProperty("metrics").GetProperty("auc").GetDouble());
        Assert.Equal(HttpStatusCode.Accepted, retrain.StatusCode);
        using JsonDocument body = JsonDocument.Parse(await retrain.Content.ReadAsStringAsync());
        Assert.Equal(42, body.RootElement.GetProperty("training_event_id").GetInt64());
    }

    // ------------------------------------------------------------------ hosted bot

    [Fact]
    public async Task The_hosted_bot_trades_the_live_feed_and_its_trades_are_served()
    {
        const string closed = "SELECT count(*) FROM trades WHERE mint = @m AND status = 'closed'";
        (string, object) mint = ("m", EndToEndFixture.BotMint);

        // The data-pipeline's part: snapshots of a tracked token. 1.0 → buy; 1.6 → through the take-profit.
        await e2e.ExecAsync("INSERT INTO features (mint, ts, price, holders) VALUES (@m, @ts, 1.0, 10)", mint, ("ts", EndToEndFixture.T0));
        await e2e.WaitForAsync("SELECT count(*) FROM trades WHERE mint = @m", 1, mint);
        await e2e.ExecAsync("INSERT INTO features (mint, ts, price, holders) VALUES (@m, @ts, 1.6, 10)", mint,
            ("ts", EndToEndFixture.T0.AddSeconds(30)));
        await e2e.WaitForAsync(closed, 1, mint);

        Assert.Equal("tp", await e2e.ScalarAsync<string>("SELECT exit_reason FROM trades WHERE mint = @m", mint));
        Assert.True(await e2e.ScalarAsync<decimal>("SELECT pnl FROM trades WHERE mint = @m", mint) > 0m);

        // The trade is served by the read API like any other.
        JsonElement page = await e2e.Client.GetFromJsonAsync<JsonElement>("/api/v1/trades/history?status=closed&page_size=100");
        JsonElement trade = page.GetProperty("items").EnumerateArray().Single(t => t.GetProperty("mint").GetString() == EndToEndFixture.BotMint);
        Assert.Equal("closed", trade.GetProperty("status").GetString());
        Assert.Equal(1.0m, trade.GetProperty("entry_price").GetDecimal());
        Assert.Equal(1.6m, trade.GetProperty("exit_price").GetDecimal());
    }

    // ------------------------------------------------------------------ secrets

    [Fact]
    public async Task Secrets_decrypt_from_the_file_and_nothing_secret_is_logged()
    {
        // The configured connection string has no password: the host reaching Postgres at all is the
        // db_password secret, decrypted from secrets.enc with SOLIA_MASTER_KEY.
        Assert.Null(new NpgsqlConnectionStringBuilder(e2e.PasswordlessConnectionString).Password);
        Assert.Equal(e2e.ApiKey, e2e.Services.GetRequiredService<ISecretsProvider>().Get(EndToEndFixture.ApiKeyName));
        Assert.Equal(HttpStatusCode.OK, (await e2e.Client.GetAsync("/api/v1/tokens/live")).StatusCode);

        // An upstream failure echoing secrets is logged (Warning: console and the `logs` table mirror) — redacted.
        Func<AiModelStatus> original = e2e.Ai.Status;
        e2e.Ai.Status = () => throw new AiServiceException(AiServiceErrorKind.Unavailable,
            $"upstream echoed key={e2e.ApiKey} password={e2e.DbPassword} master={e2e.MasterKeyValue}");
        try
        {
            Assert.Equal(HttpStatusCode.OK, (await e2e.Client.GetAsync("/api/v1/model/status")).StatusCode);
        }
        finally
        {
            e2e.Ai.Status = original;
        }

        Assert.Contains(e2e.Logs.Entries, entry => entry.Contains("upstream echoed key=***", StringComparison.Ordinal));
        await e2e.WaitForAsync("SELECT count(*) FROM logs WHERE service = 'backend' AND message LIKE '%upstream echoed key=***%'", 1);

        string[] secrets = [e2e.ApiKey, e2e.DbPassword, e2e.MasterKeyValue];
        Assert.DoesNotContain(e2e.Logs.Entries, entry => secrets.Any(s => entry.Contains(s, StringComparison.Ordinal)));
        foreach (string secret in secrets)
        {
            Assert.Equal(0L, await e2e.ScalarAsync<long>(
                "SELECT count(*) FROM logs WHERE strpos(message, @s) > 0 OR strpos(coalesce(context::text, ''), @s) > 0", ("s", secret)));
        }
    }

    // ------------------------------------------------------------------ health

    [Fact]
    public async Task Readiness_is_healthy_with_the_database_and_the_ai_service_up()
    {
        foreach (string path in new[] { "/health", "/health/ready" })
        {
            HttpResponseMessage response = await e2e.Client.GetAsync(path);

            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            Assert.Equal("Healthy", doc.RootElement.GetProperty("status").GetString());
            Assert.Equal("Healthy", EntryStatus(doc, "database"));
            Assert.Equal("Healthy", EntryStatus(doc, "ai_service"));
        }
    }

    [Fact]
    public async Task Readiness_follows_the_ai_service_down_and_back_up_while_liveness_stays_up()
    {
        e2e.Ai.Health = () => throw StubAiClient.Unreachable();
        try
        {
            HttpResponseMessage down = await e2e.Client.GetAsync("/health/ready");

            Assert.Equal(HttpStatusCode.ServiceUnavailable, down.StatusCode);
            using JsonDocument doc = JsonDocument.Parse(await down.Content.ReadAsStringAsync());
            Assert.Equal("Unhealthy", EntryStatus(doc, "ai_service"));
            Assert.Equal("Healthy", EntryStatus(doc, "database"));
            Assert.Equal(HttpStatusCode.OK, (await e2e.Client.GetAsync("/health/live")).StatusCode);
        }
        finally
        {
            e2e.Ai.Health = () => new AiHealthResponse(AiHealthResponse.Ok);
        }

        Assert.Equal(HttpStatusCode.OK, (await e2e.Client.GetAsync("/health/ready")).StatusCode);
    }

    [Fact]
    public async Task Readiness_reports_a_database_the_host_cannot_use_while_liveness_stays_up()
    {
        // The same server, but a database that does not exist: the connection is refused at login.
        string missingDb = new NpgsqlConnectionStringBuilder(e2e.ConnectionString) { Database = "missing_db", Timeout = 3 }.ConnectionString;
        await using WebApplicationFactory<Program> factory = TestApiFactory.Create("Production", [("SOLIA_DB_CONNECTION", missingDb)],
            builder => builder.ConfigureTestServices(services => services.AddSingleton<ISoliaAiClient>(new StubAiClient())));
        HttpClient client = factory.CreateClient();

        HttpResponseMessage ready = await client.GetAsync("/health/ready");

        Assert.Equal(HttpStatusCode.ServiceUnavailable, ready.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await ready.Content.ReadAsStringAsync());
        Assert.Equal("Unhealthy", EntryStatus(doc, "database"));
        Assert.Equal("Healthy", EntryStatus(doc, "ai_service"));
        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/health/live")).StatusCode);
    }

    // ------------------------------------------------------------------ helpers

    private AiPredictRequest[] PredictRequestsFor(string mint)
    {
        lock (e2e.Ai.PredictRequests)
        {
            return e2e.Ai.PredictRequests.Where(r => r.Mint == mint).ToArray();
        }
    }

    private static string EntryStatus(JsonDocument doc, string entry) =>
        doc.RootElement.GetProperty("entries").GetProperty(entry).GetProperty("status").GetString()!;

    /// <summary>An object's property names, sorted and comma-separated.</summary>
    private static string Fields(JsonElement element) =>
        string.Join(',', element.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal));

    private static async Task AssertEnvelopeAsync(HttpResponseMessage response, HttpStatusCode status, string code)
    {
        Assert.Equal(status, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("error", Fields(doc.RootElement));
        Assert.Equal("code,message", Fields(doc.RootElement.GetProperty("error")));
        Assert.Equal(code, doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }
}
