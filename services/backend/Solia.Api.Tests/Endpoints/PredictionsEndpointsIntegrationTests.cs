using System.Text.Json;
using static Solia.Api.Tests.Endpoints.ReadApiFixture;

namespace Solia.Api.Tests.Endpoints;

// The prediction reads, against the same seeded database as the other read endpoints (one container).
public partial class ReadEndpointsIntegrationTests
{
    /// <summary>Each item as <c>mint/horizon@minutes-after-T0</c>, the fixture's own key.</summary>
    private static string[] Predictions(JsonElement array) =>
        array.EnumerateArray()
            .Select(e => $"{e.GetProperty("mint").GetString()}/{e.GetProperty("horizon").GetString()}" +
                         $"@{(e.GetProperty("ts").GetDateTimeOffset() - T0).TotalMinutes}")
            .ToArray();

    [Fact]
    public async Task Live_predictions_are_the_latest_per_tracked_token_and_horizon_newest_first()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/predictions/live");

        // m1's older 1m row is superseded; m4 (inactive) and m5 (not tracked) are left out despite being newest;
        // m2's two rows tie on ts and list shortest horizon first (not "1m" < "30s" as strings).
        Assert.Equal(["m2/30s@20", "m2/1m@20", "m3/1m@15", "m1/1m@10", "m1/5m@5"], Predictions(doc.RootElement));
    }

    [Fact]
    public async Task A_live_prediction_has_the_documented_shape()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/predictions/live");
        JsonElement m1 = doc.RootElement[3];
        JsonElement modelless = doc.RootElement[4];

        string[] fields = m1.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal).ToArray();
        Assert.Equal(
            ["confidence", "horizon", "id", "mint", "model_version", "position_size", "pump_prob", "rug_prob", "stop_loss",
             "strategy", "symbol", "take_profit", "ts"],
            fields);
        Assert.True(m1.GetProperty("id").GetInt64() > 0);
        // The live symbol (tokens_live), as /tokens/live shows it.
        Assert.Equal("M1-LIVE", m1.GetProperty("symbol").GetString());
        Assert.Equal(T0.AddMinutes(10), m1.GetProperty("ts").GetDateTimeOffset());
        Assert.Equal(0.9, m1.GetProperty("pump_prob").GetDouble());
        Assert.Equal(0.05, m1.GetProperty("rug_prob").GetDouble());
        Assert.Equal("buy", m1.GetProperty("strategy").GetString());
        Assert.Equal(80, m1.GetProperty("confidence").GetDouble());
        Assert.Equal(2.5m, m1.GetProperty("position_size").GetDecimal());
        Assert.Equal(TinyPrice, m1.GetProperty("take_profit").GetDecimal());
        Assert.Equal(0.0001m, m1.GetProperty("stop_loss").GetDecimal());
        Assert.Equal("v1", m1.GetProperty("model_version").GetString());
        Assert.Equal(JsonValueKind.Null, modelless.GetProperty("model_version").ValueKind);
        Assert.Equal(JsonValueKind.Null, modelless.GetProperty("rug_prob").ValueKind);
    }

    [Theory]
    [InlineData("sort=pump_prob", new[] { "m1/1m@10", "m2/30s@20", "m3/1m@15", "m1/5m@5", "m2/1m@20" })]
    [InlineData("sort=rug_prob", new[] { "m2/1m@20", "m2/30s@20", "m3/1m@15", "m1/1m@10", "m1/5m@5" })]
    [InlineData("sort=confidence", new[] { "m3/1m@15", "m1/1m@10", "m2/30s@20", "m2/1m@20", "m1/5m@5" })]
    [InlineData("sort=ts", new[] { "m2/30s@20", "m2/1m@20", "m3/1m@15", "m1/1m@10", "m1/5m@5" })]
    [InlineData("horizon=5m", new[] { "m1/5m@5" })]
    [InlineData("horizon=30s", new[] { "m2/30s@20" })]
    [InlineData("horizon=1h", new string[0])]
    [InlineData("limit=2", new[] { "m2/30s@20", "m2/1m@20" })]
    [InlineData("horizon=1m&sort=pump_prob&limit=2", new[] { "m1/1m@10", "m3/1m@15" })]
    public async Task Live_predictions_sort_descending_with_missing_values_last_and_filter(string query, string[] expected)
    {
        using JsonDocument doc = await GetJsonAsync($"/api/v1/predictions/live?{query}");

        Assert.Equal(expected, Predictions(doc.RootElement));
    }

    [Fact]
    public async Task Prediction_history_pages_newest_first_with_the_total()
    {
        using JsonDocument all = await GetJsonAsync("/api/v1/predictions/history");
        using JsonDocument second = await GetJsonAsync("/api/v1/predictions/history?page=2&pageSize=3");
        using JsonDocument last = await GetJsonAsync("/api/v1/predictions/history?page=3&pageSize=3");
        using JsonDocument beyond = await GetJsonAsync("/api/v1/predictions/history?page=4&pageSize=3");

        // m2's rows tie on ts: the later insert (higher id) first.
        Assert.Equal(["m5/1m@40", "m4/1m@30", "m2/30s@20", "m2/1m@20", "m3/1m@15", "m1/1m@10", "m1/5m@5", "m1/1m@0"],
            Predictions(all.RootElement.GetProperty("items")));
        AssertPage(all.RootElement, 1, 50, 8);
        Assert.Equal(["m2/1m@20", "m3/1m@15", "m1/1m@10"], Predictions(second.RootElement.GetProperty("items")));
        AssertPage(second.RootElement, 2, 3, 8);
        Assert.Equal(["m1/5m@5", "m1/1m@0"], Predictions(last.RootElement.GetProperty("items")));
        Assert.Empty(beyond.RootElement.GetProperty("items").EnumerateArray());
        AssertPage(beyond.RootElement, 4, 3, 8);
        Assert.Equal("v1", all.RootElement.GetProperty("items")[0].GetProperty("model_version").GetString());
        // History keeps the registry's symbol (tokens_history), live or not.
        Assert.Equal("M1", all.RootElement.GetProperty("items")[5].GetProperty("symbol").GetString());
    }

    [Theory]
    [InlineData("mint=m1", new[] { "m1/1m@10", "m1/5m@5", "m1/1m@0" })]
    [InlineData("mint=M1", new string[0])]
    [InlineData("horizon=5m", new[] { "m1/5m@5" })]
    [InlineData("strategy=sell", new[] { "m2/1m@20" })]
    [InlineData("strategy=HOLD", new[] { "m1/5m@5" })]
    [InlineData("horizon=30s", new[] { "m2/30s@20" })]
    [InlineData("from=2026-01-10T00:10:00Z&to=2026-01-10T00:20:00Z", new[] { "m2/30s@20", "m2/1m@20", "m3/1m@15", "m1/1m@10" })]
    [InlineData("mint=m1&horizon=1m&strategy=buy&to=2026-01-10T00:05:00Z", new[] { "m1/1m@0" })]
    public async Task Prediction_history_filters(string query, string[] expected)
    {
        using JsonDocument doc = await GetJsonAsync($"/api/v1/predictions/history?{query}");

        Assert.Equal(expected, Predictions(doc.RootElement.GetProperty("items")));
        Assert.Equal(expected.Length, doc.RootElement.GetProperty("total").GetInt64());
    }
}
