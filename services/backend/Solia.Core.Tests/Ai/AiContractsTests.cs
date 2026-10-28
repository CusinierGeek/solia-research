using System.Text.Json;
using System.Text.Json.Nodes;
using Solia.Core.Ai;

namespace Solia.Core.Tests.Ai;

public class AiContractsTests
{
    [Fact]
    public void Vector_request_serializes_snake_case_and_keeps_feature_names()
    {
        AiPredictRequest request = AiPredictRequest.ForVector(
            "So1abc", new Dictionary<string, double?> { ["volume_1m"] = 2.5, ["HolderCount"] = null }, ["1m", "5m"]);

        JsonNode json = JsonSerializer.SerializeToNode(request, AiJson.Options)!;

        Assert.Equal("So1abc", (string?)json["mint"]);
        Assert.Equal(2.5, (double?)json["features"]!["volume_1m"]);
        Assert.True(json["features"]!.AsObject().ContainsKey("HolderCount")); // dictionary keys are not renamed
        Assert.Null(json["features"]!["HolderCount"]);
        Assert.Equal(["1m", "5m"], json["horizons"]!.AsArray().Select(h => (string?)h));
    }

    [Fact]
    public void Window_request_serializes_as_a_list_of_rows_and_omits_null_horizons()
    {
        AiPredictRequest request = AiPredictRequest.ForWindow(null,
        [
            new FeatureRow(new Dictionary<string, double?> { ["price"] = 1.0 }, "pumpfun"),
            new FeatureRow(new Dictionary<string, double?> { ["price"] = 1.1 }, "pumpfun"),
        ]);

        JsonObject json = JsonSerializer.SerializeToNode(request, AiJson.Options)!.AsObject();

        Assert.Equal(2, json["features"]!.AsArray().Count);
        Assert.Equal(1.1, (double?)json["features"]![1]!["price"]);
        Assert.False(json.ContainsKey("horizons")); // null = all horizons; an explicit [] would be a 422
        Assert.False(json.ContainsKey("mint"));
    }

    [Fact]
    public void Window_rows_carry_the_raw_venue_string_beside_numbers()
    {
        AiPredictRequest request = AiPredictRequest.ForWindow("m",
        [
            new FeatureRow(new Dictionary<string, double?> { ["price"] = 1.5, ["burned_liquidity"] = null }, "pumpswap"),
        ]);

        JsonObject row = JsonSerializer.SerializeToNode(request, AiJson.Options)!["features"]![0]!.AsObject();

        Assert.Equal(1.5, (double?)row["price"]);
        Assert.True(row.ContainsKey("burned_liquidity"));
        Assert.Null(row["burned_liquidity"]);
        Assert.Equal("pumpswap", (string?)row["venue"]); // one-hot encoded by the ai-service
        Assert.Equal(["price", "burned_liquidity", "venue"], row.Select(kv => kv.Key)); // one flat object
    }

    [Fact]
    public void An_unknown_venue_is_still_sent_as_null_so_every_row_carries_it()
    {
        AiPredictRequest request = AiPredictRequest.ForWindow("m",
            [new FeatureRow(new Dictionary<string, double?> { ["price"] = 1.0 }, null)]);

        JsonObject row = JsonSerializer.SerializeToNode(request, AiJson.Options)!["features"]![0]!.AsObject();

        Assert.True(row.ContainsKey("venue")); // null = venue_other, as in training
        Assert.Null(row["venue"]);
    }

    [Fact]
    public void A_feature_row_rejects_venue_among_its_numbers_and_null_numbers()
    {
        Assert.Throws<ArgumentException>(() => new FeatureRow(new Dictionary<string, double?> { ["venue"] = 1.0 }, "pumpfun"));
        Assert.Throws<ArgumentNullException>(() => new FeatureRow(null!, "pumpfun"));
    }

    [Fact]
    public void Request_rejects_empty_horizons_and_empty_window()
    {
        var features = new Dictionary<string, double?> { ["price"] = 1.0 };

        Assert.Throws<ArgumentException>(() => AiPredictRequest.ForVector("m", features, []));
        Assert.Throws<ArgumentException>(() => AiPredictRequest.ForWindow("m", Array.Empty<FeatureRow>()));
        Assert.Throws<ArgumentNullException>(() => AiPredictRequest.ForWindow("m", null!));
        Assert.Throws<ArgumentNullException>(() => AiPredictRequest.ForVector("m", null!));
    }

    [Fact]
    public void Predict_response_deserializes_from_the_ai_service_shape()
    {
        AiPredictResponse response = JsonSerializer.Deserialize<AiPredictResponse>(AiClientTestHost.PredictJson, AiJson.Options)!;

        Assert.Equal("So1abc", response.Mint);
        Assert.Equal("v3", response.ModelVersion);
        AiPrediction p = Assert.Single(response.Predictions);
        Assert.Equal(new AiPrediction("1m", 0.72, 0.08, 0.35, 81.5), p);
    }

    [Fact]
    public void Model_status_deserializes_with_metrics_and_missing_counters()
    {
        const string json = """
            {"champion_version":"v3","created_at":"2026-10-01T12:00:00+00:00","param_count":123456,
             "metrics":{"auc_pump_1m":0.71,"per_horizon":{"1m":{"brier":0.2}}}}
            """;

        AiModelStatus status = JsonSerializer.Deserialize<AiModelStatus>(json, AiJson.Options)!;

        Assert.Equal("v3", status.ChampionVersion);
        Assert.Equal(new DateTimeOffset(2026, 10, 1, 12, 0, 0, TimeSpan.Zero), status.CreatedAt);
        Assert.Equal(123456, status.ParamCount);
        Assert.Equal(0.71, status.Metrics!["auc_pump_1m"].GetDouble());
        Assert.Equal(JsonValueKind.Object, status.Metrics["per_horizon"].ValueKind);
        Assert.Null(status.TradesSinceLastTrain);
        Assert.Null(status.RetrainThreshold);
    }

    [Fact]
    public void Model_status_reads_trade_counters_when_present()
    {
        const string json = """
            {"champion_version":"v3","created_at":null,"metrics":{},"trades_since_last_train":142,"retrain_threshold":500}
            """;

        AiModelStatus status = JsonSerializer.Deserialize<AiModelStatus>(json, AiJson.Options)!;

        Assert.Null(status.CreatedAt);
        Assert.Equal(142, status.TradesSinceLastTrain);
        Assert.Equal(500, status.RetrainThreshold);
    }

    [Theory]
    [InlineData("""{"mint":null,"predictions":[]}""")]
    [InlineData("""{"mint":null,"model_version":"v3","predictions":null}""")]
    [InlineData("""{"mint":null,"model_version":"v3","predictions":[{"pump_prob":0.1,"rug_prob":0.1,"expected_return":0,"confidence":1}]}""")]
    public void Predict_response_rejects_missing_or_null_required_fields(string json)
    {
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<AiPredictResponse>(json, AiJson.Options));
    }

    [Theory]
    [InlineData("""{"metrics":{}}""")]
    [InlineData("""{"champion_version":null}""")]
    public void Model_status_rejects_a_missing_or_null_champion_version(string json)
    {
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<AiModelStatus>(json, AiJson.Options));
    }

    [Fact]
    public void Retrain_response_requires_the_training_event_id()
    {
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<AiRetrainResponse>("{}", AiJson.Options));
    }

    [Fact]
    public void Retrain_contract_round_trips_snake_case()
    {
        Assert.Equal("""{"trigger":"manual"}""",
            JsonSerializer.Serialize(new AiRetrainRequest(AiRetrainRequest.Manual), AiJson.Options));
        Assert.Equal(42, JsonSerializer.Deserialize<AiRetrainResponse>("""{"training_event_id":42}""", AiJson.Options)!.TrainingEventId);
    }
}
