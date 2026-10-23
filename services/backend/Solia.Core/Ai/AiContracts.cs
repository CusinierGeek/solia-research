using System.Text.Json;
using System.Text.Json.Serialization;

namespace Solia.Core.Ai;

/// <summary>
/// The ai-service wire format: snake_case properties (<c>pump_prob</c>, <c>model_version</c>, …) as
/// FastAPI emits them. Dictionary keys (feature names, metric names) are left untouched. A response that
/// omits a <see cref="JsonRequiredAttribute"/> member, or sends null for a non-nullable one, fails to
/// deserialize instead of yielding a half-built record.
/// </summary>
public static class AiJson
{
    public static JsonSerializerOptions Options { get; } = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        RespectNullableAnnotations = true,
    };
}

/// <summary>
/// <c>POST /predict</c> request. <see cref="Features"/> is either the latest feature vector (one object,
/// a one-row window) or the token's recent window, oldest row first — build it with
/// <see cref="ForVector"/> / <see cref="ForWindow"/>. <see cref="Horizons"/> is optional (null = all);
/// when present it must be non-empty.
/// </summary>
public sealed record AiPredictRequest
{
    private AiPredictRequest(string? mint, object features, IReadOnlyList<string>? horizons)
    {
        if (horizons is { Count: 0 })
        {
            throw new ArgumentException("Horizons must be null (all) or non-empty.", nameof(horizons));
        }

        Mint = mint;
        Features = features;
        Horizons = horizons;
    }

    public string? Mint { get; }

    /// <summary>A feature vector or a window of them (serialized as its runtime type).</summary>
    public object Features { get; }

    public IReadOnlyList<string>? Horizons { get; }

    /// <summary>A request scored on the latest feature vector only.</summary>
    public static AiPredictRequest ForVector(
        string? mint, IReadOnlyDictionary<string, double?> features, IReadOnlyList<string>? horizons = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        return new AiPredictRequest(mint, features, horizons);
    }

    /// <summary>A request scored on the token's recent window (oldest row first, non-empty).</summary>
    public static AiPredictRequest ForWindow(
        string? mint, IReadOnlyList<IReadOnlyDictionary<string, double?>> window, IReadOnlyList<string>? horizons = null)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (window.Count == 0)
        {
            throw new ArgumentException("The feature window must be non-empty.", nameof(window));
        }

        return new AiPredictRequest(mint, window, horizons);
    }
}

/// <summary>One horizon's signal. The model predicts only — trade decisions stay in the bot's policy layer.</summary>
public sealed record AiPrediction(
    [property: JsonRequired] string Horizon,
    [property: JsonRequired] double PumpProb,
    [property: JsonRequired] double RugProb,
    [property: JsonRequired] double ExpectedReturn,
    [property: JsonRequired] double Confidence);

/// <summary><c>POST /predict</c> response: per-horizon predictions plus the champion version that produced them.</summary>
public sealed record AiPredictResponse(
    string? Mint,
    [property: JsonRequired] string ModelVersion,
    [property: JsonRequired] IReadOnlyList<AiPrediction> Predictions);

/// <summary>
/// <c>GET /model/status</c> response. Fields the ai-service does not (yet) report — e.g. the trade counters
/// the retrain orchestrator owns — are null.
/// </summary>
public sealed record AiModelStatus(
    [property: JsonRequired] string ChampionVersion,
    DateTimeOffset? CreatedAt,
    long? ParamCount,
    IReadOnlyDictionary<string, JsonElement>? Metrics,
    int? TradesSinceLastTrain,
    int? RetrainThreshold);

/// <summary><c>POST /model/retrain</c> request body.</summary>
public sealed record AiRetrainRequest(string Trigger)
{
    public const string Manual = "manual";
}

/// <summary><c>POST /model/retrain</c> 202 response: the audit row of the run that was kicked off.</summary>
public sealed record AiRetrainResponse([property: JsonRequired] long TrainingEventId);

/// <summary><c>GET /health</c> response: the ai-service's liveness (<c>{"status":"ok"}</c>).</summary>
public sealed record AiHealthResponse([property: JsonRequired] string Status)
{
    public const string Ok = "ok";

    public bool IsOk => string.Equals(Status, Ok, StringComparison.OrdinalIgnoreCase);
}
