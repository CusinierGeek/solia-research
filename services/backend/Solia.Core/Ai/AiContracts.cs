using System.Text.Json;
using System.Text.Json.Serialization;
using Solia.Core.Data.Entities;

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
        string? mint, IReadOnlyList<FeatureRow> window, IReadOnlyList<string>? horizons = null)
    {
        ArgumentNullException.ThrowIfNull(window);
        if (window.Count == 0)
        {
            throw new ArgumentException("The feature window must be non-empty.", nameof(window));
        }

        return new AiPredictRequest(mint, window, horizons);
    }
}

/// <summary>
/// One row of a <c>/predict</c> window: the numeric model inputs, keyed by the ai-service's <c>FEATURE_COLUMNS</c>
/// names (null = imputed server-side), plus the raw <see cref="Venue"/>, which the ai-service one-hot encodes itself.
/// Serialized as one flat JSON object; <c>venue</c> is always written (null = the <c>venue_other</c> bucket, as in
/// training) so every row of a window carries it.
/// </summary>
[JsonConverter(typeof(FeatureRowJsonConverter))]
public sealed record FeatureRow
{
    public const string VenueKey = "venue";

    public FeatureRow(IReadOnlyDictionary<string, double?> numeric, string? venue)
    {
        ArgumentNullException.ThrowIfNull(numeric);
        if (numeric.ContainsKey(VenueKey))
        {
            throw new ArgumentException($"'{VenueKey}' is not numeric: pass it as the venue.", nameof(numeric));
        }

        Numeric = numeric;
        Venue = venue;
    }

    public IReadOnlyDictionary<string, double?> Numeric { get; }

    public string? Venue { get; }
}

internal sealed class FeatureRowJsonConverter : JsonConverter<FeatureRow>
{
    public override FeatureRow Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options) =>
        throw new NotSupportedException("Feature rows are only sent to the ai-service.");

    public override void Write(Utf8JsonWriter writer, FeatureRow value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        foreach ((string name, double? number) in value.Numeric)
        {
            if (number is { } n)
            {
                writer.WriteNumber(name, n);
            }
            else
            {
                writer.WriteNull(name);
            }
        }

        writer.WriteString(FeatureRow.VenueKey, value.Venue);
        writer.WriteEndObject();
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
    public const string Manual = TrainingTriggers.Manual;
}

/// <summary><c>POST /model/retrain</c> 202 response: the audit row of the run that was kicked off.</summary>
public sealed record AiRetrainResponse([property: JsonRequired] long TrainingEventId);

/// <summary><c>GET /health</c> response: the ai-service's liveness (<c>{"status":"ok"}</c>).</summary>
public sealed record AiHealthResponse([property: JsonRequired] string Status)
{
    public const string Ok = "ok";

    public bool IsOk => string.Equals(Status, Ok, StringComparison.OrdinalIgnoreCase);
}
