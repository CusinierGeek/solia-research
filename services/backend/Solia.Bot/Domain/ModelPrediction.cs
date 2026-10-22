namespace Solia.Bot.Domain;

/// <summary>
/// One prediction the ai-service produced for a token at a horizon — the signal the bot acts on.
/// <strong>Predictions only, no decisions</strong>: buy/sell/sizing/TP/SL are the policy layer's job
/// (see docs/ML-MODEL.md "Policy layer"), not fields here. Mirrors the <c>/predict</c> response in
/// docs/API-CONTRACTS.md. Probabilities/ratios are <see cref="double"/> (mirrors FeatureSnapshot);
/// money/price never appears here.
/// </summary>
public sealed record ModelPrediction
{
    public required string Mint { get; init; }

    /// <summary>When the prediction was produced (the event's logical time).</summary>
    public required DateTimeOffset Ts { get; init; }

    /// <summary>Horizon this prediction is for, e.g. "30s", "1m", "5m" (see API-CONTRACTS).</summary>
    public required string Horizon { get; init; }

    /// <summary>Probability the token pumps within the horizon, [0, 1].</summary>
    public required double PumpProb { get; init; }

    /// <summary>Probability the token rugs within the horizon, [0, 1].</summary>
    public required double RugProb { get; init; }

    /// <summary>Realized forward return the model expects over the horizon (e.g. 0.35 = +35%).</summary>
    public required double ExpectedReturn { get; init; }

    /// <summary>Calibrated confidence, [0, 100] (derived, not a learned head).</summary>
    public required double Confidence { get; init; }

    /// <summary>Recent price volatility the model saw; the policy layer may size TP/SL from it. Null when not derivable.</summary>
    public double? Volatility { get; init; }

    /// <summary>Champion model version that produced this prediction, for traceability.</summary>
    public string? ModelVersion { get; init; }

    /// <summary>
    /// Id of the <c>predictions</c> row this prediction was persisted as — the trade's
    /// <c>prediction_id</c> FK. Null when it was never persisted (e.g. a replay fed straight from the model).
    /// </summary>
    public long? PredictionId { get; init; }

    /// <summary>
    /// The numeric feature vector the prediction was computed from, keyed by feature name — carried
    /// for traceability/debugging only. The bot never derives decisions from it directly. Compared by
    /// content in <see cref="Equals(ModelPrediction?)"/> (a dictionary alone compares by reference).
    /// </summary>
    public IReadOnlyDictionary<string, double?> Features { get; init; } =
        new Dictionary<string, double?>();

    /// <summary>Value equality over every field, with <see cref="Features"/> compared by content.</summary>
    public bool Equals(ModelPrediction? other) =>
        other is not null
        && Mint == other.Mint
        && Ts == other.Ts
        && Horizon == other.Horizon
        && PumpProb.Equals(other.PumpProb)
        && RugProb.Equals(other.RugProb)
        && ExpectedReturn.Equals(other.ExpectedReturn)
        && Confidence.Equals(other.Confidence)
        && Volatility.Equals(other.Volatility)
        && ModelVersion == other.ModelVersion
        && PredictionId == other.PredictionId
        && FeaturesEqual(Features, other.Features);

    /// <summary>Consistent with <see cref="Equals(ModelPrediction?)"/>: order-independent over <see cref="Features"/>.</summary>
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Mint);
        hash.Add(Ts);
        hash.Add(Horizon);
        hash.Add(PumpProb);
        hash.Add(RugProb);
        hash.Add(ExpectedReturn);
        hash.Add(Confidence);
        hash.Add(Volatility);
        hash.Add(ModelVersion);
        hash.Add(PredictionId);
        int features = 0;
        foreach ((string name, double? value) in Features)
        {
            features ^= HashCode.Combine(name, value);
        }
        hash.Add(features);
        return hash.ToHashCode();
    }

    private static bool FeaturesEqual(IReadOnlyDictionary<string, double?> a, IReadOnlyDictionary<string, double?> b) =>
        ReferenceEquals(a, b)
        || (a.Count == b.Count
            && a.All(kv => b.TryGetValue(kv.Key, out double? v) && Nullable.Equals(kv.Value, v)));
}
