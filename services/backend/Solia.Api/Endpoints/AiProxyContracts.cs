using System.Text.Json;

namespace Solia.Api.Endpoints;

// Request/response DTOs of the public AI-proxy endpoints (docs/API-CONTRACTS.md), deliberately separate from
// the internal ai-service wire records (Solia.Core.Ai) so the public shape never follows an upstream change.
// Serialized snake_case by the host's JSON options.

/// <summary><c>POST /predict</c> body: the token to score and, optionally, the horizons (null = all).</summary>
public sealed record PredictRequestDto(string? Mint, IReadOnlyList<string>? Horizons);

/// <summary>One horizon's signal. Predictions only — trade decisions stay in the bot's policy layer.</summary>
public sealed record PredictionDto(string Horizon, double PumpProb, double RugProb, double ExpectedReturn, double Confidence);

/// <summary><c>POST /predict</c> response, stamped with the champion version that produced it.</summary>
public sealed record PredictResponse(string Mint, string ModelVersion, IReadOnlyList<PredictionDto> Predictions);

/// <summary>
/// <c>GET /model/status</c>: the serving champion (from the ai-service; null when it is down or has none yet)
/// plus the closed paper trades since the last training run and the threshold that triggers the next one.
/// </summary>
public sealed record ModelStatusResponse(
    string? ChampionVersion,
    DateTimeOffset? CreatedAt,
    IReadOnlyDictionary<string, JsonElement> Metrics,
    long TradesSinceLastTrain,
    int RetrainThreshold);

/// <summary><c>POST /model/retrain</c> 202 body: the audit row of the run that was kicked off.</summary>
public sealed record RetrainResponse(long TrainingEventId);
