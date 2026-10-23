using Solia.Core.Ai;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// An <see cref="ISoliaAiClient"/> whose answers a test sets up front; it records every predict request so the
/// feature window the backend built can be asserted. Callers in one test class run sequentially (xUnit), so
/// the mutable setup is safe on a shared fixture as long as each test sets what it relies on. Recording is
/// thread-safe: a hosted bot scores through the same stub while a test calls the endpoints.
/// </summary>
public sealed class StubAiClient : ISoliaAiClient
{
    public Func<AiPredictRequest, AiPredictResponse> Predict { get; set; } =
        r => new AiPredictResponse(r.Mint, "v3", [new AiPrediction("1m", 0.72, 0.08, 0.35, 81.5)]);

    public Func<AiModelStatus> Status { get; set; } =
        () => new AiModelStatus("v3", null, null, null, null, null);

    public Func<AiRetrainResponse> Retrain { get; set; } = () => new AiRetrainResponse(42);

    public Func<AiHealthResponse> Health { get; set; } = () => new AiHealthResponse(AiHealthResponse.Ok);

    public List<AiPredictRequest> PredictRequests { get; } = [];

    private int _calls;

    public int Calls => Volatile.Read(ref _calls);

    public Task<AiPredictResponse> PredictAsync(AiPredictRequest request, CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        lock (PredictRequests)
        {
            PredictRequests.Add(request);
        }

        return Task.FromResult(Predict(request));
    }

    public Task<AiModelStatus> GetModelStatusAsync(CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        return Task.FromResult(Status());
    }

    public Task<AiRetrainResponse> TriggerRetrainAsync(CancellationToken cancellationToken = default)
    {
        Interlocked.Increment(ref _calls);
        return Task.FromResult(Retrain());
    }

    public Task<AiHealthResponse> CheckHealthAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(Health()); // a probe, not counted in Calls: a test's own call count stays exact

    /// <summary>A 503 <c>no_champion</c> from the ai-service, as the real client maps it.</summary>
    public static AiServiceException NoChampion() =>
        new(AiServiceErrorKind.NoChampion, "no champion model registered", 503, "no_champion");

    /// <summary>A 422 <c>window_too_short</c> from the ai-service: the window is under the champion's <c>min_seq_len</c>.</summary>
    public static AiServiceException WindowTooShort() =>
        new(AiServiceErrorKind.InsufficientHistory, "the feature window holds 3 row(s); the champion needs at least 8", 422, "window_too_short");

    /// <summary>The ai-service could not be reached (no response), carrying a message that must not leak.</summary>
    public static AiServiceException Unreachable() =>
        new(AiServiceErrorKind.Unavailable, "ai-service unreachable: connection refused (10.0.0.7:8000)");
}
