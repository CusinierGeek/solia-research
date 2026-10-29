using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using Polly.CircuitBreaker;
using Polly.Timeout;
using Solia.Core.Http;

namespace Solia.Core.Ai;

/// <summary>
/// The backend's only door to the internal ai-service (docs/API-CONTRACTS.md, "ai-service"). Returns
/// predictions only — trade decisions remain the bot's policy layer. Every failure surfaces as an
/// <see cref="AiServiceException"/>, never as a raw <see cref="HttpRequestException"/>.
/// </summary>
public interface ISoliaAiClient
{
    /// <summary><c>POST /predict</c>: score a token's features.</summary>
    Task<AiPredictResponse> PredictAsync(AiPredictRequest request, CancellationToken cancellationToken = default);

    /// <summary><c>GET /model/status</c>: the serving champion and its metrics.</summary>
    Task<AiModelStatus> GetModelStatusAsync(CancellationToken cancellationToken = default);

    /// <summary><c>POST /model/retrain</c> with trigger <c>manual</c>. Not idempotent, so never retried.</summary>
    Task<AiRetrainResponse> TriggerRetrainAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// <c>GET /health</c>: the ai-service's liveness, for the backend's readiness check. A single attempt (bounded
    /// by <c>AttemptTimeout</c>) that the circuit breaker neither counts nor blocks.
    /// </summary>
    Task<AiHealthResponse> CheckHealthAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Typed-<see cref="HttpClient"/> implementation of <see cref="ISoliaAiClient"/>. Retries, timeouts and the
/// circuit breaker live in the resilience pipeline registered by <c>AddSoliaAiClient</c>; this class only
/// builds requests and maps outcomes.
/// </summary>
internal sealed class SoliaAiClient(HttpClient http) : ISoliaAiClient
{
    // Relative (no leading slash) so a base address with a path prefix is kept.
    internal const string PredictPath = "predict";
    internal const string ModelStatusPath = "model/status";
    internal const string RetrainPath = "model/retrain";
    internal const string HealthPath = "health";

    /// <summary>Set on a request the retry strategy must not repeat (the call has side effects).</summary>
    internal static readonly HttpRequestOptionsKey<bool> NonIdempotent = new("Solia.Ai.NonIdempotent");

    /// <summary>Set on a probe: never retried, ignored by the circuit breaker.</summary>
    internal static readonly HttpRequestOptionsKey<bool> Probe = new("Solia.Ai.Probe");

    public Task<AiPredictResponse> PredictAsync(AiPredictRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        // Inference is a pure function of the request, so it is safe to retry despite being a POST.
        return SendAsync<AiPredictResponse>(HttpMethod.Post, PredictPath, request, idempotent: true, cancellationToken);
    }

    public Task<AiModelStatus> GetModelStatusAsync(CancellationToken cancellationToken = default) =>
        SendAsync<AiModelStatus>(HttpMethod.Get, ModelStatusPath, body: null, idempotent: true, cancellationToken);

    public Task<AiRetrainResponse> TriggerRetrainAsync(CancellationToken cancellationToken = default) =>
        SendAsync<AiRetrainResponse>(
            HttpMethod.Post, RetrainPath, new AiRetrainRequest(AiRetrainRequest.Manual), idempotent: false, cancellationToken);

    public Task<AiHealthResponse> CheckHealthAsync(CancellationToken cancellationToken = default) =>
        SendAsync<AiHealthResponse>(HttpMethod.Get, HealthPath, body: null, idempotent: true, cancellationToken, probe: true);

    private async Task<T> SendAsync<T>(
        HttpMethod method, string path, object? body, bool idempotent, CancellationToken cancellationToken, bool probe = false)
    {
        using var request = new HttpRequestMessage(method, path);
        if (body is not null)
        {
            request.Content = Serialize(body);
        }

        if (!idempotent)
        {
            request.Options.Set(NonIdempotent, true);
        }

        if (probe)
        {
            request.Options.Set(Probe, true);
        }

        HttpResponseMessage response;
        try
        {
            response = await http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw; // the caller gave up — not an ai-service failure
        }
        catch (TimeoutRejectedException ex)
        {
            throw Unavailable("ai-service call timed out", ex);
        }
        catch (BrokenCircuitException ex)
        {
            throw Unavailable("ai-service circuit is open", ex);
        }
        catch (HttpRequestException ex)
        {
            throw Unavailable($"ai-service unreachable: {ex.Message}", ex);
        }
        catch (OperationCanceledException ex)
        {
            throw Unavailable("ai-service call timed out", ex);
        }

        using (response)
        {
            if (!response.IsSuccessStatusCode)
            {
                throw await ToExceptionAsync(response, cancellationToken).ConfigureAwait(false);
            }

            try
            {
                T? payload = await response.Content
                    .ReadFromJsonAsync<T>(AiJson.Options, cancellationToken)
                    .ConfigureAwait(false);
                return payload ?? throw BadPayload(response, null);
            }
            catch (Exception ex) when (IsUnreadableBody(ex))
            {
                throw BadPayload(response, ex);
            }
        }
    }

    private static async Task<AiServiceException> ToExceptionAsync(
        HttpResponseMessage response, CancellationToken cancellationToken)
    {
        int status = (int)response.StatusCode;
        ApiError? error = null;
        try
        {
            error = (await response.Content
                .ReadFromJsonAsync<ApiErrorResponse>(AiJson.Options, cancellationToken)
                .ConfigureAwait(false))?.Error;
        }
        catch (Exception ex) when (IsUnreadableBody(ex))
        {
            // Not the documented envelope (e.g. a proxy's HTML page) — classify by status alone.
        }

        string message = error?.Message ?? $"ai-service returned {status} {response.ReasonPhrase}";
        return new AiServiceException(Classify(response.StatusCode, error?.Code), message, status, error?.Code);
    }

    /// <summary>
    /// Serializes eagerly, so a body System.Text.Json rejects (e.g. a NaN feature) fails here as an invalid
    /// request instead of inside the handler chain, where it would look like a transport failure and be
    /// retried and counted by the breaker.
    /// </summary>
    private static ByteArrayContent Serialize(object body)
    {
        byte[] json;
        try
        {
            json = JsonSerializer.SerializeToUtf8Bytes(body, body.GetType(), AiJson.Options);
        }
        catch (Exception ex) when (ex is JsonException or ArgumentException or NotSupportedException)
        {
            throw new AiServiceException(
                AiServiceErrorKind.InvalidRequest, $"the request could not be serialized: {ex.Message}", innerException: ex);
        }

        var content = new ByteArrayContent(json);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/json") { CharSet = "utf-8" };
        return content;
    }

    /// <summary>
    /// A body that is not the expected JSON: malformed or incomplete (<see cref="JsonException"/>), an
    /// unsupported type (<see cref="NotSupportedException"/>) or an invalid charset (<see cref="InvalidOperationException"/>).
    /// </summary>
    internal static bool IsUnreadableBody(Exception ex) =>
        ex is JsonException or NotSupportedException or InvalidOperationException;

    /// <summary>
    /// The 503 codes meaning "no usable champion" until a model is (re)trained and promoted: none registered
    /// (<c>no_champion</c>) or one trained on another feature spec (<c>champion_incompatible</c>).
    /// </summary>
    internal static bool IsNoChampionCode(string? code) => code is "no_champion" or "champion_incompatible";

    internal static AiServiceErrorKind Classify(HttpStatusCode status, string? code) => status switch
    {
        HttpStatusCode.UnprocessableEntity when code == "window_too_short" => AiServiceErrorKind.InsufficientHistory,
        HttpStatusCode.BadRequest or HttpStatusCode.UnprocessableEntity => AiServiceErrorKind.InvalidRequest,
        HttpStatusCode.ServiceUnavailable when IsNoChampionCode(code) => AiServiceErrorKind.NoChampion,
        HttpStatusCode.Conflict => AiServiceErrorKind.Busy,
        HttpStatusCode.ServiceUnavailable
            or HttpStatusCode.RequestTimeout
            or HttpStatusCode.TooManyRequests
            or HttpStatusCode.BadGateway
            or HttpStatusCode.GatewayTimeout => AiServiceErrorKind.Unavailable,
        _ => AiServiceErrorKind.UpstreamError,
    };

    private static AiServiceException Unavailable(string message, Exception inner) =>
        new(AiServiceErrorKind.Unavailable, message, innerException: inner);

    private static AiServiceException BadPayload(HttpResponseMessage response, Exception? inner) =>
        new(AiServiceErrorKind.UpstreamError, "ai-service returned an unreadable body",
            (int)response.StatusCode, innerException: inner);
}
