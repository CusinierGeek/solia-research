using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Polly.Timeout;

namespace Solia.Dashboard.Api;

/// <summary>
/// The dashboard's only door to the backend's read endpoints (docs/API-CONTRACTS.md, base <c>/api/v1</c>). Read-only
/// by design: no <c>/predict</c>, no <c>/model/retrain</c>. Every failure surfaces as a <see cref="BackendException"/>
/// (<see cref="BackendErrorException"/> or <see cref="BackendUnavailableException"/>), except the caller's own
/// cancellation.
/// </summary>
public interface ISoliaApiClient
{
    /// <summary><c>GET /tokens/live</c>: the tracked tokens with their latest snapshot.</summary>
    Task<IReadOnlyList<LiveTokenDto>> GetLiveTokensAsync(LiveTokensQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /tokens/history</c>: the token registry, newest first.</summary>
    Task<PagedResult<TokenHistoryDto>> GetTokenHistoryAsync(TokenHistoryQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /predictions/live</c>: the latest prediction per (tracked token, horizon).</summary>
    Task<IReadOnlyList<PredictionRecordDto>> GetLivePredictionsAsync(LivePredictionsQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /predictions/history</c>: predictions, newest first.</summary>
    Task<PagedResult<PredictionRecordDto>> GetPredictionHistoryAsync(PredictionHistoryQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /trades/history</c>: paper trades, most recently opened first.</summary>
    Task<PagedResult<TradeDto>> GetTradeHistoryAsync(TradeHistoryQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// <c>GET /stats/winrate</c> over <paramref name="window"/> (backend default: all) of <paramref name="session"/>
    /// (backend default: the current session).
    /// </summary>
    Task<WinRateResponse> GetWinRateAsync(
        StatsWindow? window = null, SessionSelection? session = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// <c>GET /stats/performance</c>: the equity curve's latest <paramref name="limit"/> points of <paramref name="session"/>
    /// (backend default: the current session).
    /// </summary>
    Task<PerformanceResponse> GetPerformanceAsync(
        int? limit = null, SessionSelection? session = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /bot/sessions</c>: the bot's sessions, newest first.</summary>
    Task<IReadOnlyList<BotSessionDto>> GetBotSessionsAsync(CancellationToken cancellationToken = default);

    /// <summary><c>GET /logs/bot</c>: recent logs, newest first.</summary>
    Task<IReadOnlyList<LogDto>> GetBotLogsAsync(BotLogsQuery? query = null, CancellationToken cancellationToken = default);

    /// <summary><c>GET /model/status</c>: the champion and the retrain progress.</summary>
    Task<ModelStatusResponse> GetModelStatusAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Typed-<see cref="HttpClient"/> implementation of <see cref="ISoliaApiClient"/>. Timeouts and the retry live in
/// the resilience pipeline registered by <c>AddSoliaApiClient</c>; this class only builds requests and maps outcomes.
/// </summary>
internal sealed class SoliaApiClient(HttpClient http) : ISoliaApiClient
{
    // Relative (no leading slash) so the base address's /api/v1/ prefix is kept.
    internal const string LiveTokensPath = "tokens/live";
    internal const string TokenHistoryPath = "tokens/history";
    internal const string LivePredictionsPath = "predictions/live";
    internal const string PredictionHistoryPath = "predictions/history";
    internal const string TradeHistoryPath = "trades/history";
    internal const string WinRatePath = "stats/winrate";
    internal const string PerformancePath = "stats/performance";
    internal const string BotSessionsPath = "bot/sessions";
    internal const string BotLogsPath = "logs/bot";
    internal const string ModelStatusPath = "model/status";

    public Task<IReadOnlyList<LiveTokenDto>> GetLiveTokensAsync(LiveTokensQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<IReadOnlyList<LiveTokenDto>>(QueryValues.Build(LiveTokensPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<PagedResult<TokenHistoryDto>> GetTokenHistoryAsync(TokenHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<PagedResult<TokenHistoryDto>>(QueryValues.Build(TokenHistoryPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<IReadOnlyList<PredictionRecordDto>> GetLivePredictionsAsync(LivePredictionsQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<IReadOnlyList<PredictionRecordDto>>(QueryValues.Build(LivePredictionsPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<PagedResult<PredictionRecordDto>> GetPredictionHistoryAsync(PredictionHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<PagedResult<PredictionRecordDto>>(QueryValues.Build(PredictionHistoryPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<PagedResult<TradeDto>> GetTradeHistoryAsync(TradeHistoryQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<PagedResult<TradeDto>>(QueryValues.Build(TradeHistoryPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<WinRateResponse> GetWinRateAsync(
        StatsWindow? window = null, SessionSelection? session = null, CancellationToken cancellationToken = default) =>
        GetAsync<WinRateResponse>(
            QueryValues.Build(WinRatePath, [("window", window is { } w ? QueryValues.Format(w) : null), ("session", session?.Wire)]),
            cancellationToken);

    public Task<PerformanceResponse> GetPerformanceAsync(
        int? limit = null, SessionSelection? session = null, CancellationToken cancellationToken = default) =>
        GetAsync<PerformanceResponse>(
            QueryValues.Build(PerformancePath, [("limit", QueryValues.Format(limit)), ("session", session?.Wire)]), cancellationToken);

    public Task<IReadOnlyList<BotSessionDto>> GetBotSessionsAsync(CancellationToken cancellationToken = default) =>
        GetAsync<IReadOnlyList<BotSessionDto>>(BotSessionsPath, cancellationToken);

    public Task<IReadOnlyList<LogDto>> GetBotLogsAsync(BotLogsQuery? query = null, CancellationToken cancellationToken = default) =>
        GetAsync<IReadOnlyList<LogDto>>(QueryValues.Build(BotLogsPath, (query ?? new()).ToParameters()), cancellationToken);

    public Task<ModelStatusResponse> GetModelStatusAsync(CancellationToken cancellationToken = default) =>
        GetAsync<ModelStatusResponse>(ModelStatusPath, cancellationToken);

    private async Task<T> GetAsync<T>(string pathAndQuery, CancellationToken cancellationToken)
    {
        HttpResponseMessage response;
        try
        {
            response = await http.GetAsync(pathAndQuery, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw; // the caller gave up (e.g. the component was disposed) — not a backend failure
        }
        catch (HttpRequestException ex) when (SoliaApiClientServiceCollectionExtensions.IsTooLarge(ex))
        {
            // The backend answered, but with a body past the buffering cap (its status is gone with the response).
            throw new BackendErrorException(0, BackendErrorException.InvalidResponseCode, "The backend returned an oversized response.", ex);
        }
        catch (Exception ex) when (ex is HttpRequestException or TimeoutRejectedException or OperationCanceledException)
        {
            throw new BackendUnavailableException(ex);
        }

        using (response)
        {
            if (!response.IsSuccessStatusCode)
            {
                throw await ToExceptionAsync(response, cancellationToken).ConfigureAwait(false);
            }

            try
            {
                T? payload = await response.Content.ReadFromJsonAsync<T>(SoliaApiJson.Options, cancellationToken).ConfigureAwait(false);
                return payload ?? throw InvalidResponse(response, null);
            }
            catch (Exception ex) when (IsUnreadableBody(ex))
            {
                throw InvalidResponse(response, ex);
            }
        }
    }

    private static async Task<BackendException> ToExceptionAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        ApiErrorBody? error = null;
        try
        {
            error = (await response.Content.ReadFromJsonAsync<ApiErrorEnvelope>(SoliaApiJson.Options, cancellationToken)
                .ConfigureAwait(false))?.Error;
        }
        catch (Exception ex) when (IsUnreadableBody(ex))
        {
            // Not the documented envelope (e.g. a proxy's HTML page) — classify by status alone.
        }

        if (error?.Code is not { Length: > 0 } code)
        {
            // Without the envelope the backend itself did not answer: a gateway in front of a stopped container, or a
            // proxy that timed the request out (408) — transient either way.
            if (response.StatusCode is HttpStatusCode.RequestTimeout or HttpStatusCode.BadGateway
                or HttpStatusCode.ServiceUnavailable or HttpStatusCode.GatewayTimeout)
            {
                return new BackendUnavailableException();
            }

            int status = (int)response.StatusCode;
            return new BackendErrorException(status, CodeFor(response.StatusCode), $"The backend answered {status}.");
        }

        string message = string.IsNullOrWhiteSpace(error.Message) ? $"The backend answered {(int)response.StatusCode}." : error.Message;
        return new BackendErrorException((int)response.StatusCode, code, message);
    }

    /// <summary>The documented code for a status (docs/API-CONTRACTS.md, "Error shape"), for a body without the envelope.</summary>
    internal static string CodeFor(HttpStatusCode status) => status switch
    {
        HttpStatusCode.BadRequest or HttpStatusCode.UnprocessableEntity => "invalid_request",
        HttpStatusCode.Unauthorized => "unauthorized",
        HttpStatusCode.Forbidden => "forbidden",
        HttpStatusCode.NotFound => "not_found",
        HttpStatusCode.MethodNotAllowed => "method_not_allowed",
        HttpStatusCode.NotAcceptable => "not_acceptable",
        HttpStatusCode.Conflict => "conflict",
        HttpStatusCode.RequestEntityTooLarge => "payload_too_large",
        HttpStatusCode.UnsupportedMediaType => "unsupported_media_type",
        HttpStatusCode.TooManyRequests => "rate_limited",
        HttpStatusCode.ServiceUnavailable => "unavailable",
        _ => "internal_error",
    };

    /// <summary>
    /// A body that is not the expected JSON: malformed or incomplete (<see cref="JsonException"/>), an unsupported
    /// type (<see cref="NotSupportedException"/>) or an invalid charset (<see cref="InvalidOperationException"/>).
    /// </summary>
    private static bool IsUnreadableBody(Exception ex) =>
        ex is JsonException or NotSupportedException or InvalidOperationException;

    private static BackendErrorException InvalidResponse(HttpResponseMessage response, Exception? inner) =>
        new((int)response.StatusCode, BackendErrorException.InvalidResponseCode, "The backend returned an unreadable response.", inner);
}
