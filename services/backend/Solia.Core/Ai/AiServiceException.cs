using Solia.Core.Http;

namespace Solia.Core.Ai;

/// <summary>What went wrong calling the ai-service, independent of how the public API reports it.</summary>
public enum AiServiceErrorKind
{
    /// <summary>The ai-service rejected the request (400/422 <c>invalid_*</c>). Never retried.</summary>
    InvalidRequest,

    /// <summary>
    /// The token's feature window is shorter than the champion's training <c>min_seq_len</c>
    /// (422 <c>window_too_short</c>): too young to score yet. Never retried.
    /// </summary>
    InsufficientHistory,

    /// <summary>
    /// No usable champion: none registered yet (503 <c>no_champion</c>) or one trained on another feature
    /// spec (503 <c>champion_incompatible</c>) — both last until a retrain and promotion.
    /// </summary>
    NoChampion,

    /// <summary>
    /// The ai-service could not serve the call: unreachable, timed out, circuit open, champion failed to
    /// load (503 <c>champion_unavailable</c>) or another transient status that outlived the retries.
    /// </summary>
    Unavailable,

    /// <summary>
    /// A retrain is already running (409 <c>retrain_in_progress</c>): refused, not queued — retry once it finished.
    /// Never retried.
    /// </summary>
    Busy,

    /// <summary>The ai-service failed (500 <c>internal_error</c>, an unexpected status or an unreadable body).</summary>
    UpstreamError,
}

/// <summary>
/// A failed ai-service call, already mapped to the public error envelope: <see cref="AiServiceErrorKind.InvalidRequest"/>
/// → 400 <c>invalid_request</c>, <see cref="AiServiceErrorKind.InsufficientHistory"/> → 422 <c>insufficient_history</c>,
/// <see cref="AiServiceErrorKind.NoChampion"/> / <see cref="AiServiceErrorKind.Unavailable"/>
/// → 503 <c>unavailable</c>, <see cref="AiServiceErrorKind.Busy"/> → 409 <c>conflict</c>,
/// <see cref="AiServiceErrorKind.UpstreamError"/> → 502 <c>internal_error</c>. The public
/// message is fixed per kind: the upstream one can carry predictor internals (the ai-service echoes exception
/// text), so it stays in <see cref="UpstreamMessage"/> for logging only, with the upstream status/code.
/// </summary>
public sealed class AiServiceException : ApiException
{
    public AiServiceException(
        AiServiceErrorKind kind,
        string message,
        int? upstreamStatusCode = null,
        string? upstreamCode = null,
        Exception? innerException = null)
        : base(PublicStatus(kind), PublicCode(kind), PublicMessage(kind, message), innerException)
    {
        Kind = kind;
        UpstreamStatusCode = upstreamStatusCode;
        UpstreamCode = upstreamCode;
        UpstreamMessage = message;
    }

    public AiServiceErrorKind Kind { get; }

    /// <summary>The ai-service's HTTP status, or null when no response was received.</summary>
    public int? UpstreamStatusCode { get; }

    /// <summary>The ai-service's envelope code (e.g. <c>invalid_features</c>), or null when absent.</summary>
    public string? UpstreamCode { get; }

    /// <summary>The ai-service's envelope message, or a description of the transport failure.</summary>
    public string UpstreamMessage { get; }

    private static int PublicStatus(AiServiceErrorKind kind) => kind switch
    {
        AiServiceErrorKind.InvalidRequest => 400,
        AiServiceErrorKind.InsufficientHistory => 422,
        AiServiceErrorKind.NoChampion or AiServiceErrorKind.Unavailable => 503,
        AiServiceErrorKind.Busy => 409,
        _ => 502,
    };

    private static string PublicCode(AiServiceErrorKind kind) => kind switch
    {
        AiServiceErrorKind.InvalidRequest => ApiErrorCodes.InvalidRequest,
        AiServiceErrorKind.InsufficientHistory => ApiErrorCodes.InsufficientHistory,
        AiServiceErrorKind.NoChampion or AiServiceErrorKind.Unavailable => ApiErrorCodes.Unavailable,
        AiServiceErrorKind.Busy => ApiErrorCodes.Conflict,
        _ => ApiErrorCodes.InternalError,
    };

    private static string PublicMessage(AiServiceErrorKind kind, string message) => kind switch
    {
        AiServiceErrorKind.InvalidRequest => "the prediction service rejected the request",
        AiServiceErrorKind.InsufficientHistory => "not enough feature history yet to score this token",
        AiServiceErrorKind.NoChampion => "no champion model is available yet",
        AiServiceErrorKind.Unavailable => "the prediction service is unavailable",
        AiServiceErrorKind.Busy => "a retrain is already running",
        _ => "the prediction service failed",
    };
}
