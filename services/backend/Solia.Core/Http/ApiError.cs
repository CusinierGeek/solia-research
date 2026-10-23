using System.Text.Json.Serialization;

namespace Solia.Core.Http;

/// <summary>
/// The documented error envelope (docs/API-CONTRACTS.md): <c>{ "error": { "code": "...", "message": "..." } }</c>.
/// Every failure the API returns — unhandled exceptions and well-known failures alike — is serialized as
/// this shape, so callers parse one contract.
/// </summary>
public sealed record ApiErrorResponse([property: JsonPropertyName("error")] ApiError Error);

/// <summary>The body of an <see cref="ApiErrorResponse"/>: a machine-readable <paramref name="Code"/> and a human-readable <paramref name="Message"/>.</summary>
public sealed record ApiError(
    [property: JsonPropertyName("code")] string Code,
    [property: JsonPropertyName("message")] string Message);

/// <summary>
/// The well-known <see cref="ApiError.Code"/> values. Stable, snake_case strings callers can switch on;
/// kept together so the API and its clients share one spelling.
/// </summary>
public static class ApiErrorCodes
{
    public const string InternalError = "internal_error";
    public const string NotFound = "not_found";
    public const string InvalidRequest = "invalid_request";
    public const string Unauthorized = "unauthorized";
    public const string Forbidden = "forbidden";
    public const string Conflict = "conflict";
    public const string Unavailable = "unavailable";
    public const string MethodNotAllowed = "method_not_allowed";
    public const string NotAcceptable = "not_acceptable";
    public const string PayloadTooLarge = "payload_too_large";
    public const string UnsupportedMediaType = "unsupported_media_type";
    public const string RateLimited = "rate_limited";
    public const string InsufficientHistory = "insufficient_history";
}
