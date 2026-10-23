namespace Solia.Core.Http;

/// <summary>
/// A well-known failure the API maps straight to the error envelope: carries the HTTP
/// <see cref="StatusCode"/> and the stable <see cref="Code"/> (an <see cref="ApiErrorCodes"/> value) the
/// response should report, with the exception message as the envelope message. The error-shape middleware
/// turns this into <c>{ "error": { "code", "message" } }</c>; any other exception becomes a generic
/// <see cref="ApiErrorCodes.InternalError"/> 500 whose message is not leaked. Endpoints (T4/T5) throw the
/// convenience subclasses below.
/// </summary>
public class ApiException : Exception
{
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="statusCode"/> is not a 4xx/5xx status.</exception>
    public ApiException(int statusCode, string code, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(code);
        // Only an error status makes sense in an error envelope (and an invalid one would throw mid-response).
        ArgumentOutOfRangeException.ThrowIfLessThan(statusCode, 400);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(statusCode, 599);
        StatusCode = statusCode;
        Code = code;
    }

    /// <summary>The HTTP status code the response carries.</summary>
    public int StatusCode { get; }

    /// <summary>The stable <see cref="ApiErrorCodes"/> value reported in the envelope.</summary>
    public string Code { get; }
}

/// <summary>404 — a requested resource does not exist (<see cref="ApiErrorCodes.NotFound"/>).</summary>
public sealed class NotFoundException(string message) : ApiException(404, ApiErrorCodes.NotFound, message);

/// <summary>400 — the request was malformed or failed validation (<see cref="ApiErrorCodes.InvalidRequest"/>).</summary>
public sealed class InvalidRequestException(string message) : ApiException(400, ApiErrorCodes.InvalidRequest, message);
