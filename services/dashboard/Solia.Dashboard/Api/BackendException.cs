namespace Solia.Dashboard.Api;

/// <summary>
/// Any failure of an <see cref="ISoliaApiClient"/> call. <see cref="Exception.Message"/> is always safe to show
/// in the UI: it is either the backend's own envelope message or a fixed sentence — never a transport
/// exception's text, which only travels as <see cref="Exception.InnerException"/> (for logs).
/// </summary>
public abstract class BackendException(string message, Exception? innerException)
    : Exception(message, innerException);

/// <summary>
/// The backend answered with an error: its envelope's <see cref="Code"/> (e.g. <c>invalid_request</c>,
/// <c>not_found</c>, <c>unavailable</c>) and message, or — for a body that is not the envelope — a code derived
/// from <see cref="StatusCode"/> with a generic message.
/// </summary>
public sealed class BackendErrorException(int statusCode, string code, string message, Exception? innerException = null)
    : BackendException(message, innerException)
{
    /// <summary>Code of a 2xx answer whose body is not the documented payload.</summary>
    public const string InvalidResponseCode = "invalid_response";

    /// <summary>The HTTP status; 0 when the response was dropped unread (an oversized <see cref="InvalidResponseCode"/> body).</summary>
    public int StatusCode { get; } = statusCode;

    public string Code { get; } = code;
}

/// <summary>
/// The backend could not be reached: connection failure, timeout, or a proxy answering 408/502/503/504 without the
/// backend's envelope (the container is down behind a proxy). Transient by nature — the UI offers a retry.
/// </summary>
public sealed class BackendUnavailableException(Exception? innerException = null)
    : BackendException(DefaultMessage, innerException)
{
    public const string DefaultMessage = "The backend is unavailable. Try again in a moment.";
}
