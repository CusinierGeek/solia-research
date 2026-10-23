using Solia.Core.Http;

namespace Solia.Api.Middleware;

/// <summary>
/// Turns every failure into the one documented envelope (docs/API-CONTRACTS.md):
/// <c>{ "error": { "code": "...", "message": "..." } }</c>.
/// <list type="bullet">
///   <item>An <see cref="ApiException"/> (a well-known failure) maps to its own status + code + message.</item>
///   <item>A framework <see cref="BadHttpRequestException"/> (oversized or malformed request) keeps its 4xx
///   status and gets that status's code.</item>
///   <item>A request the client aborted is not an error: it is logged at Debug and nothing is written.</item>
///   <item>Any other unhandled exception becomes a 500 <see cref="ApiErrorCodes.InternalError"/> whose real
///   message is logged but never leaked to the caller.</item>
///   <item>A bare error status with no body — a routing 404/405, an auth 401 challenge, a rate-limit 429 —
///   is wrapped in the same envelope, keeping the headers it relies on (Allow, WWW-Authenticate,
///   Retry-After, …), so clients get one shape for every failure.</item>
/// </list>
/// Expected client failures (4xx) are logged at Information without a stack trace, so they don't flood
/// the Warning-level `logs` table mirror; server failures (5xx) are logged at Warning/Error.
/// Registered first so it wraps the whole pipeline. Once the response has started it can't rewrite it, so
/// an exception after the first bytes is rethrown.
/// </summary>
public sealed class ErrorShapeMiddleware(RequestDelegate next, ILogger<ErrorShapeMiddleware> logger)
{
    public async Task InvokeAsync(HttpContext context)
    {
        try
        {
            await next(context);
        }
        catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested)
        {
            // The client went away: there is nobody to answer and nothing went wrong server-side.
            logger.LogDebug("request aborted by the client");
            return;
        }
        catch (Exception ex) when (context.Response.HasStarted)
        {
            logger.LogError(ex, "failure after the response had started");
            throw;
        }
        catch (ApiException ex)
        {
            await HandleKnownFailureAsync(context, ex, ex.StatusCode, ex.Code, ex.Message);
            return;
        }
        catch (BadHttpRequestException ex)
        {
            (string code, string message) = Describe(ex.StatusCode);
            await HandleKnownFailureAsync(context, ex, ex.StatusCode, code, message);
            return;
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "unhandled exception");
            await WriteFailureEnvelopeAsync(context, StatusCodes.Status500InternalServerError, ApiErrorCodes.InternalError,
                "An unexpected error occurred.");
            return;
        }

        await WrapBareErrorStatusAsync(context);
    }

    private async Task HandleKnownFailureAsync(HttpContext context, Exception ex, int statusCode, string code, string message)
    {
        if (statusCode >= StatusCodes.Status500InternalServerError)
        {
            logger.LogWarning(ex, "handled API failure {Code} ({StatusCode})", code, statusCode);
        }
        else
        {
            // An expected client failure: no stack trace, and below the `logs` table's Warning threshold.
            logger.LogInformation("handled API failure {Code} ({StatusCode}): {Message}", code, statusCode, ex.Message);
        }

        await WriteFailureEnvelopeAsync(context, statusCode, code, message);
    }

    private static Task WrapBareErrorStatusAsync(HttpContext context)
    {
        HttpResponse response = context.Response;
        bool hasBody = response.ContentLength is > 0 || !string.IsNullOrEmpty(response.ContentType);
        if (response.HasStarted || response.StatusCode < StatusCodes.Status400BadRequest || hasBody)
        {
            return Task.CompletedTask;
        }

        // No Clear(): the headers already set (Allow, WWW-Authenticate, Retry-After, CORS…) belong to this
        // status and must survive; there is no body to discard.
        (string code, string message) = Describe(response.StatusCode);
        return WriteEnvelopeAsync(response, code, message);
    }

    /// <summary>
    /// Replaces whatever the failed request left in the response (status, headers, buffered body) with the
    /// envelope — the endpoint's headers describe a response that no longer exists.
    /// </summary>
    private static Task WriteFailureEnvelopeAsync(HttpContext context, int statusCode, string code, string message)
    {
        context.Response.Clear();
        context.Response.StatusCode = statusCode;
        return WriteEnvelopeAsync(context.Response, code, message);
    }

    private static Task WriteEnvelopeAsync(HttpResponse response, string code, string message)
    {
        response.ContentType = "application/json; charset=utf-8";
        return response.WriteAsJsonAsync(new ApiErrorResponse(new ApiError(code, message)));
    }

    private static (string Code, string Message) Describe(int statusCode) => statusCode switch
    {
        StatusCodes.Status400BadRequest => (ApiErrorCodes.InvalidRequest, "The request was invalid."),
        StatusCodes.Status401Unauthorized => (ApiErrorCodes.Unauthorized, "Authentication is required."),
        StatusCodes.Status403Forbidden => (ApiErrorCodes.Forbidden, "Access is forbidden."),
        StatusCodes.Status404NotFound => (ApiErrorCodes.NotFound, "The requested resource was not found."),
        StatusCodes.Status405MethodNotAllowed => (ApiErrorCodes.MethodNotAllowed, "The HTTP method is not allowed for this resource."),
        StatusCodes.Status406NotAcceptable => (ApiErrorCodes.NotAcceptable, "The requested response format is not available."),
        StatusCodes.Status409Conflict => (ApiErrorCodes.Conflict, "The request conflicts with the current state."),
        StatusCodes.Status413PayloadTooLarge => (ApiErrorCodes.PayloadTooLarge, "The request body is too large."),
        StatusCodes.Status415UnsupportedMediaType => (ApiErrorCodes.UnsupportedMediaType, "The request content type is not supported."),
        StatusCodes.Status429TooManyRequests => (ApiErrorCodes.RateLimited, "Too many requests; retry later."),
        StatusCodes.Status503ServiceUnavailable => (ApiErrorCodes.Unavailable, "The service is unavailable."),
        >= 500 => (ApiErrorCodes.InternalError, "An unexpected error occurred."),
        _ => (ApiErrorCodes.InvalidRequest, "The request could not be processed."),
    };
}
