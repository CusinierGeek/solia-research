using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.Api.Middleware;
using Solia.Core.Http;

namespace Solia.Api.Tests;

public class ErrorShapeMiddlewareTests
{
    private static async Task<(int Status, string Body)> RunAsync(RequestDelegate next, int initialStatus = StatusCodes.Status200OK)
    {
        (int status, string body, _) = await RunWithContextAsync(next, initialStatus);
        return (status, body);
    }

    private static async Task<(int Status, string Body, HttpContext Context)> RunWithContextAsync(
        RequestDelegate next, int initialStatus = StatusCodes.Status200OK, ILogger<ErrorShapeMiddleware>? logger = null,
        CancellationToken requestAborted = default)
    {
        var context = new DefaultHttpContext { RequestAborted = requestAborted };
        context.Response.Body = new MemoryStream();
        context.Response.StatusCode = initialStatus;

        var middleware = new ErrorShapeMiddleware(next, logger ?? NullLogger<ErrorShapeMiddleware>.Instance);
        await middleware.InvokeAsync(context);

        context.Response.Body.Position = 0;
        string body = await new StreamReader(context.Response.Body).ReadToEndAsync();
        return (context.Response.StatusCode, body, context);
    }

    private static (string Code, string Message) ReadEnvelope(string body)
    {
        using JsonDocument doc = JsonDocument.Parse(body);
        JsonElement error = doc.RootElement.GetProperty("error");
        return (error.GetProperty("code").GetString()!, error.GetProperty("message").GetString()!);
    }

    [Fact]
    public async Task Api_exception_maps_to_its_status_code_and_message()
    {
        (int status, string body) = await RunAsync(_ => throw new NotFoundException("unknown mint"));

        Assert.Equal(StatusCodes.Status404NotFound, status);
        (string code, string message) = ReadEnvelope(body);
        Assert.Equal(ApiErrorCodes.NotFound, code);
        Assert.Equal("unknown mint", message);
    }

    [Fact]
    public async Task Unhandled_exception_becomes_500_without_leaking_the_message()
    {
        (int status, string body) = await RunAsync(_ => throw new InvalidOperationException("secret connection string"));

        Assert.Equal(StatusCodes.Status500InternalServerError, status);
        (string code, string message) = ReadEnvelope(body);
        Assert.Equal(ApiErrorCodes.InternalError, code);
        Assert.DoesNotContain("secret", message);
    }

    [Fact]
    public async Task Bare_error_status_with_no_body_is_wrapped_in_the_envelope()
    {
        (int status, string body) = await RunAsync(
            ctx => { ctx.Response.StatusCode = StatusCodes.Status404NotFound; return Task.CompletedTask; });

        Assert.Equal(StatusCodes.Status404NotFound, status);
        Assert.Equal(ApiErrorCodes.NotFound, ReadEnvelope(body).Code);
    }

    [Fact]
    public async Task Successful_response_is_left_untouched()
    {
        (int status, string body) = await RunAsync(
            async ctx => { ctx.Response.ContentType = "text/plain"; await ctx.Response.WriteAsync("ok"); });

        Assert.Equal(StatusCodes.Status200OK, status);
        Assert.Equal("ok", body);
    }

    [Fact]
    public async Task Exception_after_the_response_started_is_rethrown()
    {
        var context = new DefaultHttpContext();
        context.Features.Set<IHttpResponseFeature>(new StartedResponseFeature());
        var middleware = new ErrorShapeMiddleware(
            _ => throw new InvalidOperationException("too late"),
            NullLogger<ErrorShapeMiddleware>.Instance);

        await Assert.ThrowsAsync<InvalidOperationException>(() => middleware.InvokeAsync(context));
    }

    [Theory]
    [InlineData(StatusCodes.Status405MethodNotAllowed, ApiErrorCodes.MethodNotAllowed)]
    [InlineData(StatusCodes.Status406NotAcceptable, ApiErrorCodes.NotAcceptable)]
    [InlineData(StatusCodes.Status413PayloadTooLarge, ApiErrorCodes.PayloadTooLarge)]
    [InlineData(StatusCodes.Status415UnsupportedMediaType, ApiErrorCodes.UnsupportedMediaType)]
    [InlineData(StatusCodes.Status429TooManyRequests, ApiErrorCodes.RateLimited)]
    [InlineData(StatusCodes.Status422UnprocessableEntity, ApiErrorCodes.InvalidRequest)]
    [InlineData(StatusCodes.Status502BadGateway, ApiErrorCodes.InternalError)]
    public async Task Bare_status_maps_to_its_own_stable_code(int statusCode, string expectedCode)
    {
        (int status, string body) = await RunAsync(ctx => { ctx.Response.StatusCode = statusCode; return Task.CompletedTask; });

        Assert.Equal(statusCode, status);
        Assert.Equal(expectedCode, ReadEnvelope(body).Code);
    }

    [Fact]
    public async Task Wrapping_a_bare_status_keeps_the_headers_it_relies_on()
    {
        (_, string body, HttpContext context) = await RunWithContextAsync(ctx =>
        {
            ctx.Response.StatusCode = StatusCodes.Status401Unauthorized;
            ctx.Response.Headers.WWWAuthenticate = "Bearer";
            ctx.Response.Headers.RetryAfter = "30";
            ctx.Response.Headers.AccessControlAllowOrigin = "https://dashboard.example";
            return Task.CompletedTask;
        });

        Assert.Equal(ApiErrorCodes.Unauthorized, ReadEnvelope(body).Code);
        Assert.Equal("Bearer", context.Response.Headers.WWWAuthenticate.ToString());
        Assert.Equal("30", context.Response.Headers.RetryAfter.ToString());
        Assert.Equal("https://dashboard.example", context.Response.Headers.AccessControlAllowOrigin.ToString());
        Assert.StartsWith("application/json", context.Response.ContentType);
    }

    [Fact]
    public async Task Exception_discards_the_failed_endpoints_headers()
    {
        (_, _, HttpContext context) = await RunWithContextAsync(ctx =>
        {
            ctx.Response.Headers.ETag = "\"stale\"";
            throw new InvalidOperationException("boom");
        });

        Assert.False(context.Response.Headers.ContainsKey("ETag"));
    }

    [Theory]
    [InlineData(StatusCodes.Status413PayloadTooLarge, ApiErrorCodes.PayloadTooLarge)]
    [InlineData(StatusCodes.Status400BadRequest, ApiErrorCodes.InvalidRequest)]
    public async Task Bad_http_request_keeps_its_4xx_status(int statusCode, string expectedCode)
    {
        (int status, string body) = await RunAsync(_ => throw new BadHttpRequestException("framework detail", statusCode));

        Assert.Equal(statusCode, status);
        (string code, string message) = ReadEnvelope(body);
        Assert.Equal(expectedCode, code);
        Assert.DoesNotContain("framework detail", message);
    }

    [Fact]
    public async Task Client_abort_writes_nothing_and_is_not_logged_as_an_error()
    {
        using var aborted = new CancellationTokenSource();
        await aborted.CancelAsync();
        var logger = new CapturingLogger();

        (int status, string body, _) = await RunWithContextAsync(
            ctx => throw new TaskCanceledException("client gone", null, ctx.RequestAborted),
            logger: logger, requestAborted: aborted.Token);

        Assert.Equal(StatusCodes.Status200OK, status);
        Assert.Empty(body);
        Assert.DoesNotContain(logger.Entries, e => e.Level >= LogLevel.Warning);
    }

    [Fact]
    public async Task Cancellation_not_caused_by_the_client_is_still_a_500()
    {
        (int status, string body) = await RunAsync(_ => throw new OperationCanceledException("internal timeout"));

        Assert.Equal(StatusCodes.Status500InternalServerError, status);
        Assert.Equal(ApiErrorCodes.InternalError, ReadEnvelope(body).Code);
    }

    [Fact]
    public async Task Client_failures_are_logged_below_warning_without_a_stack_trace()
    {
        var logger = new CapturingLogger();

        await RunWithContextAsync(_ => throw new NotFoundException("unknown mint"), logger: logger);

        (LogLevel level, Exception? exception) = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Information, level);
        Assert.Null(exception);
    }

    [Fact]
    public async Task Server_api_failures_are_logged_as_warnings_with_the_exception()
    {
        var logger = new CapturingLogger();

        await RunWithContextAsync(_ => throw new ApiException(503, ApiErrorCodes.Unavailable, "ai-service down"), logger: logger);

        (LogLevel level, Exception? exception) = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Warning, level);
        Assert.NotNull(exception);
    }

    /// <summary>Records the level and exception of every entry.</summary>
    private sealed class CapturingLogger : ILogger<ErrorShapeMiddleware>
    {
        public List<(LogLevel Level, Exception? Exception)> Entries { get; } = [];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(
            LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            Entries.Add((logLevel, exception));
    }

    /// <summary>A response feature that reports the response as already committed, so the body can't be rewritten.</summary>
    private sealed class StartedResponseFeature : IHttpResponseFeature
    {
        public int StatusCode { get; set; } = StatusCodes.Status200OK;
        public string? ReasonPhrase { get; set; }
        public IHeaderDictionary Headers { get; set; } = new HeaderDictionary();
        public Stream Body { get; set; } = new MemoryStream();
        public bool HasStarted => true;

        public void OnStarting(Func<object, Task> callback, object state) { }

        public void OnCompleted(Func<object, Task> callback, object state) { }
    }
}
