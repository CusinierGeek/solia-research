using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.SignalR;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Solia.Dashboard.Tests.EndToEnd;

/// <summary>
/// A contract-faithful stand-in for the backend (docs/API-CONTRACTS.md): the read endpoints under <c>/api/v1</c>
/// answer the raw snake_case JSON the test seeds (by path; the query string is recorded, not interpreted), unknown
/// paths answer the documented error envelope, and a SignalR hub at <c>/hubs/live</c> pushes raw JSON events. In
/// memory (TestServer): no socket, no Docker, no database. Can play an outage: <see cref="RestDown"/> answers 503
/// with no envelope (a stopped container behind a gateway), <see cref="TakeHubDown"/> answers 503 to every hub
/// request until <see cref="BringHubUp"/>.
/// </summary>
internal sealed class StubBackend : IAsyncDisposable
{
    /// <summary>The backend's address as the dashboard is configured with it (never resolved: TestServer handles it).</summary>
    public const string BaseUrl = "http://backend";

    private static readonly TimeSpan PollTimeout = TimeSpan.FromMilliseconds(200);

    private readonly ConcurrentDictionary<string, string> _responses = new();
    private readonly WebApplication _app;
    private volatile bool _restDown;
    private volatile bool _hubDown;

    private StubBackend(WebApplication app) => _app = app;

    /// <summary>Paths and queries requested under <c>/api/v1/</c>, in order.</summary>
    public ConcurrentQueue<string> Requests { get; } = new();

    /// <summary>Whether every REST call answers 503 without the error envelope.</summary>
    public bool RestDown
    {
        get => _restDown;
        set => _restDown = value;
    }

    public static async Task<StubBackend> StartAsync()
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSignalR();
        WebApplication app = builder.Build();
        StubBackend backend = new(app);

        app.Use((context, next) =>
            backend._hubDown && context.Request.Path.StartsWithSegments("/hubs/live")
                ? Results.StatusCode(StatusCodes.Status503ServiceUnavailable).ExecuteAsync(context)
                : next(context));
        app.MapGet("/api/v1/{**path}", backend.Answer);
        // Short polls, so an outage is noticed by the client's next poll within a fraction of a second.
        app.MapHub<LiveHub>("/hubs/live", o => o.LongPolling.PollTimeout = PollTimeout);

        await app.StartAsync();
        return backend;
    }

    /// <summary>A handler into the in-memory server, for the dashboard's HTTP client and hub connection.</summary>
    public HttpMessageHandler CreateHandler() => _app.GetTestServer().CreateHandler();

    /// <summary>Seeds the body <c>GET /api/v1/{path}</c> answers (200), whatever its query.</summary>
    public void Respond(string path, string json)
    {
        using JsonDocument _ = JsonDocument.Parse(json); // a typo in a fixture fails here, not as a confusing UI state
        _responses[path] = json;
    }

    /// <summary>The requests made to <c>/api/v1/{path}</c> so far.</summary>
    public int RequestCount(string path) =>
        Requests.Count(r => r == path || r.StartsWith(path + "?", StringComparison.Ordinal));

    /// <summary>Broadcasts a hub event with a raw JSON payload, as the backend does.</summary>
    public Task PushAsync(string eventName, string json) =>
        _app.Services.GetRequiredService<IHubContext<LiveHub>>().Clients.All
            .SendAsync(eventName, JsonDocument.Parse(json).RootElement);

    /// <summary>
    /// Makes the hub unreachable: every request to it (the next poll, negotiate on reconnect) answers 503, so the
    /// client sees a transport failure — a backend gone away, not one closing the connection cleanly.
    /// </summary>
    public void TakeHubDown() => _hubDown = true;

    public void BringHubUp() => _hubDown = false;

    private IResult Answer(string path, HttpRequest request)
    {
        Requests.Enqueue(path + request.QueryString);
        if (_restDown)
        {
            return Results.StatusCode(StatusCodes.Status503ServiceUnavailable);
        }
        return _responses.TryGetValue(path, out string? json)
            ? Results.Content(json, "application/json")
            : Results.Json(new { error = new { code = "not_found", message = $"No stub for /{path}." } }, statusCode: StatusCodes.Status404NotFound);
    }

    public ValueTask DisposeAsync() => _app.DisposeAsync();

    /// <summary>The live hub: server → client only, like the backend's.</summary>
    public sealed class LiveHub : Hub;
}
