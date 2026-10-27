using System.Text.Json;
using Microsoft.AspNetCore.Http.Connections.Client;
using Microsoft.AspNetCore.SignalR.Client;
using Solia.Dashboard.Api;

namespace Solia.Dashboard.Live;

/// <summary>
/// The slice of a SignalR <see cref="HubConnection"/> the <see cref="LiveFeed"/> uses, so the feed's dispatch and
/// state logic is testable without a hub.
/// </summary>
internal interface ILiveConnection : IAsyncDisposable
{
    /// <summary>The connection dropped and automatic reconnect started.</summary>
    event Func<Exception?, Task>? Reconnecting;

    /// <summary>Automatic reconnect succeeded.</summary>
    event Func<string?, Task>? Reconnected;

    /// <summary>The connection stopped for good (stopped, closed by the server, or reconnect gave up).</summary>
    event Func<Exception?, Task>? Closed;

    IDisposable On<T>(string eventName, Func<T, Task> handler);

    Task StartAsync(CancellationToken cancellationToken);

    Task StopAsync(CancellationToken cancellationToken);
}

/// <summary>A real <see cref="HubConnection"/> to the backend hub: JSON payloads in snake_case like REST.</summary>
internal sealed class HubLiveConnection(HubConnection hub) : ILiveConnection
{
    private int _disposed;

    public static HubLiveConnection Create(
        Uri hubUrl,
        IRetryPolicy retryPolicy,
        ILoggerFactory loggerFactory,
        Action<HttpConnectionOptions>? configureHttp = null)
    {
        HubConnectionBuilder builder = new();
        builder.Services.AddSingleton(loggerFactory);
        builder.WithUrl(hubUrl, options => configureHttp?.Invoke(options))
            .AddJsonProtocol(o => o.PayloadSerializerOptions = new JsonSerializerOptions(SoliaApiJson.Options))
            .WithAutomaticReconnect(retryPolicy);
        return new HubLiveConnection(builder.Build());
    }

    public event Func<Exception?, Task>? Reconnecting
    {
        add => hub.Reconnecting += value;
        remove => hub.Reconnecting -= value;
    }

    public event Func<string?, Task>? Reconnected
    {
        add => hub.Reconnected += value;
        remove => hub.Reconnected -= value;
    }

    public event Func<Exception?, Task>? Closed
    {
        add => hub.Closed += value;
        remove => hub.Closed -= value;
    }

    public IDisposable On<T>(string eventName, Func<T, Task> handler) => hub.On(eventName, handler);

    public Task StartAsync(CancellationToken cancellationToken) => hub.StartAsync(cancellationToken);

    public Task StopAsync(CancellationToken cancellationToken) => hub.StopAsync(cancellationToken);

    // Disposed by both the feed and the container that registered it.
    public ValueTask DisposeAsync() =>
        Interlocked.Exchange(ref _disposed, 1) == 0 ? hub.DisposeAsync() : ValueTask.CompletedTask;
}
