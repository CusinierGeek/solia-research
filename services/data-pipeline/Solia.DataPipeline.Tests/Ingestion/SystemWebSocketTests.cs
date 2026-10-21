using System.Diagnostics;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Tests.Ingestion;

/// <summary>Runs the real ClientWebSocket wrapper against an in-process Kestrel WebSocket server.</summary>
public class SystemWebSocketTests
{
    [Fact]
    public async Task Receives_a_text_message()
    {
        await using WebApplication server = await StartServer(async ws =>
        {
            await ws.SendAsync(Encoding.UTF8.GetBytes("hello"), System.Net.WebSockets.WebSocketMessageType.Text, true, CancellationToken.None);
            await Task.Delay(Timeout.Infinite);
        });
        await using var socket = new SystemWebSocket();
        await socket.ConnectAsync(WsUri(server), CancellationToken.None);

        byte[] buffer = new byte[64];
        WebSocketReceive result = await socket.ReceiveAsync(buffer, CancellationToken.None);

        Assert.Equal("hello", Encoding.UTF8.GetString(buffer, 0, result.Count));
        Assert.True(result.EndOfMessage);
        Assert.False(result.Closed);
    }

    [Fact]
    public async Task Dispose_is_bounded_when_the_server_never_answers_the_close()
    {
        // The server accepts and then never reads, so it never completes the close handshake.
        await using WebApplication server = await StartServer(_ => Task.Delay(Timeout.Infinite));
        var socket = new SystemWebSocket(closeTimeout: TimeSpan.FromMilliseconds(200));
        await socket.ConnectAsync(WsUri(server), CancellationToken.None);

        var sw = Stopwatch.StartNew();
        await socket.DisposeAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10)); // throws instead of hanging
        sw.Stop();

        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(3), $"dispose took {sw.Elapsed}");
    }

    [Fact]
    public async Task Connect_to_an_unreachable_endpoint_throws()
    {
        await using var socket = new SystemWebSocket();

        await Assert.ThrowsAnyAsync<Exception>(() => socket.ConnectAsync(new Uri("ws://127.0.0.1:1/"), CancellationToken.None));
    }

    private static async Task<WebApplication> StartServer(Func<System.Net.WebSockets.WebSocket, Task> handler)
    {
        WebApplicationBuilder builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Logging.ClearProviders();
        WebApplication app = builder.Build();
        app.UseWebSockets();
        app.Run(async ctx =>
        {
            if (!ctx.WebSockets.IsWebSocketRequest)
            {
                ctx.Response.StatusCode = StatusCodes.Status400BadRequest;
                return;
            }

            using System.Net.WebSockets.WebSocket ws = await ctx.WebSockets.AcceptWebSocketAsync();
            await handler(ws).WaitAsync(ctx.RequestAborted).ContinueWith(_ => { }, TaskScheduler.Default);
        });
        await app.StartAsync();
        return app;
    }

    private static Uri WsUri(WebApplication app) => new(app.Urls.First().Replace("http://", "ws://"));
}
