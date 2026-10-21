using System.Net.WebSockets;

namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>Real <see cref="IWebSocket"/> backed by <see cref="ClientWebSocket"/>.</summary>
public sealed class SystemWebSocket : IWebSocket
{
    private static readonly TimeSpan DefaultCloseTimeout = TimeSpan.FromSeconds(5);
    private static readonly TimeSpan KeepAlive = TimeSpan.FromSeconds(15);

    private readonly ClientWebSocket _socket = new();
    private readonly TimeSpan _closeTimeout;

    public SystemWebSocket()
        : this(DefaultCloseTimeout)
    {
    }

    internal SystemWebSocket(TimeSpan closeTimeout)
    {
        _closeTimeout = closeTimeout;

        // Ping/pong keep-alive: a peer that stops answering faults the socket instead of
        // leaving a receive hanging on a half-open connection.
        _socket.Options.KeepAliveInterval = KeepAlive;
        _socket.Options.KeepAliveTimeout = KeepAlive;
    }

    public Task ConnectAsync(Uri uri, CancellationToken cancellationToken) =>
        _socket.ConnectAsync(uri, cancellationToken);

    public async Task<WebSocketReceive> ReceiveAsync(Memory<byte> buffer, CancellationToken cancellationToken)
    {
        ValueWebSocketReceiveResult result = await _socket.ReceiveAsync(buffer, cancellationToken);
        return new WebSocketReceive(result.Count, result.EndOfMessage, result.MessageType == WebSocketMessageType.Close);
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_socket.State == WebSocketState.Open)
            {
                // Bounded: a peer that never answers the close handshake must not block
                // reconnects or shutdown. Cancelling aborts the socket.
                using var timeout = new CancellationTokenSource(_closeTimeout);
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, timeout.Token);
            }
        }
        catch
        {
            // Closing a broken socket is best-effort; never throw from dispose.
        }

        _socket.Dispose();
    }
}
