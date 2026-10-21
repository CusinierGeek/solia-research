using System.Net.WebSockets;

namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>Real <see cref="IWebSocket"/> backed by <see cref="ClientWebSocket"/>.</summary>
public sealed class SystemWebSocket : IWebSocket
{
    private readonly ClientWebSocket _socket = new();

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
                await _socket.CloseAsync(WebSocketCloseStatus.NormalClosure, null, CancellationToken.None);
            }
        }
        catch
        {
            // Closing a broken socket is best-effort; never throw from dispose.
        }

        _socket.Dispose();
    }
}
