namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>Outcome of one <see cref="IWebSocket.ReceiveAsync"/> call.</summary>
public readonly record struct WebSocketReceive(int Count, bool EndOfMessage, bool Closed);

/// <summary>
/// Minimal WebSocket seam over <see cref="System.Net.WebSockets.ClientWebSocket"/>, so the
/// stream source's reassembly/reconnect logic can be unit-tested with a scripted fake (no
/// network). A fresh instance is created per connection attempt (ClientWebSocket is single-use).
/// </summary>
public interface IWebSocket : IAsyncDisposable
{
    Task ConnectAsync(Uri uri, CancellationToken cancellationToken);

    Task<WebSocketReceive> ReceiveAsync(Memory<byte> buffer, CancellationToken cancellationToken);
}
