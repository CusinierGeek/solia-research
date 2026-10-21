namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>
/// Shared, thread-safe view of the live stream's connection status. Updated by
/// <see cref="PumpApiStreamSource"/> and read by the health check. Registered as a singleton.
/// </summary>
public sealed class StreamConnectionState
{
    private volatile bool _connected;

    public bool IsConnected => _connected;

    public DateTimeOffset? LastChangeUtc { get; private set; }

    public string? LastError { get; private set; }

    public void MarkConnected()
    {
        _connected = true;
        LastChangeUtc = DateTimeOffset.UtcNow;
        LastError = null;
    }

    public void MarkDisconnected(string? error = null)
    {
        _connected = false;
        LastChangeUtc = DateTimeOffset.UtcNow;
        LastError = error;
    }
}
