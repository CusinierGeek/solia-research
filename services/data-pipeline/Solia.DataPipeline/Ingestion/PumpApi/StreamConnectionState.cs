namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>One consistent reading of the stream's connection status.</summary>
public sealed record StreamStatus(bool IsConnected, DateTimeOffset? LastChangeUtc, string? LastError);

/// <summary>
/// Shared, thread-safe view of the live stream's connection status. Updated by
/// <see cref="PumpApiStreamSource"/> and read by the health check. Registered as a singleton.
/// Each change swaps in a whole immutable <see cref="StreamStatus"/>, so a reader never sees
/// fields from two different updates.
/// </summary>
public sealed class StreamConnectionState
{
    private StreamStatus _status = new(false, null, null);

    /// <summary>Read this once when you need several fields together.</summary>
    public StreamStatus Current => Volatile.Read(ref _status);

    public bool IsConnected => Current.IsConnected;

    public DateTimeOffset? LastChangeUtc => Current.LastChangeUtc;

    public string? LastError => Current.LastError;

    public void MarkConnected() =>
        Volatile.Write(ref _status, new StreamStatus(true, DateTimeOffset.UtcNow, null));

    public void MarkDisconnected(string? error = null) =>
        Volatile.Write(ref _status, new StreamStatus(false, DateTimeOffset.UtcNow, error));
}
