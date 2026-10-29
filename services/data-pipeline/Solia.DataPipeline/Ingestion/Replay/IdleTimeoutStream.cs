namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Read-only wrapper that fails a read making no progress for <see cref="IdleTimeout"/> with a
/// <see cref="TimeoutException"/>. <see cref="HttpClient.Timeout"/> only bounds the wait for response
/// headers; without this a body stalled mid-hour (no reset, no data) would block a replay forever
/// instead of failing the attempt so it is retried.
/// </summary>
public sealed class IdleTimeoutStream(Stream inner, TimeSpan idleTimeout) : Stream
{
    public TimeSpan IdleTimeout => idleTimeout;

    public override bool CanRead => true;

    public override bool CanSeek => false;

    public override bool CanWrite => false;

    public override long Length => throw new NotSupportedException();

    public override long Position
    {
        get => throw new NotSupportedException();
        set => throw new NotSupportedException();
    }

    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(idleTimeout);
        try
        {
            return await inner.ReadAsync(buffer, timeout.Token);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested && !cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"no data received for {idleTimeout}; the download stalled.");
        }
    }

    public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

    // Synchronous callers get the same bound (the archive readers are async, so this is only a fallback).
    public override int Read(byte[] buffer, int offset, int count) =>
        ReadAsync(buffer.AsMemory(offset, count)).AsTask().GetAwaiter().GetResult();

    public override void Flush()
    {
    }

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            inner.Dispose();
        }

        base.Dispose(disposing);
    }
}
