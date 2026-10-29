using Solia.DataPipeline.Ingestion.Replay;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public class IdleTimeoutStreamTests
{
    private static readonly TimeSpan Short = TimeSpan.FromMilliseconds(100);

    [Fact]
    public async Task Passes_data_through()
    {
        await using var stream = new IdleTimeoutStream(new MemoryStream([1, 2, 3]), Short);
        using var copy = new MemoryStream();

        await stream.CopyToAsync(copy);

        Assert.Equal(new byte[] { 1, 2, 3 }, copy.ToArray());
    }

    [Fact]
    public async Task Stalled_read_throws_a_timeout()
    {
        await using var stream = new IdleTimeoutStream(new NeverStream(), Short);

        var ex = await Assert.ThrowsAsync<TimeoutException>(() => stream.ReadAsync(new byte[8]).AsTask());
        Assert.Contains("stalled", ex.Message);
    }

    [Fact]
    public void Synchronous_reads_are_bounded_too()
    {
        using var stream = new IdleTimeoutStream(new NeverStream(), Short);

        Assert.Throws<TimeoutException>(() => stream.Read(new byte[8], 0, 8));
    }

    [Fact]
    public async Task Caller_cancellation_stays_a_cancellation()
    {
        await using var stream = new IdleTimeoutStream(new NeverStream(), TimeSpan.FromMinutes(5));
        using var cts = new CancellationTokenSource(Short);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => stream.ReadAsync(new byte[8], cts.Token).AsTask());
    }

    [Fact]
    public void Disposing_releases_the_inner_stream()
    {
        var inner = new MemoryStream([1]);

        new IdleTimeoutStream(inner, Short).Dispose();

        Assert.False(inner.CanRead);
    }

    private sealed class NeverStream : MemoryStream
    {
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await Task.Delay(Timeout.Infinite, cancellationToken);
            return 0;
        }
    }
}
