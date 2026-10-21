using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests.Ingestion;

/// <summary>Scripted WebSocket: replays its steps, then idles until cancelled.</summary>
internal sealed class FakeWebSocket(params Step[] steps) : IWebSocket
{
    private readonly Queue<Step> _steps = new(steps);

    public Exception? ConnectError { get; init; }

    public bool Disposed { get; private set; }

    public Task ConnectAsync(Uri uri, CancellationToken cancellationToken) =>
        ConnectError is null ? Task.CompletedTask : Task.FromException(ConnectError);

    public async Task<WebSocketReceive> ReceiveAsync(Memory<byte> buffer, CancellationToken cancellationToken)
    {
        if (_steps.Count == 0)
        {
            await Task.Delay(Timeout.Infinite, cancellationToken); // idle connection until cancelled
        }

        return _steps.Dequeue().Apply(buffer);
    }

    public ValueTask DisposeAsync()
    {
        Disposed = true;
        return ValueTask.CompletedTask;
    }
}

internal sealed class Step
{
    private readonly Func<Memory<byte>, WebSocketReceive> _apply;

    private Step(Func<Memory<byte>, WebSocketReceive> apply) => _apply = apply;

    public WebSocketReceive Apply(Memory<byte> buffer) => _apply(buffer);

    public static Step Frame(string text, bool endOfMessage = true) => new(buffer =>
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        bytes.CopyTo(buffer.Span);
        return new WebSocketReceive(bytes.Length, endOfMessage, Closed: false);
    });

    public static Step Close() => new(_ => new WebSocketReceive(0, EndOfMessage: true, Closed: true));

    public static Step Throw() => new(_ => throw new IOException("simulated drop"));
}

internal static class StreamFakes
{
    public const string BuyA = """{"action":"buy","mint":"A","pool":"pump","price":0.001,"timestamp":1700000000000}""";
    public const string BuyB = """{"action":"buy","mint":"B","pool":"pump","price":0.002,"timestamp":1700000000001}""";

    /// <summary>Source over the given sockets, in order; once they run out, new sockets just idle.</summary>
    public static PumpApiStreamSource Source(
        StreamConnectionState state,
        FakeWebSocket[] sockets,
        PumpApiOptions? options = null,
        Func<TimeSpan, CancellationToken, Task>? delay = null,
        TimeSpan? idleTimeout = null)
    {
        var queue = new Queue<FakeWebSocket>(sockets);
        Func<IWebSocket> factory = () => queue.Count > 0 ? queue.Dequeue() : new FakeWebSocket();
        return new PumpApiStreamSource(
            factory,
            options ?? new PumpApiOptions(),
            state,
            NullLogger<PumpApiStreamSource>.Instance,
            delay ?? (static (_, _) => Task.CompletedTask),
            idleTimeout);
    }

    /// <summary>Takes <paramref name="count"/> events then cancels; fails fast instead of hanging.</summary>
    public static async Task<List<RawEvent>> Collect(PumpApiStreamSource source, int count)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        var events = new List<RawEvent>();
        await foreach (RawEvent e in source.StreamAsync(cts.Token))
        {
            events.Add(e);
            if (events.Count >= count)
            {
                await cts.CancelAsync();
            }
        }
        return events;
    }
}
