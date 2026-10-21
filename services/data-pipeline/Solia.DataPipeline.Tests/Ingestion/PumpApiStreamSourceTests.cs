using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests.Ingestion;

public class PumpApiStreamSourceTests
{
    private const string BuyA = """{"action":"buy","mint":"A","pool":"pump","price":0.001,"timestamp":1700000000000}""";
    private const string BuyB = """{"action":"buy","mint":"B","pool":"pump","price":0.002,"timestamp":1700000000001}""";

    [Fact]
    public async Task Yields_parsed_events_and_skips_unparseable()
    {
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Build(state, Socket(Step.Frame(BuyA), Step.Frame("garbage"), Step.Frame(BuyB)));

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "A", "B" }, events.Select(e => e.Mint));
    }

    [Fact]
    public async Task Reassembles_a_fragmented_message()
    {
        int mid = BuyA.Length / 2;
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Build(state,
            Socket(Step.Frame(BuyA[..mid], endOfMessage: false), Step.Frame(BuyA[mid..], endOfMessage: true)));

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
    }

    [Fact]
    public async Task Reconnects_and_resumes_after_a_drop()
    {
        var state = new StreamConnectionState();
        // First connection yields A then throws; the source should reconnect and get B.
        PumpApiStreamSource source = Build(state,
            Socket(Step.Frame(BuyA), Step.Throw()),
            Socket(Step.Frame(BuyB)));

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "A", "B" }, events.Select(e => e.Mint));
    }

    [Fact]
    public async Task Marks_connected_while_streaming()
    {
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Build(state, Socket(Step.Frame(BuyA)));

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        await foreach (RawEvent _ in source.StreamAsync(cts.Token))
        {
            Assert.True(state.IsConnected);
            await cts.CancelAsync();
        }
    }

    [Fact]
    public async Task Stops_promptly_on_cancellation()
    {
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Build(state, Socket()); // connects, then idles

        using var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromMilliseconds(50));

        var events = new List<RawEvent>();
        await foreach (RawEvent e in source.StreamAsync(cts.Token))
        {
            events.Add(e);
        }

        Assert.Empty(events);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 4)]
    [InlineData(3, 8)]
    [InlineData(10, 30)] // capped at max
    public void Backoff_doubles_then_caps(int attempt, double expectedSeconds)
    {
        Assert.Equal(expectedSeconds, PumpApiStreamSource.ComputeBackoff(attempt, 1, 30, jitter01: 0).TotalSeconds, 3);
    }

    [Fact]
    public void Backoff_jitter_adds_at_most_20_percent_and_respects_cap()
    {
        Assert.Equal(1.2, PumpApiStreamSource.ComputeBackoff(0, 1, 30, jitter01: 1).TotalSeconds, 3);
        Assert.Equal(30, PumpApiStreamSource.ComputeBackoff(10, 1, 30, jitter01: 1).TotalSeconds, 3); // 36 -> capped
    }

    // --- helpers ---

    private static PumpApiStreamSource Build(StreamConnectionState state, params FakeWebSocket[] sockets)
    {
        var queue = new Queue<FakeWebSocket>(sockets);
        Func<IWebSocket> factory = () => queue.Count > 0 ? queue.Dequeue() : new FakeWebSocket();
        return new PumpApiStreamSource(
            factory,
            new PumpApiOptions(),
            state,
            NullLogger<PumpApiStreamSource>.Instance,
            delay: static (_, _) => Task.CompletedTask);
    }

    private static FakeWebSocket Socket(params Step[] steps) => new(steps);

    private static async Task<List<RawEvent>> Collect(PumpApiStreamSource source, int count)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5)); // fail fast instead of hanging
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

    private sealed class FakeWebSocket(params Step[] steps) : IWebSocket
    {
        private readonly Queue<Step> _steps = new(steps);

        public Task ConnectAsync(Uri uri, CancellationToken cancellationToken) => Task.CompletedTask;

        public async Task<WebSocketReceive> ReceiveAsync(Memory<byte> buffer, CancellationToken cancellationToken)
        {
            if (_steps.Count == 0)
            {
                await Task.Delay(Timeout.Infinite, cancellationToken); // idle connection until cancelled
            }

            return _steps.Dequeue().Apply(buffer);
        }

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    private sealed class Step
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

        public static Step Throw() => new(_ => throw new IOException("simulated drop"));
    }
}
