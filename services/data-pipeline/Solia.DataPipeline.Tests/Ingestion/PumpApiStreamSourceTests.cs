using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;
using static Solia.DataPipeline.Tests.Ingestion.StreamFakes;

namespace Solia.DataPipeline.Tests.Ingestion;

public class PumpApiStreamSourceTests
{
    [Fact]
    public async Task Yields_parsed_events_and_skips_unparseable()
    {
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Source(state, [Socket(Step.Frame(BuyA), Step.Frame("garbage"), Step.Frame(BuyB))]);

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "A", "B" }, events.Select(e => e.Mint));
    }

    [Fact]
    public async Task Reassembles_a_fragmented_message()
    {
        int mid = BuyA.Length / 2;
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Frame(BuyA[..mid], endOfMessage: false), Step.Frame(BuyA[mid..], endOfMessage: true))]);

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
    }

    [Fact]
    public async Task Reused_buffer_does_not_bleed_a_long_message_into_a_shorter_one()
    {
        string longA = BuyA + new string(' ', 500);
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Source(state, [Socket(Step.Frame(longA), Step.Frame(BuyB))]);

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "A", "B" }, events.Select(e => e.Mint));
    }

    [Fact]
    public async Task A_frame_that_would_overflow_the_price_does_not_kill_the_stream()
    {
        const string overflow = """
        {"action":"buy","mint":"X","pool":"pump","vQuoteInBondingCurve":79000000000000000000000000000,
         "vTokensInBondingCurve":0.001,"timestamp":1700000000000}
        """;
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Source(state, [Socket(Step.Frame(overflow), Step.Frame(BuyB))]);

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "X", "B" }, events.Select(e => e.Mint));
        Assert.Null(events[0].Price);
    }

    [Fact]
    public async Task Reconnects_and_resumes_after_a_drop()
    {
        var state = new StreamConnectionState();
        // First connection yields A then throws; the source should reconnect and get B.
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Frame(BuyA), Step.Throw()), Socket(Step.Frame(BuyB))]);

        List<RawEvent> events = await Collect(source, 2);

        Assert.Equal(new[] { "A", "B" }, events.Select(e => e.Mint));
    }

    [Fact]
    public async Task Connect_failure_marks_disconnected_and_retries()
    {
        var state = new StreamConnectionState();
        var seen = new List<StreamStatus>();
        PumpApiStreamSource source = Source(state,
            [new FakeWebSocket { ConnectError = new IOException("host unreachable") }, Socket(Step.Frame(BuyA))],
            delay: (_, _) => { seen.Add(state.Current); return Task.CompletedTask; });

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
        StreamStatus whileWaiting = Assert.Single(seen);
        Assert.False(whileWaiting.IsConnected);
        Assert.Equal("host unreachable", whileWaiting.LastError);
    }

    [Fact]
    public async Task Server_close_frame_marks_disconnected_and_reconnects()
    {
        var state = new StreamConnectionState();
        var seen = new List<StreamStatus>();
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Close()), Socket(Step.Frame(BuyA))],
            delay: (_, _) => { seen.Add(state.Current); return Task.CompletedTask; });

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
        StreamStatus whileWaiting = Assert.Single(seen);
        Assert.False(whileWaiting.IsConnected);
        Assert.Equal("closed by server", whileWaiting.LastError);
    }

    [Fact]
    public async Task Idle_connection_is_treated_as_dead_and_replaced()
    {
        var state = new StreamConnectionState();
        var seen = new List<StreamStatus>();
        FakeWebSocket silent = Socket(); // connects, then never sends anything
        PumpApiStreamSource source = Source(state,
            [silent, Socket(Step.Frame(BuyA))],
            delay: (_, _) => { seen.Add(state.Current); return Task.CompletedTask; },
            idleTimeout: TimeSpan.FromMilliseconds(100));

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
        Assert.True(silent.Disposed);
        Assert.StartsWith("no data for", Assert.Single(seen).LastError);
    }

    [Fact]
    public async Task Oversized_message_drops_the_connection()
    {
        var state = new StreamConnectionState();
        var seen = new List<StreamStatus>();
        string chunk = new('x', 800);
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Frame(chunk, endOfMessage: false), Step.Frame(chunk, endOfMessage: false)), Socket(Step.Frame(BuyA))],
            options: new PumpApiOptions { MaxMessageBytes = 1024 },
            delay: (_, _) => { seen.Add(state.Current); return Task.CompletedTask; });

        List<RawEvent> events = await Collect(source, 1);

        Assert.Equal("A", Assert.Single(events).Mint);
        Assert.Equal("message exceeded 1024 bytes", Assert.Single(seen).LastError);
    }

    [Fact]
    public async Task Backoff_keeps_growing_when_the_server_closes_right_after_connecting()
    {
        var state = new StreamConnectionState();
        var delays = new List<TimeSpan>();
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Close()), Socket(Step.Close()), Socket(Step.Close()), Socket(Step.Frame(BuyA))],
            delay: (d, _) => { delays.Add(d); return Task.CompletedTask; });

        await Collect(source, 1);

        // base 1s doubling (each with up to 20% jitter): ~1, ~2, ~4 — never reset by the bare connect.
        Assert.Equal(3, delays.Count);
        Assert.InRange(delays[0].TotalSeconds, 1, 1.2);
        Assert.InRange(delays[1].TotalSeconds, 2, 2.4);
        Assert.InRange(delays[2].TotalSeconds, 4, 4.8);
    }

    [Fact]
    public async Task Backoff_resets_after_a_connection_that_delivered_data()
    {
        var state = new StreamConnectionState();
        var delays = new List<TimeSpan>();
        PumpApiStreamSource source = Source(state,
            [Socket(Step.Close()), Socket(Step.Close()), Socket(Step.Frame(BuyA), Step.Close()), Socket(Step.Frame(BuyB))],
            delay: (d, _) => { delays.Add(d); return Task.CompletedTask; });

        await Collect(source, 2);

        Assert.Equal(3, delays.Count);
        Assert.InRange(delays[1].TotalSeconds, 2, 2.4);
        Assert.InRange(delays[2].TotalSeconds, 1, 1.2); // back to base after a healthy session
    }

    [Fact]
    public async Task Consumer_stopping_early_disposes_the_socket_and_marks_disconnected()
    {
        var state = new StreamConnectionState();
        FakeWebSocket socket = Socket(Step.Frame(BuyA), Step.Frame(BuyB));
        PumpApiStreamSource source = Source(state, [socket]);

        await foreach (RawEvent _ in source.StreamAsync(CancellationToken.None))
        {
            Assert.True(state.IsConnected);
            break;
        }

        Assert.True(socket.Disposed);
        Assert.False(state.IsConnected);
        Assert.Equal("stopped", state.LastError);
    }

    [Fact]
    public async Task Marks_connected_while_streaming()
    {
        var state = new StreamConnectionState();
        PumpApiStreamSource source = Source(state, [Socket(Step.Frame(BuyA))]);

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
        PumpApiStreamSource source = Source(state, [Socket()]); // connects, then idles

        using var cts = new CancellationTokenSource();
        cts.CancelAfter(TimeSpan.FromMilliseconds(50));

        var events = new List<RawEvent>();
        await foreach (RawEvent e in source.StreamAsync(cts.Token))
        {
            events.Add(e);
        }

        Assert.Empty(events);
        Assert.False(state.IsConnected);
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

    private static FakeWebSocket Socket(params Step[] steps) => new(steps);
}
