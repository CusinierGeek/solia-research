using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.Replay;
using ZstdSharp;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public class PumpApiReplaySourceTests
{
    private static readonly DateTimeOffset H0 = new(2026, 4, 18, 0, 0, 0, TimeSpan.Zero);

    [Fact]
    public async Task Streams_across_hours_and_skips_missing()
    {
        var window = new ReplayWindow(H0, H0.AddHours(3)); // hours 00, 01, 02
        var archives = new Dictionary<DateTimeOffset, byte[]?>
        {
            [H0] = Zstd(BuyLine("A", H0.AddMinutes(1)) + "\n" + BuyLine("B", H0.AddMinutes(2))),
            [H0.AddHours(1)] = null, // missing hour -> skipped
            [H0.AddHours(2)] = Zstd(BuyLine("C", H0.AddHours(2).AddMinutes(1))),
        };
        var requested = new List<DateTimeOffset>();

        Task<Stream?> Opener(DateTimeOffset hour, CancellationToken _)
        {
            requested.Add(hour);
            byte[]? bytes = archives.GetValueOrDefault(hour);
            return Task.FromResult<Stream?>(bytes is null ? null : new MemoryStream(bytes));
        }

        PumpApiReplaySource source = Source(window, Opener);
        List<RawEvent> events = await Collect(source);

        Assert.Equal(new[] { "A", "B", "C" }, events.Select(e => e.Mint));
        Assert.Equal(new[] { H0, H0.AddHours(1), H0.AddHours(2) }, requested);
        Assert.Equal(H0.AddHours(1), Assert.Single(source.MissingHours));
        Assert.Empty(source.FailedHours);
    }

    [Fact]
    public async Task Trims_events_to_the_exact_window_bounds()
    {
        var window = new ReplayWindow(H0.AddMinutes(30), H0.AddHours(1).AddMinutes(15));
        var archives = new Dictionary<DateTimeOffset, byte[]>
        {
            [H0] = Zstd(BuyLine("early", H0.AddMinutes(10)) + "\n" + BuyLine("in1", H0.AddMinutes(45))),
            [H0.AddHours(1)] = Zstd(BuyLine("in2", H0.AddHours(1).AddMinutes(10)) + "\n" + BuyLine("late", H0.AddHours(1).AddMinutes(15))),
        };

        List<RawEvent> events = await Collect(Source(window, (h, _) => Task.FromResult<Stream?>(new MemoryStream(archives[h]))));

        Assert.Equal(new[] { "in1", "in2" }, events.Select(e => e.Mint)); // end is exclusive
    }

    [Fact]
    public void Inverted_or_empty_window_is_rejected()
    {
        Assert.Throws<ArgumentException>(() => Source(new ReplayWindow(H0.AddHours(2), H0.AddHours(1)), NoArchives));
        Assert.Throws<ArgumentException>(() => Source(new ReplayWindow(H0, H0), NoArchives));
    }

    [Fact]
    public void Default_window_is_rejected()
    {
        Assert.Throws<ArgumentException>(() => Source(default, NoArchives));
    }

    [Fact]
    public void Window_before_the_archive_start_is_rejected()
    {
        var ex = Assert.Throws<ArgumentException>(() => Source(new ReplayWindow(H0.AddDays(-1), H0.AddHours(1)), NoArchives));
        Assert.Contains("before the archive", ex.Message);
    }

    [Fact]
    public void Non_positive_attempt_count_is_rejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PumpApiReplaySource(new ReplayWindow(H0, H0.AddHours(1)), NoArchives, NullLogger<PumpApiReplaySource>.Instance, maxAttemptsPerHour: 0));
    }

    [Fact]
    public async Task Pre_cancelled_token_throws_and_opens_nothing()
    {
        var requested = new List<DateTimeOffset>();
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(2)), (h, _) =>
        {
            requested.Add(h);
            return Task.FromResult<Stream?>(null);
        });
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Collect(source, cts.Token));
        Assert.Empty(requested);
    }

    [Fact]
    public async Task Cancelling_mid_hour_throws_rather_than_ending_quietly()
    {
        byte[] archive = Zstd(BuyLine("A", H0.AddMinutes(1)) + "\n" + BuyLine("B", H0.AddMinutes(2)));
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(1)), (_, _) => Task.FromResult<Stream?>(new MemoryStream(archive)));
        using var cts = new CancellationTokenSource();

        var events = new List<RawEvent>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (RawEvent e in source.StreamAsync(cts.Token))
            {
                events.Add(e);
                await cts.CancelAsync();
            }
        });
        Assert.Equal("A", Assert.Single(events).Mint);
    }

    [Fact]
    public async Task Transient_open_failure_is_retried()
    {
        int calls = 0;
        var delays = new List<TimeSpan>();
        byte[] archive = Zstd(BuyLine("A", H0.AddMinutes(1)));
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(1)), (_, _) =>
            ++calls == 1
                ? Task.FromException<Stream?>(new HttpRequestException("503 Service Unavailable"))
                : Task.FromResult<Stream?>(new MemoryStream(archive)),
            delays);

        List<RawEvent> events = await Collect(source);

        Assert.Equal("A", Assert.Single(events).Mint);
        Assert.Equal(2, calls);
        Assert.Single(delays);
        Assert.Empty(source.FailedHours);
    }

    [Fact]
    public async Task Dropped_download_resumes_without_duplicating_events()
    {
        // Enough varied lines for several zstd blocks, so some events decode before the drop.
        string[] mints = Enumerable.Range(0, 5000).Select(i => $"M{i}-{Guid.NewGuid():N}").ToArray();
        byte[] archive = Zstd(string.Join("\n", mints.Select((m, i) => BuyLine(m, H0.AddMilliseconds(i)))));
        int calls = 0;
        int yieldedBeforeRetry = -1;
        var events = new List<RawEvent>();
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(1)), (_, _) =>
        {
            if (++calls == 1)
            {
                return Task.FromResult<Stream?>(new FailingStream(archive, failAfter: archive.Length * 3 / 5));
            }

            yieldedBeforeRetry = events.Count;
            return Task.FromResult<Stream?>(new MemoryStream(archive));
        });

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await foreach (RawEvent e in source.StreamAsync(cts.Token))
        {
            events.Add(e);
        }

        Assert.Equal(2, calls);
        Assert.InRange(yieldedBeforeRetry, 1, mints.Length - 1); // the resume path really ran
        Assert.Equal(mints, events.Select(e => e.Mint));        // every event once, in order
    }

    [Fact]
    public async Task Persistently_broken_hour_is_skipped_and_recorded_and_the_replay_continues()
    {
        byte[] good = Zstd(BuyLine("NEXT", H0.AddHours(1).AddMinutes(1)));
        byte[] truncated = Zstd(string.Join("\n", Enumerable.Range(0, 200).Select(i => BuyLine($"T{i}", H0.AddSeconds(i)))));
        truncated = truncated[..(truncated.Length / 2)];
        int brokenCalls = 0;
        var delays = new List<TimeSpan>();
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(2)), (h, _) =>
        {
            if (h == H0)
            {
                brokenCalls++;
                return Task.FromResult<Stream?>(new MemoryStream(truncated));
            }

            return Task.FromResult<Stream?>(new MemoryStream(good));
        }, delays);

        List<RawEvent> events = await Collect(source);

        Assert.Equal("NEXT", Assert.Single(events, e => e.Mint == "NEXT").Mint);
        Assert.Equal(PumpApiReplaySource.DefaultMaxAttemptsPerHour, brokenCalls);
        Assert.Equal(PumpApiReplaySource.DefaultMaxAttemptsPerHour - 1, delays.Count);
        Assert.Equal(H0, Assert.Single(source.FailedHours).Hour);
    }

    [Fact]
    public async Task Persistent_http_error_is_skipped_and_recorded()
    {
        PumpApiReplaySource source = Source(new ReplayWindow(H0, H0.AddHours(1)),
            (_, _) => Task.FromException<Stream?>(new HttpRequestException("403 Forbidden")));

        List<RawEvent> events = await Collect(source);

        Assert.Empty(events);
        ReplayHourFailure failure = Assert.Single(source.FailedHours);
        Assert.Equal(H0, failure.Hour);
        Assert.Contains("403", failure.Error);
    }

    private static Task<Stream?> NoArchives(DateTimeOffset hour, CancellationToken ct) => Task.FromResult<Stream?>(null);

    private static PumpApiReplaySource Source(
        ReplayWindow window,
        Func<DateTimeOffset, CancellationToken, Task<Stream?>> opener,
        List<TimeSpan>? delays = null) =>
        new(window, opener, NullLogger<PumpApiReplaySource>.Instance, PumpApiReplaySource.DefaultMaxAttemptsPerHour, (d, _) =>
        {
            delays?.Add(d);
            return Task.CompletedTask;
        });

    private static string BuyLine(string mint, DateTimeOffset at) =>
        $$"""{"txType":"buy","mint":"{{mint}}","pool":"pump","price":0.001,"timestamp":{{at.ToUnixTimeMilliseconds()}}}""";

    private static byte[] Zstd(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        using var output = new MemoryStream();
        using (var compressor = new CompressionStream(output))
        {
            compressor.Write(bytes, 0, bytes.Length);
        }
        return output.ToArray();
    }

    private static async Task<List<RawEvent>> Collect(IDataSource source, CancellationToken ct = default)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(10)); // fail fast rather than hang
        var events = new List<RawEvent>();
        await foreach (RawEvent e in source.StreamAsync(timeout.Token))
        {
            events.Add(e);
        }
        return events;
    }

    /// <summary>Serves the first <c>failAfter</c> bytes, then fails like a reset connection.</summary>
    private sealed class FailingStream(byte[] data, int failAfter) : MemoryStream(data, 0, failAfter)
    {
        public override int Read(byte[] buffer, int offset, int count) =>
            Position >= Length ? throw new IOException("connection reset") : base.Read(buffer, offset, count);

        public override int Read(Span<byte> buffer) =>
            Position >= Length ? throw new IOException("connection reset") : base.Read(buffer);

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            Position >= Length ? throw new IOException("connection reset") : base.ReadAsync(buffer, cancellationToken);

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            Position >= Length ? throw new IOException("connection reset") : base.ReadAsync(buffer, offset, count, cancellationToken);
    }
}
