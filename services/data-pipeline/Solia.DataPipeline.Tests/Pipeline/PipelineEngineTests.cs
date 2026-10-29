using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

/// <summary>Engine behavior with an in-memory source and sink (no database).</summary>
public class PipelineEngineTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    internal static RawEvent Buy(string mint, double offsetSec) => new()
    {
        Mint = mint,
        Kind = EventKind.Buy,
        Timestamp = T0.AddSeconds(offsetSec),
        Price = 1m,
        QuoteAmount = 1m,
        Trader = $"w{offsetSec}",
    };

    private static IEnumerable<RawEvent> Seconds(string mint, int from, int to) =>
        Enumerable.Range(from, to - from + 1).Select(i => Buy(mint, i));

    [Fact]
    public async Task Flushes_every_cadence_and_once_more_for_the_last_window()
    {
        var sink = new RecordingSink();

        await Engine(Seconds("M", 0, 25), sink).RunAsync(default);

        Assert.Equal(new[] { 10.0, 20.0, 25.0 }, sink.Offsets());
    }

    [Fact]
    public async Task No_final_snapshot_when_the_last_event_already_flushed()
    {
        var sink = new RecordingSink();

        await Engine(Seconds("M", 0, 20), sink).RunAsync(default);

        Assert.Equal(new[] { 10.0, 20.0 }, sink.Offsets()); // no duplicate at 20 with zeroed growth
    }

    [Fact]
    public async Task A_single_far_future_event_does_not_fast_forward_the_clock()
    {
        var sink = new RecordingSink();
        RawEvent[] events = [.. Seconds("M", 0, 5), Buy("BOGUS", 86_400), .. Seconds("M", 6, 15)];

        await Engine(events, sink).RunAsync(default);

        Assert.Equal(new[] { 10.0, 15.0 }, sink.Offsets());
        Assert.All(sink.Batches, b => Assert.Contains(b.Snapshots, s => s.Mint == "M")); // window not wiped
    }

    [Fact]
    public async Task A_real_gap_in_time_is_followed_after_confirmation()
    {
        var sink = new RecordingSink();
        RawEvent[] events = [.. Seconds("M", 0, 5), .. Seconds("M", 3_600, 3_620)]; // e.g. a missing replay hour

        await Engine(events, sink).RunAsync(default);

        Assert.Contains(sink.Offsets(), o => o >= 3_600); // flushing resumed after the gap
    }

    [Fact]
    public async Task Late_events_never_move_the_clock_backwards()
    {
        var sink = new RecordingSink();
        RawEvent[] events = [.. Seconds("M", 0, 12), Buy("M", 3), .. Seconds("M", 13, 30)];

        await Engine(events, sink).RunAsync(default);

        double[] offsets = sink.Offsets();
        Assert.Equal(offsets.Order(), offsets);
        Assert.Equal(offsets.Distinct(), offsets);
    }

    [Fact]
    public async Task A_transient_write_failure_is_retried()
    {
        var sink = new RecordingSink { Fail = call => call == 1 ? new IOException("db blip") : null };
        var delays = new List<TimeSpan>();

        PipelineEngine engine = Engine(Seconds("M", 0, 10), sink, delays);
        await engine.RunAsync(default);

        Assert.Equal(new[] { 10.0 }, sink.Offsets());
        Assert.Single(delays);
        Assert.Equal(0, engine.FailedBatches);
    }

    [Fact]
    public async Task A_batch_that_keeps_failing_is_dropped_and_ingestion_continues()
    {
        int call = 0;
        var sink = new RecordingSink { Fail = _ => ++call <= PipelineEngine.MaxWriteAttempts ? new IOException("db down") : null };

        PipelineEngine engine = Engine(Seconds("M", 0, 25), sink);
        await engine.RunAsync(default); // doesn't throw

        Assert.Equal(1, engine.FailedBatches);
        Assert.Equal(new[] { 20.0, 25.0 }, sink.Offsets()); // the 10 s batch is lost, later ones land
    }

    [Fact]
    public async Task A_db_timeout_is_a_write_failure_not_a_shutdown()
    {
        var sink = new RecordingSink { Fail = call => call == 1 ? new TaskCanceledException("command timeout") : null };

        PipelineEngine engine = Engine(Seconds("M", 0, 25), sink);
        await engine.RunAsync(default);

        Assert.Equal(new[] { 10.0, 20.0, 25.0 }, sink.Offsets()); // kept ingesting after the timeout
    }

    [Fact]
    public async Task A_source_failure_propagates_after_the_final_flush()
    {
        var sink = new RecordingSink();
        var source = new ListSource(Seconds("M", 0, 5)) { ThrowAtEnd = new OperationCanceledException("source timeout") };

        await Assert.ThrowsAsync<OperationCanceledException>(() => Engine(source, sink).RunAsync(default));

        Assert.Equal(new[] { 5.0 }, sink.Offsets()); // the last window was still persisted
    }

    [Fact]
    public async Task Shutdown_mid_write_persists_the_interrupted_batch_instead_of_resnapshotting()
    {
        using var cts = new CancellationTokenSource();
        var sink = new RecordingSink
        {
            Fail = call =>
            {
                if (call != 1)
                {
                    return null;
                }

                cts.Cancel(); // shutdown arrives while the first write is in flight
                return new OperationCanceledException(cts.Token);
            },
        };
        var source = new ListSource(Seconds("M", 0, 10)) { WaitAtEnd = true };

        await Engine(source, sink).RunAsync(cts.Token);

        Assert.Equal(2, sink.Calls.Count);
        Assert.Same(sink.Calls[0], sink.Calls[1]); // same batch rewritten, not a dt=0 re-snapshot
        Assert.Single(sink.Batches);
    }

    [Fact]
    public async Task Final_flush_is_bounded_when_the_database_hangs()
    {
        var sink = new RecordingSink { Hang = true };

        PipelineEngine engine = Engine(new ListSource(Seconds("M", 0, 5)), sink, finalFlushTimeout: TimeSpan.FromMilliseconds(200));
        var sw = Stopwatch.StartNew();
        await engine.RunAsync(default).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"shutdown took {sw.Elapsed}");
        Assert.Equal(1, engine.FailedBatches);
    }

    [Fact]
    public async Task Tokens_dropped_from_the_window_are_reported_for_deactivation_once()
    {
        var sink = new RecordingSink();
        RawEvent[] events = [Buy("A", 0), .. Enumerable.Range(0, 13).Select(i => Buy("B", i * 10))];

        await Engine(events, sink, window: 60).RunAsync(default);

        Assert.Equal(1, sink.Batches.Sum(b => b.Deactivated.Count(m => m == "A")));
        Assert.DoesNotContain(sink.Batches, b => b.Deactivated.Contains("B"));
    }

    private static PipelineEngine Engine(
        IEnumerable<RawEvent> events, ITokenSink sink, List<TimeSpan>? delays = null, int window = 300) =>
        Engine(new ListSource(events), sink, delays, window: window);

    [Fact]
    public async Task Reset_deactivates_previous_runs_tokens()
    {
        var sink = new RecordingSink();

        await Engine([], sink).ResetActiveTokensAsync(default);

        Assert.Equal(1, sink.ResetCalls);
    }

    [Fact]
    public async Task Reset_is_retried_on_a_transient_failure()
    {
        var sink = new RecordingSink { FailReset = call => call == 1 ? new IOException("db blip") : null };
        var delays = new List<TimeSpan>();

        await Engine([], sink, delays).ResetActiveTokensAsync(default);

        Assert.Equal(2, sink.ResetCalls);
        Assert.Single(delays);
    }

    [Fact]
    public async Task Reset_gives_up_without_throwing_when_the_database_stays_down()
    {
        var sink = new RecordingSink { FailReset = _ => new IOException("db down") };

        await Engine([], sink).ResetActiveTokensAsync(default); // doesn't throw: ingestion still starts

        Assert.Equal(PipelineEngine.MaxWriteAttempts, sink.ResetCalls);
    }

    [Fact]
    public async Task Reset_honors_cancellation()
    {
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Engine([], new RecordingSink()).ResetActiveTokensAsync(cts.Token));
    }

    [Fact]
    public async Task Release_on_stop_deactivates_still_tracked_tokens_after_the_last_snapshot()
    {
        var sink = new RecordingSink();
        RawEvent[] events = [.. Seconds("A", 0, 12), Buy("B", 13)];

        await Engine(new ListSource(events), sink, releaseTracked: true).RunAsync(default);

        FlushBatch last = sink.Batches[^1];
        Assert.Empty(last.Snapshots); // its own batch: deactivating alongside the upsert would be undone by it
        Assert.Equal(new[] { "A", "B" }, last.Deactivated.Order());
        Assert.Equal(T0.AddSeconds(13), last.Ts); // as of the last event, so newer live rows are untouched
        Assert.Contains(sink.Batches[^2].Snapshots, s => s.Mint == "B"); // the final window was persisted first
    }

    [Fact]
    public async Task Release_on_stop_still_runs_when_the_last_event_already_flushed()
    {
        var sink = new RecordingSink();

        await Engine(new ListSource(Seconds("M", 0, 20)), sink, releaseTracked: true).RunAsync(default);

        Assert.Equal(new[] { 10.0, 20.0, 20.0 }, sink.Batches.Select(b => (b.Ts - T0).TotalSeconds));
        Assert.Equal(new[] { "M" }, sink.Batches[^1].Deactivated);
    }

    [Fact]
    public async Task Live_engine_never_releases_tracked_tokens_on_stop()
    {
        var sink = new RecordingSink();

        await Engine(Seconds("M", 0, 25), sink).RunAsync(default);

        Assert.All(sink.Batches, b => Assert.Empty(b.Deactivated));
    }

    [Fact]
    public async Task Release_failure_counts_as_a_failed_batch()
    {
        var sink = new RecordingSink { Fail = call => call > 1 ? new IOException("db down") : null };

        PipelineEngine engine = Engine(new ListSource(Seconds("M", 0, 5)), sink, releaseTracked: true);
        await engine.RunAsync(default);

        Assert.Equal(1, engine.FailedBatches);
    }

    internal static PipelineEngine Engine(
        IDataSource source, ITokenSink sink, List<TimeSpan>? delays = null, TimeSpan? finalFlushTimeout = null, int window = 300,
        bool releaseTracked = false)
    {
        var options = new PipelineOptions { SnapshotCadenceSeconds = 10, WindowSeconds = window };
        return new PipelineEngine(
            source,
            new TokenWindowStore(options),
            sink,
            options,
            NullLogger<PipelineEngine>.Instance,
            (d, _) =>
            {
                delays?.Add(d);
                return Task.CompletedTask;
            },
            finalFlushTimeout ?? TimeSpan.FromSeconds(5))
        {
            ReleaseTrackedOnStop = releaseTracked,
        };
    }

    internal sealed class ListSource(IEnumerable<RawEvent> events) : IDataSource
    {
        public Exception? ThrowAtEnd { get; init; }

        /// <summary>Like a live stream: after the events, wait for cancellation instead of ending.</summary>
        public bool WaitAtEnd { get; init; }

        public async IAsyncEnumerable<RawEvent> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
        {
            foreach (RawEvent e in events)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return e;
            }

            if (ThrowAtEnd is not null)
            {
                throw ThrowAtEnd;
            }

            if (WaitAtEnd)
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }
        }
    }

    internal sealed class RecordingSink : ITokenSink
    {
        /// <summary>Given the 1-based reset call number, the exception to throw (null = succeed).</summary>
        public Func<int, Exception?>? FailReset { get; init; }

        public int ResetCalls { get; private set; }

        /// <summary>Every sink operation in order: "reset" or "write".</summary>
        public ConcurrentQueue<string> Operations { get; } = new();

        public Task DeactivateAllAsync(CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ResetCalls++;
            Operations.Enqueue("reset");
            return FailReset?.Invoke(ResetCalls) is { } error ? Task.FromException(error) : Task.CompletedTask;
        }

        /// <summary>Given the 1-based call number, the exception to throw (null = succeed).</summary>
        public Func<int, Exception?>? Fail { get; init; }

        public bool Hang { get; init; }

        public List<FlushBatch> Calls { get; } = [];

        public ConcurrentQueue<FlushBatch> Written { get; } = new();

        public IReadOnlyList<FlushBatch> Batches => Written.ToList();

        public async Task WriteAsync(FlushBatch batch, CancellationToken cancellationToken)
        {
            Calls.Add(batch);
            if (Hang)
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }

            if (Fail?.Invoke(Calls.Count) is { } error)
            {
                throw error;
            }

            Written.Enqueue(batch);
            Operations.Enqueue("write");
        }

        public double[] Offsets() => Written.Select(b => (b.Ts - T0).TotalSeconds).ToArray();
    }
}
