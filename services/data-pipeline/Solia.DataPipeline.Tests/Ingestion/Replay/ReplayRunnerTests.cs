using System.Net;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives;
using static Solia.DataPipeline.Tests.Pipeline.PipelineEngineTests;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public sealed class ReplayRunnerTests : IDisposable
{
    private static readonly DateTimeOffset H0 = new(2026, 4, 20, 0, 0, 0, TimeSpan.Zero);
    private static readonly ReplayWindow ThreeHours = new(H0, H0.AddHours(3));
    private static readonly PipelineOptions Options = new() { SnapshotCadenceSeconds = 1, WindowSeconds = 86_400 };

    private readonly string _dir = NewTempDir();

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static ReplayRunner Runner() => new(NullLoggerFactory.Instance, (_, _) => Task.CompletedTask);

    /// <summary>
    /// Hours 00 and 02 have trades for A then B (2 s apart, so each flushes); hour 01 is missing. B has
    /// enough trades for the event clock to accept the 2 h jump (it needs several confirmations).
    /// </summary>
    private static Dictionary<DateTimeOffset, byte[]?> Archives() => new()
    {
        [H0] = Zstd(Trades("A", H0.AddMinutes(1), 2)),
        [H0.AddHours(2)] = Zstd(Trades("B", H0.AddHours(2), PipelineEngine.ClockJumpConfirmations + 3)),
    };

    private static string Trades(string mint, DateTimeOffset start, int count) =>
        string.Join("\n", Enumerable.Range(0, count).Select(i => BuyLine(mint, start.AddSeconds(2 * i))));

    private static Task<ReplaySummary> Replay(IReplayArchive archive, RecordingSink sink) =>
        Runner().ReplayAsync(ThreeHours, archive, sink, () => sink.Batches.Sum(b => b.Snapshots.Count), Options, default);

    private static Task<ReplaySummary> Replay(Func<DateTimeOffset, CancellationToken, Task<Stream?>> opener, RecordingSink sink) =>
        Replay(new DelegateArchive(opener), sink);

    private static IEnumerable<string> Mints(RecordingSink sink) =>
        sink.Batches.SelectMany(b => b.Snapshots).Select(s => s.Mint).Distinct();

    [Fact]
    public async Task Replays_through_the_engine_and_reports_missing_hours()
    {
        var sink = new RecordingSink();

        ReplaySummary summary = await Replay(Opener(Archives()), sink);

        Assert.Equal(new[] { "A", "B" }, Mints(sink));
        Assert.Equal(3, summary.Hours);
        Assert.Equal(2, summary.ProcessedHours);
        Assert.Equal(H0.AddHours(1), Assert.Single(summary.MissingHours));
        Assert.Empty(summary.FailedHours);
        Assert.True(summary.FeatureRowsWritten > 0);
        Assert.True(summary.IsComplete);
        Assert.Equal(ReplayCommand.ExitOk, ReplayCommand.ExitCodeFor(summary));
        Assert.Equal(0, sink.ResetCalls); // replay never resets live tokens
    }

    [Fact]
    public async Task Failed_hour_is_reported_and_makes_the_run_incomplete()
    {
        Dictionary<DateTimeOffset, byte[]?> archives = Archives();
        archives[H0.AddHours(2)] = [1, 2, 3]; // not zstd: fails every attempt
        var sink = new RecordingSink();

        ReplaySummary summary = await Replay(Opener(archives), sink);

        Assert.Equal(H0.AddHours(2), Assert.Single(summary.FailedHours).Hour);
        Assert.Equal(1, summary.ProcessedHours);
        Assert.False(summary.IsComplete);
        Assert.Equal(ReplayCommand.ExitIncomplete, ReplayCommand.ExitCodeFor(summary));
        Assert.Equal(new[] { "A" }, Mints(sink));
    }

    [Fact]
    public async Task Dropped_batches_make_the_run_incomplete()
    {
        var sink = new RecordingSink { Fail = _ => new InvalidOperationException("db down") };

        ReplaySummary summary = await Replay(Opener(Archives()), sink);

        Assert.True(summary.FailedBatches > 0);
        Assert.Equal(ReplayCommand.ExitIncomplete, ReplayCommand.ExitCodeFor(summary));
    }

    [Fact]
    public async Task Unreachable_source_skips_every_hour_without_crashing_the_run()
    {
        var sink = new RecordingSink();

        ReplaySummary summary = await Replay((_, _) => throw new HttpRequestException("name not resolved"), sink);

        Assert.Equal(3, summary.FailedHours.Count);
        Assert.All(summary.FailedHours, f => Assert.Contains("name not resolved", f.Error));
        Assert.Equal(0, summary.ProcessedHours);
        Assert.Empty(sink.Batches);
        Assert.Equal(ReplayCommand.ExitIncomplete, ReplayCommand.ExitCodeFor(summary));
    }

    [Fact]
    public async Task Direct_mode_streams_from_http_and_keeps_nothing_on_disk()
    {
        string[] before = ArchiveLikeFiles();
        using var http = new HttpClient(new ArchiveHandler(Archives()));
        var opener = new HttpReplayArchiveOpener(http, Microsoft.Extensions.Options.Options.Create(new PumpApiOptions()));
        var sink = new RecordingSink();

        ReplaySummary summary = await Replay(opener, sink);

        Assert.Equal(new[] { "A", "B" }, Mints(sink));
        Assert.Equal(0, summary.MirrorDownloads);
        Assert.Equal(before, ArchiveLikeFiles());
    }

    [Fact]
    public async Task Mirrored_rerun_reads_disk_only_and_yields_the_same_rows()
    {
        var requested = new List<DateTimeOffset>();
        var first = new RecordingSink();
        var mirror = new ArchiveMirror(_dir, Opener(Archives(), requested), NullLogger<ArchiveMirror>.Instance);
        ReplaySummary run1 = await Replay(mirror, first);

        var second = new RecordingSink();
        var mirror2 = new ArchiveMirror(_dir, Opener(Archives(), requested), NullLogger<ArchiveMirror>.Instance);
        ReplaySummary run2 = await Replay(mirror2, second);

        Assert.Equal((0, 2), (run1.MirrorHits, run1.MirrorDownloads));
        Assert.Equal((2, 0), (run2.MirrorHits, run2.MirrorDownloads));
        Assert.Equal(4, requested.Count); // run 1: 3 hours + run 2: only the missing hour re-asked
        Assert.Equal(
            first.Batches.SelectMany(b => b.Snapshots).Select(s => (s.Mint, s.Ts)),
            second.Batches.SelectMany(b => b.Snapshots).Select(s => (s.Mint, s.Ts)));
    }

    [Fact]
    public async Task Corrupt_mirror_file_is_redownloaded_during_replay()
    {
        var mirror = new ArchiveMirror(_dir, Opener(Archives()), NullLogger<ArchiveMirror>.Instance);
        string path = mirror.PathFor(H0);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, [9, 9, 9]);
        var sink = new RecordingSink();

        ReplaySummary summary = await Replay(mirror, sink);

        Assert.Empty(summary.FailedHours);
        Assert.Equal(new[] { "A", "B" }, Mints(sink));
        Assert.Equal(Archives()[H0], await File.ReadAllBytesAsync(path));
    }

    [Fact]
    public async Task Mirror_only_downloads_without_replaying_and_reports_hours()
    {
        Dictionary<DateTimeOffset, byte[]?> archives = Archives();
        int calls = 0;
        var mirror = new ArchiveMirror(_dir, (h, ct) =>
        {
            if (h == H0.AddHours(2) && ++calls <= 10)
            {
                return Task.FromResult<Stream?>(new DroppingStream([1, 2]));
            }

            return Opener(archives)(h, ct);
        }, NullLogger<ArchiveMirror>.Instance);

        ReplaySummary summary = await Runner().MirrorOnlyAsync(ThreeHours, mirror, default);

        Assert.Equal(3, summary.Hours);
        Assert.Equal(H0.AddHours(1), Assert.Single(summary.MissingHours));
        Assert.Equal(H0.AddHours(2), Assert.Single(summary.FailedHours).Hour);
        Assert.Equal(3, calls); // retried up to the attempt limit
        Assert.Equal(1, summary.MirrorDownloads);
        Assert.Equal(0, summary.FeatureRowsWritten);
        Assert.True(File.Exists(mirror.PathFor(H0)));
        Assert.False(File.Exists(mirror.PathFor(H0.AddHours(2))));
        Assert.Equal(ReplayCommand.ExitIncomplete, ReplayCommand.ExitCodeFor(summary));
    }

    [Fact]
    public async Task Cancellation_throws_instead_of_reporting_a_partial_run_as_done()
    {
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            Runner().ReplayAsync(ThreeHours, new DelegateArchive(Opener(Archives())), new RecordingSink(), () => 0, Options, cts.Token));
    }

    [Fact]
    public async Task Tokens_still_tracked_at_the_end_are_marked_inactive_as_of_the_last_event()
    {
        var sink = new RecordingSink();

        await Replay(Opener(Archives()), sink);

        FlushBatch last = sink.Batches[^1];
        Assert.Empty(last.Snapshots);
        Assert.Equal(new[] { "A", "B" }, last.Deactivated.Order());
        Assert.Equal(sink.Batches[^2].Ts, last.Ts); // not "now": newer live rows keep is_active
        Assert.True(last.Ts < H0.AddHours(3));
    }

    [Fact]
    public async Task Corrupt_mirror_file_whose_hour_vanished_at_the_source_fails_the_hour_and_is_kept()
    {
        Dictionary<DateTimeOffset, byte[]?> archives = Archives();
        archives.Remove(H0); // the source no longer serves hour 00
        var mirror = new ArchiveMirror(_dir, Opener(archives), NullLogger<ArchiveMirror>.Instance);
        string path = mirror.PathFor(H0);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, [9, 9, 9]);

        ReplaySummary summary = await Replay(mirror, new RecordingSink());

        Assert.Equal(H0, Assert.Single(summary.FailedHours).Hour); // failed, not "missing"
        Assert.DoesNotContain(H0, summary.MissingHours);
        Assert.False(summary.IsComplete);
        Assert.Equal(new byte[] { 9, 9, 9 }, await File.ReadAllBytesAsync(path)); // the local copy isn't destroyed
    }

    [Fact]
    public async Task Stalled_http_body_fails_the_attempt_instead_of_hanging()
    {
        using var http = new HttpClient(new StallingHandler());
        var opener = new HttpReplayArchiveOpener(http, Microsoft.Extensions.Options.Options.Create(new PumpApiOptions()),
            TimeSpan.FromMilliseconds(100));

        ReplaySummary summary = await Replay(opener, new RecordingSink()).WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(3, summary.FailedHours.Count);
        Assert.All(summary.FailedHours, f => Assert.Contains("stalled", f.Error));
    }

    private static string[] ArchiveLikeFiles() =>
        new[] { Path.GetTempPath(), AppContext.BaseDirectory, Environment.CurrentDirectory }
            .Distinct()
            .SelectMany(d => Directory.EnumerateFiles(d, "*.jsonl.zst*", SearchOption.TopDirectoryOnly))
            .Order()
            .ToArray();

    /// <summary>Answers 200 with a body that never sends a byte (a connection stalled mid-download).</summary>
    private sealed class StallingHandler : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StreamContent(new StalledStream()) });
    }

    private sealed class StalledStream : MemoryStream
    {
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            await Task.Delay(Timeout.Infinite, cancellationToken);
            return 0;
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }

    private sealed class ArchiveHandler(IReadOnlyDictionary<DateTimeOffset, byte[]?> archives) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            string[] p = request.RequestUri!.AbsolutePath.Trim('/').Split('/'); // yyyy/MM/dd/HH.jsonl.zst
            var hour = new DateTimeOffset(int.Parse(p[0]), int.Parse(p[1]), int.Parse(p[2]), int.Parse(p[3][..2]), 0, 0, TimeSpan.Zero);
            return Task.FromResult(archives.GetValueOrDefault(hour) is { } body
                ? new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) }
                : new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
