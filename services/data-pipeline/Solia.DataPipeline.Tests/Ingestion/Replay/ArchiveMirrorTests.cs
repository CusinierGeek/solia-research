using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion.Replay;
using static Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public sealed class ArchiveMirrorTests : IDisposable
{
    private static readonly DateTimeOffset H = new(2026, 4, 20, 7, 0, 0, TimeSpan.Zero);

    private readonly string _dir = NewTempDir();

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private ArchiveMirror Mirror(Func<DateTimeOffset, CancellationToken, Task<Stream?>> download) =>
        new(_dir, download, NullLogger<ArchiveMirror>.Instance);

    private static async Task<byte[]> ReadAll(Stream? stream)
    {
        Assert.NotNull(stream);
        await using (stream)
        {
            using var copy = new MemoryStream();
            await stream!.CopyToAsync(copy);
            return copy.ToArray();
        }
    }

    private string[] AllFiles() => Directory.GetFiles(_dir, "*", SearchOption.AllDirectories);

    [Fact]
    public void Lays_hours_out_like_the_source()
    {
        Assert.Equal(Path.Combine(Path.GetFullPath(_dir), "2026", "04", "20", "07.jsonl.zst"), Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?>())).PathFor(H));
    }

    [Fact]
    public void Layout_is_culture_invariant()
    {
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?>()));
        string expected = Path.Combine(Path.GetFullPath(_dir), "2026", "04", "20", "07.jsonl.zst");

        WithCulture("th-TH", () => Assert.Equal(expected, mirror.PathFor(H))); // Buddhist calendar: year 2569
    }

    [Fact]
    public async Task Counts_only_hours_not_yet_mirrored()
    {
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?> { [H] = Zstd(BuyLine("A", H)) }));
        DateTimeOffset[] hours = [H, H.AddHours(1), H.AddHours(2)];
        Assert.Equal(3, mirror.CountAbsent(hours));

        await mirror.EnsureAsync(H, default);

        Assert.Equal(2, mirror.CountAbsent(hours));
    }

    [Fact]
    public async Task Miss_downloads_atomically_then_hit_reads_from_disk_only()
    {
        byte[] archive = Zstd(BuyLine("A", H));
        var requested = new List<DateTimeOffset>();
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?> { [H] = archive }, requested));

        Assert.Equal(archive, await ReadAll(await mirror.OpenAsync(H, default)));
        Assert.Equal(new[] { mirror.PathFor(H) }, AllFiles()); // final name only, no .part left over
        Assert.Equal(1, mirror.Downloads);

        ArchiveMirror again = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?> { [H] = archive }, requested));
        Assert.Equal(archive, await ReadAll(await again.OpenAsync(H, default)));
        Assert.Single(requested); // second run never downloaded
        Assert.Equal(1, again.Hits);
        Assert.Equal(0, again.Downloads);
    }

    [Fact]
    public async Task Missing_hour_returns_null_and_writes_nothing()
    {
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?>()));

        Assert.Null(await mirror.OpenAsync(H, default));
        Assert.Equal(MirrorOutcome.Missing, await mirror.EnsureAsync(H, default));
        Assert.Empty(AllFiles());
    }

    [Fact]
    public async Task Dropped_download_leaves_no_partial_or_final_file()
    {
        ArchiveMirror mirror = Mirror((_, _) => Task.FromResult<Stream?>(new DroppingStream(new byte[1000])));

        await Assert.ThrowsAsync<IOException>(() => mirror.EnsureAsync(H, default));

        Assert.Empty(AllFiles());
        Assert.Equal(0, mirror.Downloads);
    }

    [Fact]
    public async Task Stale_part_file_from_an_earlier_crash_is_overwritten()
    {
        string path = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?>())).PathFor(H);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllTextAsync(path + ".part", "half-written garbage");
        byte[] archive = Zstd(BuyLine("A", H));

        Assert.Equal(MirrorOutcome.Downloaded,
            await Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?> { [H] = archive })).EnsureAsync(H, default));

        Assert.Equal(archive, await File.ReadAllBytesAsync(path));
        Assert.Equal(new[] { path }, AllFiles());
    }

    [Fact]
    public async Task Reopening_an_hour_already_served_from_disk_redownloads_it_as_corrupt()
    {
        byte[] good = Zstd(BuyLine("A", H));
        var requested = new List<DateTimeOffset>();
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?> { [H] = good }, requested));
        string path = mirror.PathFor(H);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, [1, 2, 3]); // corrupt file on disk

        Assert.Equal(new byte[] { 1, 2, 3 }, await ReadAll(await mirror.OpenAsync(H, default))); // first: trusted hit
        Assert.Empty(requested);

        Assert.Equal(good, await ReadAll(await mirror.OpenAsync(H, default))); // retry: replaced
        Assert.Single(requested);
        Assert.Equal(good, await File.ReadAllBytesAsync(path));
        Assert.Equal(new[] { path }, AllFiles());
    }

    [Fact]
    public async Task Redownload_of_an_hour_gone_at_the_source_throws_and_keeps_the_copy_on_disk()
    {
        ArchiveMirror mirror = Mirror(Opener(new Dictionary<DateTimeOffset, byte[]?>())); // source answers 404
        string path = mirror.PathFor(H);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, [1, 2, 3]);
        await ReadAll(await mirror.OpenAsync(H, default)); // served from disk, then "fails to read"

        await Assert.ThrowsAsync<InvalidDataException>(() => mirror.OpenAsync(H, default));

        Assert.Equal(new byte[] { 1, 2, 3 }, await File.ReadAllBytesAsync(path));
    }

    [Fact]
    public async Task Failed_redownload_keeps_the_copy_on_disk()
    {
        ArchiveMirror mirror = Mirror((_, _) => Task.FromResult<Stream?>(new DroppingStream([7, 7])));
        string path = mirror.PathFor(H);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await File.WriteAllBytesAsync(path, [1, 2, 3]);
        await ReadAll(await mirror.OpenAsync(H, default)); // trusted hit, no download

        await Assert.ThrowsAsync<IOException>(() => mirror.OpenAsync(H, default));

        Assert.Equal(new byte[] { 1, 2, 3 }, await File.ReadAllBytesAsync(path));
        Assert.Equal(new[] { path }, AllFiles()); // no .part left over
    }

    [Fact]
    public void Directory_check_accepts_a_writable_dir_and_rejects_a_missing_one()
    {
        Assert.Null(ArchiveMirror.CheckDirectory(_dir));
        Assert.Empty(AllFiles()); // the write probe is cleaned up

        string missing = Path.Combine(_dir, "nope");
        Assert.Contains("does not exist", ArchiveMirror.CheckDirectory(missing));
        Assert.False(Directory.Exists(missing));
    }

    [Fact]
    public void Directory_check_rejects_a_path_that_is_a_file()
    {
        string file = Path.Combine(_dir, "a-file");
        File.WriteAllText(file, "x");

        Assert.NotNull(ArchiveMirror.CheckDirectory(file));
    }

    [Fact]
    public void Disk_use_warning_states_the_estimate()
    {
        string warning = ArchiveMirror.DiskUseWarning(_dir, 72);

        Assert.Contains("72 absent hour(s)", warning);
        Assert.Contains("28.1 GiB", warning); // 72 × 400 MiB
        Assert.Contains("free", warning);
    }
}
