using System.Globalization;
using Microsoft.Extensions.Logging;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>What <see cref="ArchiveMirror.EnsureAsync"/> found or did for an hour.</summary>
public enum MirrorOutcome
{
    /// <summary>Already in the mirror; nothing downloaded.</summary>
    Present,

    /// <summary>Downloaded into the mirror.</summary>
    Downloaded,

    /// <summary>The source has no archive for that hour (nothing written).</summary>
    Missing,
}

/// <summary>
/// Opt-in local mirror of the hourly replay archives, laid out like the source
/// (<c>{root}/yyyy/MM/dd/HH.jsonl.zst</c>), so a window is downloaded once and re-replayed from disk.
/// Hours are served from the mirror when present and downloaded through the injected
/// <c>download</c> opener (the HTTP archive) otherwise. A download goes to a temp file renamed into
/// place only once complete, so a mirror file is never half-written; an interrupted one leaves at
/// most a <c>.part</c> file, overwritten by the next attempt.
/// <para>
/// A mirror file that fails to read (corrupt or truncated on disk) is detected by
/// <see cref="PumpApiReplaySource"/> retrying the hour: a second <see cref="OpenAsync"/> of an hour
/// already served from disk in this run downloads it again, replacing the file only once the new copy
/// is complete. If the source no longer has that hour, the copy on disk is kept and the attempt fails
/// (so the hour ends up failed, not silently "missing").
/// </para>
/// </summary>
public sealed class ArchiveMirror : IReplayArchive
{
    /// <summary>Rough compressed size of one archive hour, for the disk-use warning.</summary>
    public const long ApproxBytesPerHour = 400L * 1024 * 1024;

    private const string PartSuffix = ".part";

    private readonly string _root;
    private readonly Func<DateTimeOffset, CancellationToken, Task<Stream?>> _download;
    private readonly ILogger<ArchiveMirror> _logger;
    private readonly HashSet<DateTimeOffset> _servedFromDisk = [];

    public ArchiveMirror(
        string root,
        Func<DateTimeOffset, CancellationToken, Task<Stream?>> download,
        ILogger<ArchiveMirror> logger)
    {
        _root = Path.GetFullPath(root);
        _download = download;
        _logger = logger;
    }

    public string Root => _root;

    /// <summary>Hours read from the mirror without downloading.</summary>
    public int Hits { get; private set; }

    /// <summary>Hours downloaded into the mirror (including re-downloads of corrupt files).</summary>
    public int Downloads { get; private set; }

    /// <summary>
    /// Null when <paramref name="directory"/> exists and is writable; otherwise why it can't be a mirror.
    /// The directory isn't created: a typo shouldn't silently start a 400 MB/hour download elsewhere.
    /// </summary>
    public static string? CheckDirectory(string directory)
    {
        if (!Directory.Exists(directory))
        {
            return $"mirror directory '{directory}' does not exist (create it first).";
        }

        string probe = Path.Combine(directory, $".solia-write-probe-{Guid.NewGuid():N}");
        try
        {
            File.WriteAllBytes(probe, []);
            File.Delete(probe);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return $"mirror directory '{directory}' is not writable: {ex.Message}";
        }
    }

    /// <summary>Culture-invariant (Gregorian) layout, so every host maps an hour to the same file.</summary>
    public string PathFor(DateTimeOffset hour)
    {
        DateTimeOffset utc = hour.ToUniversalTime();
        CultureInfo c = CultureInfo.InvariantCulture;
        return Path.Combine(_root, utc.ToString("yyyy", c), utc.ToString("MM", c), utc.ToString("dd", c),
            utc.ToString("HH", c) + ".jsonl.zst");
    }

    /// <summary>How many of <paramref name="hours"/> aren't in the mirror yet (i.e. would be downloaded).</summary>
    public int CountAbsent(IEnumerable<DateTimeOffset> hours) => hours.Count(h => !File.Exists(PathFor(h)));

    /// <summary>
    /// The compressed archive for <paramref name="hour"/> from the mirror, downloading it first when
    /// absent (or when it already failed to read in this run); null when the source has no such hour.
    /// Usable as <see cref="PumpApiReplaySource"/>'s archive opener.
    /// </summary>
    public async Task<Stream?> OpenAsync(DateTimeOffset hour, CancellationToken cancellationToken)
    {
        string path = PathFor(hour);
        MirrorOutcome outcome;
        if (File.Exists(path) && _servedFromDisk.Contains(hour))
        {
            _logger.LogWarning("mirror: {Path} failed to read; downloading it again", path);
            outcome = await DownloadAsync(hour, path, cancellationToken)
                      ?? throw new InvalidDataException(
                          $"{path} failed to read and the source no longer has that hour; the copy on disk is kept.");
        }
        else
        {
            outcome = await EnsureAsync(hour, cancellationToken);
        }

        if (outcome == MirrorOutcome.Missing)
        {
            return null;
        }

        _servedFromDisk.Add(hour);
        return new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, JsonlEventReader.BufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
    }

    /// <summary>Makes sure <paramref name="hour"/> is in the mirror, downloading it atomically if absent.</summary>
    public async Task<MirrorOutcome> EnsureAsync(DateTimeOffset hour, CancellationToken cancellationToken)
    {
        string path = PathFor(hour);
        if (File.Exists(path))
        {
            Hits++;
            return MirrorOutcome.Present;
        }

        return await DownloadAsync(hour, path, cancellationToken) ?? MirrorOutcome.Missing;
    }

    /// <summary>
    /// Downloads <paramref name="hour"/> to a temp file, then atomically moves it over <paramref name="path"/>
    /// (replacing any existing copy only once the download is complete). Null when the source has no such hour.
    /// </summary>
    private async Task<MirrorOutcome?> DownloadAsync(DateTimeOffset hour, string path, CancellationToken cancellationToken)
    {
        Stream? source = await _download(hour, cancellationToken);
        if (source is null)
        {
            return null;
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string part = path + PartSuffix;
        try
        {
            await using (source)
            await using (var file = new FileStream(part, FileMode.Create, FileAccess.Write, FileShare.None,
                             JsonlEventReader.BufferSize, FileOptions.Asynchronous))
            {
                await source.CopyToAsync(file, cancellationToken);
            }

            File.Move(part, path, overwrite: true);
        }
        catch
        {
            TryDelete(part); // dropped download / cancellation: never leave a partial file behind
            throw;
        }

        Downloads++;
        _logger.LogInformation("mirror: downloaded {Hour:yyyy-MM-dd HH}:00 UTC to {Path}", hour, path);
        return MirrorOutcome.Downloaded;
    }

    /// <summary>
    /// Disk-use warning text for downloading <paramref name="hours"/> absent hours into <paramref name="directory"/>:
    /// the estimated size and, when known, the free space left on that drive.
    /// </summary>
    public static string DiskUseWarning(string directory, int hours)
    {
        long estimate = hours * ApproxBytesPerHour;
        string message = FormattableString.Invariant($"mirroring {hours} absent hour(s) needs up to ~{Gib(estimate):0.#} GiB (~400 MB/hour compressed) in '{directory}'");
        try
        {
            long free = new DriveInfo(Path.GetFullPath(directory)).AvailableFreeSpace;
            message += FormattableString.Invariant($"; {Gib(free):0.#} GiB free on that drive");
            if (free < estimate)
            {
                message += " — NOT ENOUGH SPACE for the whole window";
            }
        }
        catch (Exception ex) when (ex is IOException or ArgumentException or UnauthorizedAccessException)
        {
            // Free space unknown (e.g. a network path): the estimate alone still helps.
        }

        return message;
    }

    private static double Gib(long bytes) => bytes / (1024d * 1024 * 1024);

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Best effort: the next attempt overwrites it (FileMode.Create).
        }
    }
}
