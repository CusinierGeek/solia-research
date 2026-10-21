using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging;
using Solia.DataPipeline.Ingestion.PumpApi;
using ZstdSharp;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>One archive hour that couldn't be read after every retry.</summary>
public sealed record ReplayHourFailure(DateTimeOffset Hour, string Error);

/// <summary>
/// Replays a window of hourly pumpapi archives. Transport-agnostic: an injected opener returns
/// the compressed archive for an hour (or <c>null</c> if that hour is missing), which this source
/// zstd-decompresses and decodes through the same parser as live, keeping only events inside the
/// window. Missing hours are skipped. Any other failure (HTTP error, dropped download, corrupt
/// or truncated archive) is retried with backoff, resuming after the last line already read so
/// nothing is emitted twice; an hour that still fails is logged, recorded in
/// <see cref="FailedHours"/> and skipped, so one bad hour doesn't end a multi-day replay.
/// Cancellation throws <see cref="OperationCanceledException"/>.
/// </summary>
public sealed class PumpApiReplaySource : IDataSource
{
    public const int DefaultMaxAttemptsPerHour = 3;

    private readonly ReplayWindow _window;
    private readonly Func<DateTimeOffset, CancellationToken, Task<Stream?>> _openArchive;
    private readonly ILogger<PumpApiReplaySource> _logger;
    private readonly int _maxAttemptsPerHour;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly List<DateTimeOffset> _missingHours = [];
    private readonly List<ReplayHourFailure> _failedHours = [];

    public PumpApiReplaySource(
        ReplayWindow window,
        Func<DateTimeOffset, CancellationToken, Task<Stream?>> openArchive,
        ILogger<PumpApiReplaySource> logger,
        int maxAttemptsPerHour = DefaultMaxAttemptsPerHour)
        : this(window, openArchive, logger, maxAttemptsPerHour, static (d, ct) => Task.Delay(d, ct))
    {
    }

    // Test seam: inject the retry delay so unit tests run without real waits.
    internal PumpApiReplaySource(
        ReplayWindow window,
        Func<DateTimeOffset, CancellationToken, Task<Stream?>> openArchive,
        ILogger<PumpApiReplaySource> logger,
        int maxAttemptsPerHour,
        Func<TimeSpan, CancellationToken, Task> delay)
    {
        window.EnsureValid();
        ArgumentOutOfRangeException.ThrowIfLessThan(maxAttemptsPerHour, 1);

        _window = window;
        _openArchive = openArchive;
        _logger = logger;
        _maxAttemptsPerHour = maxAttemptsPerHour;
        _delay = delay;
    }

    /// <summary>Hours with no archive (404), from the latest enumeration.</summary>
    public IReadOnlyList<DateTimeOffset> MissingHours => _missingHours;

    /// <summary>Hours skipped after exhausting retries, from the latest enumeration.</summary>
    public IReadOnlyList<ReplayHourFailure> FailedHours => _failedHours;

    public async IAsyncEnumerable<RawEvent> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        _missingHours.Clear();
        _failedHours.Clear();

        foreach (DateTimeOffset hour in _window.Hours())
        {
            long linesDone = 0; // lines of this hour already processed; a retry skips past them

            for (int attempt = 1; ; attempt++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                Exception? failure = null;
                Stream? archive = null;

                try
                {
                    archive = await _openArchive(hour, cancellationToken);
                }
                catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
                {
                    failure = ex;
                }

                if (failure is null && archive is null)
                {
                    _missingHours.Add(hour);
                    _logger.LogInformation("replay: no archive for {Hour:yyyy-MM-dd HH}:00 UTC", hour);
                    break;
                }

                if (archive is not null)
                {
                    await using (archive)
                    await using (var content = new DecompressionStream(archive))
                    using (StreamReader reader = JsonlEventReader.CreateReader(content))
                    {
                        long line = 0;
                        while (true)
                        {
                            string? text;
                            try
                            {
                                text = await reader.ReadLineAsync(cancellationToken);
                            }
                            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
                            {
                                failure = ex; // dropped download, corrupt or truncated archive
                                break;
                            }

                            if (text is null)
                            {
                                break; // hour complete
                            }

                            if (++line <= linesDone)
                            {
                                continue;
                            }

                            linesDone = line;
                            if (JsonlEventReader.ParseLine(text) is { } evt && _window.Contains(evt.Timestamp))
                            {
                                yield return evt;
                            }
                        }
                    }

                    if (failure is null)
                    {
                        break;
                    }
                }

                if (attempt >= _maxAttemptsPerHour)
                {
                    _failedHours.Add(new ReplayHourFailure(hour, failure!.Message));
                    _logger.LogWarning(failure,
                        "replay: skipping {Hour:yyyy-MM-dd HH}:00 UTC after {Attempts} attempts ({Lines} lines read)",
                        hour, attempt, linesDone);
                    break;
                }

                TimeSpan wait = PumpApiStreamSource.ComputeBackoff(attempt - 1, 1, 30, Random.Shared.NextDouble());
                _logger.LogWarning(failure,
                    "replay: {Hour:yyyy-MM-dd HH}:00 UTC failed (attempt {Attempt}/{Max}); retrying in {Delay}",
                    hour, attempt, _maxAttemptsPerHour, wait);
                await _delay(wait, cancellationToken);
            }
        }
    }
}
