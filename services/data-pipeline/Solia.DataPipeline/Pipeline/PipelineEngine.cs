using System.Runtime.ExceptionServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// Drives ingestion end to end: pulls <see cref="RawEvent"/>s from a data source into the
/// <see cref="TokenWindowStore"/>, and every snapshot cadence persists the active tokens'
/// feature snapshots (plus deactivated mints) through the sink. The snapshot clock is an
/// <see cref="EventClock"/> over event timestamps, so the behavior is identical for live and
/// replay and a stray timestamp can't stall or wipe it.
/// <para>
/// A failed write is retried with backoff; a batch that still fails is logged, counted in
/// <see cref="FailedBatches"/> and dropped, so a database blip never stops ingestion. Only
/// cancellation of the caller's token is a shutdown; any other exception from the source
/// propagates (after the final flush) so the host fails loudly rather than idling.
/// </para>
/// <para>
/// A bounded final flush on stream end / shutdown persists the last window (important for finite
/// replay sources). If a flush was interrupted mid-write, the final flush writes that same batch
/// rather than snapshotting again at the same instant (which would zero the growth rates).
/// </para>
/// </summary>
public sealed class PipelineEngine
{
    internal const int MaxWriteAttempts = 3;
    internal const int ClockJumpConfirmations = 5;
    internal static readonly TimeSpan DefaultFinalFlushTimeout = TimeSpan.FromSeconds(10);

    private readonly IDataSource _source;
    private readonly TokenWindowStore _store;
    private readonly ITokenSink _sink;
    private readonly PipelineOptions _options;
    private readonly ILogger<PipelineEngine> _logger;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly TimeSpan _finalFlushTimeout;

    private FlushBatch? _pending; // snapshotted but not yet persisted

    public PipelineEngine(
        IDataSource source,
        TokenWindowStore store,
        ITokenSink sink,
        IOptions<PipelineOptions> options,
        ILogger<PipelineEngine> logger)
        : this(source, store, sink, options.Value, logger, static (d, ct) => Task.Delay(d, ct), DefaultFinalFlushTimeout)
    {
    }

    // Test seam: inject the retry delay and final-flush timeout so unit tests run without real waits.
    internal PipelineEngine(
        IDataSource source,
        TokenWindowStore store,
        ITokenSink sink,
        PipelineOptions options,
        ILogger<PipelineEngine> logger,
        Func<TimeSpan, CancellationToken, Task> delay,
        TimeSpan finalFlushTimeout)
    {
        _source = source;
        _store = store;
        _sink = sink;
        _options = options;
        _logger = logger;
        _delay = delay;
        _finalFlushTimeout = finalFlushTimeout;
    }

    /// <summary>Batches dropped after exhausting write retries (or a timed-out final flush).</summary>
    public long FailedBatches { get; private set; }

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        var cadence = TimeSpan.FromSeconds(_options.SnapshotCadenceSeconds);
        var clock = new EventClock(TokenWindowStore.MaxFutureSkew, ClockJumpConfirmations);
        DateTimeOffset? windowStart = null; // start of the current cadence interval
        DateTimeOffset? lastFlushAt = null;
        long events = 0;
        Exception? failure = null;

        _logger.LogInformation("pipeline engine started (cadence={Cadence}s)", _options.SnapshotCadenceSeconds);
        try
        {
            await foreach (RawEvent e in _source.StreamAsync(cancellationToken))
            {
                _store.Apply(e);
                events++;
                DateTimeOffset now = clock.Observe(e.Timestamp);

                if (windowStart is null || now < windowStart)
                {
                    windowStart = now; // first event, or the clock was reset backwards
                }

                if (now - windowStart >= cadence)
                {
                    windowStart = now;
                    lastFlushAt = now;
                    await FlushAsync(now, cancellationToken);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // Graceful shutdown — fall through to the final flush.
        }
        catch (Exception ex)
        {
            failure = ex;
            _logger.LogError(ex, "pipeline source failed");
        }

        await FinalFlushAsync(clock.Now, lastFlushAt);
        _logger.LogInformation("pipeline engine stopped after {Events} events", events);

        if (failure is not null)
        {
            ExceptionDispatchInfo.Throw(failure);
        }
    }

    private async Task FlushAsync(DateTimeOffset now, CancellationToken cancellationToken)
    {
        _pending = new FlushBatch(now, _store.SnapshotActive(now), _store.DrainRemoved());
        await WritePendingAsync(cancellationToken);
    }

    private async Task WritePendingAsync(CancellationToken cancellationToken)
    {
        FlushBatch batch = _pending!;
        if (batch.IsEmpty)
        {
            _pending = null;
            return;
        }

        bool written = await WithRetriesAsync(ct => _sink.WriteAsync(batch, ct), "feature write", cancellationToken);
        _pending = null;
        if (written)
        {
            _logger.LogDebug("persisted {Count} feature snapshots at {Now:O}", batch.Snapshots.Count, batch.Ts);
        }
        else
        {
            FailedBatches++;
            _logger.LogError("dropped {Count} feature snapshots at {Now:O} after {Attempts} write attempts",
                batch.Snapshots.Count, batch.Ts, MaxWriteAttempts);
        }
    }

    /// <summary>
    /// Marks tokens left active by a previous run inactive (see <see cref="ITokenSink.DeactivateAllAsync"/>).
    /// Call once before <see cref="RunAsync"/> in live ingestion only — never for a replay that may run
    /// alongside live ingestion. Best-effort: retried like a write, then logged and skipped, so an
    /// unreachable database at startup doesn't stop ingestion.
    /// </summary>
    public async Task ResetActiveTokensAsync(CancellationToken cancellationToken)
    {
        if (await WithRetriesAsync(_sink.DeactivateAllAsync, "active-token reset", cancellationToken))
        {
            _logger.LogInformation("marked tokens from previous runs inactive");
        }
        else
        {
            _logger.LogWarning("could not reset active tokens; rows from previous runs may stay active until re-tracked");
        }
    }

    /// <summary>Runs a sink operation with backoff retries. False once every attempt failed; cancellation throws.</summary>
    private async Task<bool> WithRetriesAsync(
        Func<CancellationToken, Task> operation, string what, CancellationToken cancellationToken)
    {
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                await operation(cancellationToken);
                return true;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                // Includes a TaskCanceledException from a DB command timeout: a failure, not a shutdown.
                if (attempt >= MaxWriteAttempts)
                {
                    _logger.LogWarning(ex, "{What} failed (attempt {Attempt}/{Max}); giving up", what, attempt, MaxWriteAttempts);
                    return false;
                }

                TimeSpan wait = PumpApiStreamSource.ComputeBackoff(attempt - 1, 1, 30, Random.Shared.NextDouble());
                _logger.LogWarning(ex, "{What} failed (attempt {Attempt}/{Max}); retrying in {Delay}",
                    what, attempt, MaxWriteAttempts, wait);
                await _delay(wait, cancellationToken);
            }
        }
    }

    /// <summary>
    /// Persists the last window. Writes an interrupted batch as-is; otherwise snapshots once more,
    /// unless nothing changed since the last flush. Bounded so an unreachable DB can't block shutdown.
    /// </summary>
    private async Task FinalFlushAsync(DateTimeOffset? now, DateTimeOffset? lastFlushAt)
    {
        using var timeout = new CancellationTokenSource(_finalFlushTimeout);
        try
        {
            if (_pending is null)
            {
                if (now is not { } ts || ts == lastFlushAt)
                {
                    return;
                }

                _pending = new FlushBatch(ts, _store.SnapshotActive(ts), _store.DrainRemoved());
            }

            await WritePendingAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            FailedBatches++;
            _logger.LogWarning("final flush timed out after {Timeout}; last window not persisted", _finalFlushTimeout);
        }
    }
}
