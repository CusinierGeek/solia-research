using System.Runtime.ExceptionServices;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Metadata;
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
/// <para>
/// With an <see cref="IWalletStateStore"/> (live ingestion only), the cross-token wallet memory is loaded before the
/// first event and its changes are saved every <see cref="PipelineOptions.WalletPersistSeconds"/> of event time and
/// at shutdown, in batches — never per trade. Both are best-effort like a write: a failed load starts from an empty
/// memory, a failed save is retried at the next one.
/// </para>
/// <para>
/// With an <see cref="IMetadataStore"/>, the stored metadata fetch results (FE-T12) are loaded into the store's
/// <see cref="MetadataCache"/> before the first event: the last <see cref="MetadataCache.Horizon"/> for live ingestion,
/// or <see cref="MetadataRange"/> for a replay. Best-effort too: a failed load leaves the link features NULL.
/// </para>
/// </summary>
public sealed class PipelineEngine
{
    internal const int MaxWriteAttempts = 3;
    internal const int ClockJumpConfirmations = 5;
    internal static readonly TimeSpan DefaultFinalFlushTimeout = TimeSpan.FromSeconds(10);
    private static readonly RetryPolicy WriteRetry = new(MaxWriteAttempts);

    private readonly IDataSource _source;
    private readonly TokenWindowStore _store;
    private readonly ITokenSink _sink;
    private readonly PipelineOptions _options;
    private readonly ILogger<PipelineEngine> _logger;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly TimeSpan _finalFlushTimeout;
    private readonly IWalletStateStore? _walletState;
    private readonly IMetadataStore? _metadataStore;

    private FlushBatch? _pending; // snapshotted but not yet persisted

    public PipelineEngine(
        IDataSource source,
        TokenWindowStore store,
        ITokenSink sink,
        IOptions<PipelineOptions> options,
        ILogger<PipelineEngine> logger,
        IWalletStateStore? walletState = null,
        IMetadataStore? metadataStore = null)
        : this(source, store, sink, options.Value, logger, static (d, ct) => Task.Delay(d, ct), DefaultFinalFlushTimeout,
            walletState, metadataStore)
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
        TimeSpan finalFlushTimeout,
        IWalletStateStore? walletState = null,
        IMetadataStore? metadataStore = null)
    {
        _source = source;
        _store = store;
        _sink = sink;
        _options = options;
        _logger = logger;
        _delay = delay;
        _finalFlushTimeout = finalFlushTimeout;
        _walletState = walletState is not null && store.Wallets.TracksChanges ? walletState : null;
        _metadataStore = metadataStore;
    }

    /// <summary>
    /// Replay: the fetch times whose metadata results are loaded (the replayed span, from one horizon before it). Null
    /// for live ingestion, which loads the last <see cref="MetadataCache.Horizon"/>.
    /// </summary>
    public (DateTimeOffset Since, DateTimeOffset Until)? MetadataRange { get; init; }

    /// <summary>Batches dropped after exhausting write retries (or a timed-out final flush).</summary>
    public long FailedBatches { get; private set; }

    /// <summary>
    /// When set (finite replays), the final flush also marks every token still tracked inactive, as of
    /// the last event: nothing will update them once the run ends, so they must not stay live. Never
    /// set for live ingestion, whose tokens stay active until pruned or the next restart's reset.
    /// </summary>
    public bool ReleaseTrackedOnStop { get; init; }

    public async Task RunAsync(CancellationToken cancellationToken)
    {
        var cadence = TimeSpan.FromSeconds(_options.SnapshotCadenceSeconds);
        var clock = new EventClock(TokenWindowStore.MaxFutureSkew, ClockJumpConfirmations);
        DateTimeOffset? windowStart = null; // start of the current cadence interval
        DateTimeOffset? lastFlushAt = null;
        DateTimeOffset? lastWalletSave = null;
        var walletCadence = TimeSpan.FromSeconds(_options.WalletPersistSeconds);
        long events = 0;
        Exception? failure = null;

        _logger.LogInformation("pipeline engine started (cadence={Cadence}s)", _options.SnapshotCadenceSeconds);
        try
        {
            await LoadWalletsAsync(cancellationToken);
            await LoadMetadataAsync(cancellationToken);

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

                    lastWalletSave ??= now;
                    if (now - lastWalletSave >= walletCadence)
                    {
                        lastWalletSave = now;
                        await SaveWalletsAsync(cancellationToken);
                    }
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
        await FinalWalletSaveAsync();
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

    /// <summary>Restores the saved wallet memory (live only); on failure starts from an empty one, which only costs coverage.</summary>
    private async Task LoadWalletsAsync(CancellationToken cancellationToken)
    {
        if (_walletState is null)
        {
            return;
        }

        WalletStateSnapshot? saved = null;
        bool loaded = await WithRetriesAsync(async ct => saved = await _walletState.LoadAsync(ct), "wallet state load", cancellationToken);
        if (!loaded)
        {
            _logger.LogWarning("could not load the wallet state; starting from an empty one (wallet features stay NULL until re-covered)");
        }
        else if (saved is not null)
        {
            _store.Wallets.Load(saved);
            _logger.LogInformation("loaded {Count} wallets (history since {Start:O}, last event {Last:O})",
                saved.Wallets.Count, saved.HistoryStart, saved.LastEvent);
        }
    }

    /// <summary>Loads the stored metadata fetch results; on failure the link features stay NULL (only costs coverage).</summary>
    private async Task LoadMetadataAsync(CancellationToken cancellationToken)
    {
        if (_metadataStore is null)
        {
            return;
        }

        DateTimeOffset since = MetadataRange?.Since ?? DateTimeOffset.UtcNow - _store.Metadata.Horizon;
        DateTimeOffset? until = MetadataRange?.Until;
        IReadOnlyList<TokenMetadataRecord> records = [];
        if (await WithRetriesAsync(async ct => records = await _metadataStore.LoadAsync(since, until, ct), "metadata load", cancellationToken))
        {
            _store.Metadata.Load(records);
            _logger.LogInformation("loaded {Count} token metadata results fetched since {Since:O}", records.Count, since);
        }
        else
        {
            _logger.LogWarning("could not load the token metadata results; link features stay NULL for those tokens");
        }
    }

    /// <summary>Saves the last wallet changes at shutdown, bounded like the final flush.</summary>
    private async Task FinalWalletSaveAsync()
    {
        using var timeout = new CancellationTokenSource(_finalFlushTimeout);
        try
        {
            await SaveWalletsAsync(timeout.Token);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            _logger.LogWarning("final wallet state save timed out after {Timeout}; the changes since the last save are lost", _finalFlushTimeout);
        }
    }

    /// <summary>Saves the wallets changed or evicted since the last save; a failed save keeps them for the next one.</summary>
    private async Task SaveWalletsAsync(CancellationToken cancellationToken)
    {
        if (_walletState is null || _store.Wallets.DrainChanges() is not { } changes)
        {
            return;
        }

        if (await WithRetriesAsync(ct => _walletState.SaveAsync(changes, ct), "wallet state save", cancellationToken))
        {
            // Size of the memory, for the bound measured on live traffic (docs/DATA-PIPELINE.md).
            _logger.LogInformation("saved wallet state: {Upserts} upserted, {Deletes} evicted; {Count} wallets in memory, managed heap {HeapMb} MB",
                changes.Upserts.Count, changes.Deletes.Count, _store.Wallets.Count, GC.GetTotalMemory(false) / (1024 * 1024));
        }
        else
        {
            _store.Wallets.RestoreChanges(changes);
            _logger.LogError("wallet state save failed after {Attempts} attempts; retrying with the next save", MaxWriteAttempts);
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
    private Task<bool> WithRetriesAsync(
        Func<CancellationToken, Task> operation, string what, CancellationToken cancellationToken) =>
        WriteRetry.RunAsync(operation, _delay, (ex, attempt, wait) =>
        {
            if (wait is null)
            {
                _logger.LogWarning(ex, "{What} failed (attempt {Attempt}/{Max}); giving up", what, attempt, MaxWriteAttempts);
            }
            else
            {
                _logger.LogWarning(ex, "{What} failed (attempt {Attempt}/{Max}); retrying in {Delay}",
                    what, attempt, MaxWriteAttempts, wait);
            }
        }, cancellationToken);

    /// <summary>
    /// Persists the last window. Writes an interrupted batch as-is; otherwise snapshots once more,
    /// unless nothing changed since the last flush. Bounded so an unreachable DB can't block shutdown.
    /// </summary>
    private async Task FinalFlushAsync(DateTimeOffset? now, DateTimeOffset? lastFlushAt)
    {
        using var timeout = new CancellationTokenSource(_finalFlushTimeout);
        try
        {
            if (_pending is not null)
            {
                await WritePendingAsync(timeout.Token);
            }
            else if (now is { } ts && ts != lastFlushAt)
            {
                _pending = new FlushBatch(ts, _store.SnapshotActive(ts), _store.DrainRemoved());
                await WritePendingAsync(timeout.Token);
            }

            // A separate batch, after the snapshots: within one batch deactivation runs first and the
            // snapshot upsert would mark the tokens active again.
            if (ReleaseTrackedOnStop && now is { } end && _store.TrackedCount > 0)
            {
                _pending = new FlushBatch(end, [], _store.TrackedMints());
                await WritePendingAsync(timeout.Token);
            }
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            FailedBatches++;
            _logger.LogWarning("final flush timed out after {Timeout}; last window not persisted", _finalFlushTimeout);
        }
    }
}
