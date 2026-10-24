using Solia.Api.Endpoints;

namespace Solia.Api.Live;

/// <summary>
/// Turns notifications into hub events: re-reads the rows (the table is the source of truth, the NOTIFY only says
/// which row) and broadcasts them, plus <see cref="LiveEvents.StatsChanged"/> after a close. At most once and
/// best-effort: a missing row, a failed read or a broadcast that fails or outlasts the send timeout
/// (<see cref="DefaultSendTimeout"/>) drops that event with a warning and never throws — clients re-fetch over
/// REST. Each broadcast stands alone, so a failed <see cref="LiveEvents.TradeClosed"/> doesn't take its
/// <see cref="LiveEvents.StatsChanged"/> down with it.
/// </summary>
public sealed class LiveEventDispatcher(
    ILiveRowReader rows,
    ILiveBroadcaster broadcaster,
    ILogger<LiveEventDispatcher> logger,
    TimeSpan? sendTimeout = null,
    int maxStalledSends = LiveEventDispatcher.DefaultMaxStalledSends)
{
    /// <summary>How long one broadcast may take before it's abandoned (a slow client must not stall the feed).</summary>
    public static readonly TimeSpan DefaultSendTimeout = TimeSpan.FromSeconds(5);

    /// <summary>
    /// How many abandoned broadcasts may still be running (a transport that ignored the cancellation) before new
    /// events are dropped without being sent — so a stuck client can't pile up sends and payloads in memory.
    /// </summary>
    public const int DefaultMaxStalledSends = 16;

    private readonly TimeSpan _sendTimeout = sendTimeout ?? DefaultSendTimeout;

    private int _stalledSends;

    /// <summary>Abandoned broadcasts that haven't completed yet.</summary>
    internal int StalledSends => Volatile.Read(ref _stalledSends);

    public async Task DispatchAsync(LiveNotification notification, CancellationToken cancellationToken)
    {
        if (notification.Event == LiveEvents.LogWritten)
        {
            await DispatchLogsAsync([notification.Id], cancellationToken);
            return;
        }

        object? row;
        try
        {
            row = notification.Event switch
            {
                LiveEvents.PredictionCreated => await rows.PredictionAsync(notification.Id, cancellationToken),
                LiveEvents.TradeOpened or LiveEvents.TradeClosed => await rows.TradeAsync(notification.Id, cancellationToken),
                _ => null,
            };
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "live: dropped {Event} for row {Id}: re-read failed", notification.Event, notification.Id);
            return;
        }

        if (row is null)
        {
            logger.LogDebug("live: {Event} row {Id} no longer exists (or unknown event)", notification.Event, notification.Id);
            return;
        }

        await TrySendAsync(notification.Event, row, notification.Id, cancellationToken);
        if (notification.Event == LiveEvents.TradeClosed && row is TradeDto trade)
        {
            await TrySendAsync(LiveEvents.StatsChanged, new StatsChangedDto(trade.Id, trade.ClosedAt), trade.Id, cancellationToken);
        }
    }

    /// <summary>Broadcasts the <c>logs</c> rows <paramref name="ids"/> (a burst re-read in one round trip), in id order.</summary>
    public async Task DispatchLogsAsync(IReadOnlyCollection<long> ids, CancellationToken cancellationToken)
    {
        IReadOnlyList<LogDto> logs;
        try
        {
            logs = await rows.LogsAsync(ids, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "live: dropped {Count} {Event}: re-read failed", ids.Count, LiveEvents.LogWritten);
            return;
        }

        foreach (LogDto log in logs)
        {
            await TrySendAsync(LiveEvents.LogWritten, log, log.Id, cancellationToken);
        }
    }

    private async Task TrySendAsync(string eventName, object payload, long id, CancellationToken cancellationToken)
    {
        int stalled = StalledSends;
        if (stalled >= maxStalledSends)
        {
            logger.LogWarning("live: dropped {Event} for row {Id}: {Stalled} broadcasts still stalled", eventName, id, stalled);
            return;
        }

        using CancellationTokenSource timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(_sendTimeout);
        Task send = Task.CompletedTask;
        try
        {
            send = broadcaster.BroadcastAsync(eventName, payload, timeout.Token);

            // WaitAsync bounds a send even if the transport ignores the token.
            await send.WaitAsync(_sendTimeout, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            await timeout.CancelAsync();
            if (!send.IsCompleted)
            {
                TrackStalled(send);
            }

            logger.LogWarning(ex, "live: dropped {Event} for row {Id}", eventName, id);
        }
    }

    // Counted until it ends (its outcome is observed, never rethrown).
    private void TrackStalled(Task send)
    {
        Interlocked.Increment(ref _stalledSends);
        _ = send.ContinueWith(
            t =>
            {
                _ = t.Exception;
                Interlocked.Decrement(ref _stalledSends);
            },
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }
}
