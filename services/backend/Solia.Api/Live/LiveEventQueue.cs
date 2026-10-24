using System.Threading.Channels;
using Solia.Api.Logging;

namespace Solia.Api.Live;

/// <summary>
/// The buffer between the LISTEN connection and the <see cref="LiveEventDispatcher"/>. Writing never blocks: two
/// bounded queues each drop their oldest entry when full — one for prediction/trade events, one for log ids — so a
/// flood of <c>logs</c> rows can't push trade events out. The pump always serves prediction/trade events first and
/// re-reads queued logs in batches (one round trip per <see cref="LogBatchSize"/> rows).
/// </summary>
public sealed class LiveEventQueue(int capacity = 1024)
{
    /// <summary>The most log rows re-read in one query.</summary>
    public const int LogBatchSize = 100;

    private readonly Channel<LiveNotification> _rows = Channel.CreateBounded<LiveNotification>(
        new BoundedChannelOptions(capacity) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = true });

    private readonly Channel<long> _logs = Channel.CreateBounded<long>(
        new BoundedChannelOptions(capacity) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true, SingleWriter = true });

    // One pending wake-up is enough: the pump drains both queues each time it wakes.
    private readonly Channel<bool> _wake = Channel.CreateBounded<bool>(
        new BoundedChannelOptions(1) { FullMode = BoundedChannelFullMode.DropWrite, SingleReader = true });

    /// <summary>Queues one notification; returns at once (a full queue drops its oldest entry).</summary>
    public void Enqueue(LiveNotification notification)
    {
        if (notification.Event == LiveEvents.LogWritten)
        {
            _logs.Writer.TryWrite(notification.Id);
        }
        else
        {
            _rows.Writer.TryWrite(notification);
        }

        _wake.Writer.TryWrite(true);
    }

    /// <summary>Stops the pump once it has drained what is queued.</summary>
    public void Complete() => _wake.Writer.TryComplete();

    /// <summary>
    /// Dispatches until <see cref="Complete"/> or shutdown. Nothing logged while dispatching reaches the `logs`
    /// table (EF, Npgsql and SignalR errors included): each row there is itself a live event, so a failing re-read
    /// would otherwise feed itself.
    /// </summary>
    public async Task PumpAsync(LiveEventDispatcher dispatcher, CancellationToken stoppingToken)
    {
        using IDisposable quiet = PostgresLoggerProvider.Suppress();
        List<long> batch = new(LogBatchSize);
        await foreach (bool _ in _wake.Reader.ReadAllAsync(stoppingToken))
        {
            while (true)
            {
                if (_rows.Reader.TryRead(out LiveNotification notification))
                {
                    await dispatcher.DispatchAsync(notification, stoppingToken);
                    continue;
                }

                while (batch.Count < LogBatchSize && _logs.Reader.TryRead(out long id))
                {
                    batch.Add(id);
                }

                if (batch.Count == 0)
                {
                    break;
                }

                await dispatcher.DispatchLogsAsync(batch, stoppingToken);
                batch.Clear();
            }
        }
    }
}
