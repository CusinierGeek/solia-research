using Microsoft.Extensions.Logging.Abstractions;
using Solia.Api.Endpoints;
using Solia.Api.Live;
using Solia.Api.Logging;

namespace Solia.Api.Tests.Live;

public class LiveEventQueueTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static readonly TradeDto Trade =
        new(2, "m1", 1, T0, null, 0.1m, null, 2m, null, null, null, "buy", 80, null, "open");

    private readonly Rows _rows = new();
    private readonly Broadcaster _broadcaster = new();

    [Fact]
    public async Task Prediction_and_trade_events_go_out_before_queued_logs_which_are_read_in_one_batch()
    {
        var queue = new LiveEventQueue();
        queue.Enqueue(new LiveNotification(LiveEvents.LogWritten, 3));
        queue.Enqueue(new LiveNotification(LiveEvents.LogWritten, 4));
        queue.Enqueue(new LiveNotification(LiveEvents.TradeOpened, 2));

        await PumpToEndAsync(queue);

        Assert.Equal([LiveEvents.TradeOpened, LiveEvents.LogWritten, LiveEvents.LogWritten], _broadcaster.Events);
        Assert.Equal([[3L, 4L]], _rows.LogBatches);
    }

    [Fact]
    public async Task A_log_flood_drops_old_logs_but_never_pushes_a_trade_event_out()
    {
        var queue = new LiveEventQueue(capacity: 2);
        queue.Enqueue(new LiveNotification(LiveEvents.TradeOpened, 2));
        for (long id = 10; id < 20; id++)
        {
            queue.Enqueue(new LiveNotification(LiveEvents.LogWritten, id));
        }

        await PumpToEndAsync(queue);

        Assert.Equal(LiveEvents.TradeOpened, _broadcaster.Events[0]);
        Assert.Equal([[18L, 19L]], _rows.LogBatches);
    }

    [Fact]
    public async Task Logs_are_read_in_batches_of_at_most_the_batch_size()
    {
        var queue = new LiveEventQueue();
        for (long id = 1; id <= LiveEventQueue.LogBatchSize + 50; id++)
        {
            queue.Enqueue(new LiveNotification(LiveEvents.LogWritten, id));
        }

        await PumpToEndAsync(queue);

        Assert.Equal([LiveEventQueue.LogBatchSize, 50], _rows.LogBatches.Select(b => b.Length));
    }

    [Fact]
    public async Task Nothing_logged_while_dispatching_is_mirrored_into_the_logs_table()
    {
        var queue = new LiveEventQueue();
        queue.Enqueue(new LiveNotification(LiveEvents.LogWritten, 3));

        await PumpToEndAsync(queue);

        Assert.True(_rows.SawSuppressedLogging);
        Assert.False(PostgresLoggerProvider.IsSuppressed);
    }

    [Fact]
    public async Task Events_queued_while_the_pump_waits_wake_it_up()
    {
        var queue = new LiveEventQueue();
        Task pump = queue.PumpAsync(Dispatcher(), CancellationToken.None);

        queue.Enqueue(new LiveNotification(LiveEvents.TradeOpened, 2));
        DateTime deadline = DateTime.UtcNow.AddSeconds(10);
        while (_broadcaster.Events.Count == 0)
        {
            Assert.True(DateTime.UtcNow < deadline, "the pump never woke up");
            await Task.Delay(10);
        }

        queue.Complete();
        await pump.WaitAsync(TimeSpan.FromSeconds(10));
    }

    private LiveEventDispatcher Dispatcher() => new(_rows, _broadcaster, NullLogger<LiveEventDispatcher>.Instance);

    private async Task PumpToEndAsync(LiveEventQueue queue)
    {
        queue.Complete();
        await queue.PumpAsync(Dispatcher(), CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(10));
    }

    private sealed class Rows : ILiveRowReader
    {
        public List<long[]> LogBatches { get; } = [];

        public bool SawSuppressedLogging { get; private set; }

        public Task<PredictionRecordDto?> PredictionAsync(long id, CancellationToken cancellationToken) =>
            Task.FromResult<PredictionRecordDto?>(null);

        public Task<TradeDto?> TradeAsync(long id, CancellationToken cancellationToken) =>
            Task.FromResult<TradeDto?>(Trade with { Id = id });

        public Task<IReadOnlyList<LogDto>> LogsAsync(IReadOnlyCollection<long> ids, CancellationToken cancellationToken)
        {
            SawSuppressedLogging = PostgresLoggerProvider.IsSuppressed;
            LogBatches.Add([.. ids]);
            return Task.FromResult<IReadOnlyList<LogDto>>(
                [.. ids.Select(id => new LogDto(id, T0, "ai-service", "info", "m", null))]);
        }
    }

    private sealed class Broadcaster : ILiveBroadcaster
    {
        public List<string> Events { get; } = [];

        public Task BroadcastAsync(string eventName, object payload, CancellationToken cancellationToken)
        {
            lock (Events)
            {
                Events.Add(eventName);
            }

            return Task.CompletedTask;
        }
    }
}
