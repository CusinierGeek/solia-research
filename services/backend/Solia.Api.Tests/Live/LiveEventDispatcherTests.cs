using Microsoft.Extensions.Logging;
using Solia.Api.Endpoints;
using Solia.Api.Live;

namespace Solia.Api.Tests.Live;

public class LiveEventDispatcherTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static readonly PredictionRecordDto Prediction =
        new(1, "m1", "M1", T0, "1m", 0.8, 0.1, null, 80, null, null, null, "v1");

    private static readonly TradeDto OpenTrade =
        new(2, "m1", 1, T0, null, 0.1m, null, 2m, null, null, null, "buy", 80, null, "open");

    private static readonly TradeDto ClosedTrade =
        OpenTrade with { ClosedAt = T0.AddMinutes(1), ExitPrice = 0.2m, Pnl = 2m, PnlPct = 1, Status = "closed" };

    private static readonly LogDto Log = new(3, T0, "data-pipeline", "info", "replay done", new RawJson("{\"n\":1}"));

    private static readonly LogDto OtherLog = Log with { Id = 4, Message = "replay started" };

    private readonly FakeRows _rows = new();
    private readonly RecordingBroadcaster _broadcaster = new();
    private readonly CapturingLogger _logger = new();

    private LiveEventDispatcher Dispatcher(TimeSpan? sendTimeout = null, int maxStalledSends = LiveEventDispatcher.DefaultMaxStalledSends) =>
        new(_rows, _broadcaster, _logger, sendTimeout, maxStalledSends);

    [Fact]
    public async Task A_new_prediction_is_broadcast_as_its_history_item()
    {
        _rows.Predictions[1] = Prediction;

        await Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.PredictionCreated, 1), default);

        Assert.Equal([(LiveEvents.PredictionCreated, (object)Prediction)], _broadcaster.Sent);
    }

    [Fact]
    public async Task An_opened_trade_is_broadcast_without_a_stats_change()
    {
        _rows.Trades[2] = OpenTrade;

        await Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.TradeOpened, 2), default);

        Assert.Equal([(LiveEvents.TradeOpened, (object)OpenTrade)], _broadcaster.Sent);
    }

    [Fact]
    public async Task A_closed_trade_is_broadcast_then_followed_by_a_stats_change()
    {
        _rows.Trades[2] = ClosedTrade;

        await Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.TradeClosed, 2), default);

        Assert.Equal(
            [(LiveEvents.TradeClosed, (object)ClosedTrade), (LiveEvents.StatsChanged, new StatsChangedDto(2, T0.AddMinutes(1)))],
            _broadcaster.Sent);
    }

    [Fact]
    public async Task A_log_row_is_broadcast_as_its_logs_item()
    {
        _rows.Logs[3] = Log;

        await Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);

        Assert.Equal([(LiveEvents.LogWritten, (object)Log)], _broadcaster.Sent);
    }

    [Theory]
    [InlineData(LiveEvents.PredictionCreated)]
    [InlineData(LiveEvents.TradeOpened)]
    [InlineData(LiveEvents.TradeClosed)]
    [InlineData(LiveEvents.LogWritten)]
    public async Task A_row_that_no_longer_exists_sends_nothing(string eventName)
    {
        await Dispatcher().DispatchAsync(new LiveNotification(eventName, 99), default);

        Assert.Empty(_broadcaster.Sent);
        Assert.DoesNotContain(_logger.Levels, l => l >= LogLevel.Warning);
    }

    [Fact]
    public async Task A_failed_read_drops_the_event_with_a_warning_instead_of_throwing()
    {
        _rows.Failure = new InvalidOperationException("db down");

        await Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);

        Assert.Empty(_broadcaster.Sent);
        Assert.Contains(LogLevel.Warning, _logger.Levels);
    }

    [Fact]
    public async Task A_failed_broadcast_drops_only_that_event_and_the_next_one_still_goes_out()
    {
        _rows.Trades[2] = ClosedTrade;
        _rows.Logs[3] = Log;
        _broadcaster.FailNext = 1;
        LiveEventDispatcher dispatcher = Dispatcher();

        await dispatcher.DispatchAsync(new LiveNotification(LiveEvents.TradeClosed, 2), default);
        await dispatcher.DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);

        // The failed trade_closed doesn't take its stats_changed with it: the stats did move.
        Assert.Equal(
            [(LiveEvents.StatsChanged, (object)new StatsChangedDto(2, T0.AddMinutes(1))), (LiveEvents.LogWritten, Log)],
            _broadcaster.Sent);
        Assert.Contains(LogLevel.Warning, _logger.Levels);
    }

    [Fact]
    public async Task A_burst_of_log_rows_is_read_once_and_broadcast_in_id_order_skipping_vanished_rows()
    {
        _rows.Logs[3] = Log;
        _rows.Logs[4] = OtherLog;

        await Dispatcher().DispatchLogsAsync([4, 99, 3], default);

        Assert.Equal([(LiveEvents.LogWritten, (object)Log), (LiveEvents.LogWritten, OtherLog)], _broadcaster.Sent);
        Assert.Equal(1, _rows.LogReads);
    }

    [Fact]
    public async Task A_failed_batch_read_drops_the_logs_with_a_warning_instead_of_throwing()
    {
        _rows.Failure = new InvalidOperationException("db down");

        await Dispatcher().DispatchLogsAsync([3, 4], default);

        Assert.Empty(_broadcaster.Sent);
        Assert.Contains(LogLevel.Warning, _logger.Levels);
    }

    [Fact]
    public async Task A_send_that_honours_cancellation_is_cancelled_at_the_timeout_and_not_left_running()
    {
        _rows.Logs[3] = Log;
        _broadcaster.HangUntilCancelled = true;
        LiveEventDispatcher dispatcher = Dispatcher(TimeSpan.FromMilliseconds(50));

        await dispatcher.DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);

        Assert.True(await _broadcaster.LastSend!.ContinueWith(t => t.IsCanceled).WaitAsync(TimeSpan.FromSeconds(10)));
        await Eventually(() => dispatcher.StalledSends == 0);
    }

    [Fact]
    public async Task Stalled_sends_are_capped_then_released_when_they_end()
    {
        _rows.Logs[3] = Log;
        _broadcaster.Hang = true;
        LiveEventDispatcher dispatcher = Dispatcher(TimeSpan.FromMilliseconds(20), maxStalledSends: 2);

        for (int i = 0; i < 4; i++)
        {
            await dispatcher.DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);
        }

        // Only two sends ever started; the next ones were dropped without piling up behind them.
        Assert.Equal(2, _broadcaster.Attempts);
        Assert.Equal(2, dispatcher.StalledSends);

        _broadcaster.Hang = false;
        _broadcaster.ReleaseHung();
        await Eventually(() => dispatcher.StalledSends == 0);
        await dispatcher.DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);
        Assert.Equal([(LiveEvents.LogWritten, (object)Log)], _broadcaster.Sent);
    }

    [Fact]
    public async Task A_broadcast_stuck_on_a_slow_client_is_abandoned_after_the_send_timeout()
    {
        _rows.Logs[3] = Log;
        _broadcaster.Hang = true;

        Task dispatch = Dispatcher(TimeSpan.FromMilliseconds(50))
            .DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), default);

        Assert.Same(dispatch, await Task.WhenAny(dispatch, Task.Delay(TimeSpan.FromSeconds(10))));
        await dispatch;
        Assert.Contains(LogLevel.Warning, _logger.Levels);
    }

    [Fact]
    public async Task Shutdown_cancellation_propagates()
    {
        _rows.Logs[3] = Log;
        _broadcaster.Hang = true;
        using var stopping = new CancellationTokenSource();

        Task dispatch = Dispatcher().DispatchAsync(new LiveNotification(LiveEvents.LogWritten, 3), stopping.Token);
        await stopping.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => dispatch);
    }

    private static async Task Eventually(Func<bool> condition)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(10);
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, "condition never became true");
            await Task.Delay(10);
        }
    }

    private sealed class FakeRows : ILiveRowReader
    {
        public Dictionary<long, PredictionRecordDto> Predictions { get; } = [];

        public Dictionary<long, TradeDto> Trades { get; } = [];

        public Dictionary<long, LogDto> Logs { get; } = [];

        public Exception? Failure { get; set; }

        public int LogReads { get; private set; }

        public Task<PredictionRecordDto?> PredictionAsync(long id, CancellationToken cancellationToken) => Find(Predictions, id);

        public Task<TradeDto?> TradeAsync(long id, CancellationToken cancellationToken) => Find(Trades, id);

        public Task<IReadOnlyList<LogDto>> LogsAsync(IReadOnlyCollection<long> ids, CancellationToken cancellationToken)
        {
            LogReads++;
            return Failure is { } failure
                ? Task.FromException<IReadOnlyList<LogDto>>(failure)
                : Task.FromResult<IReadOnlyList<LogDto>>([.. ids.Order().Where(Logs.ContainsKey).Select(id => Logs[id])]);
        }

        private Task<T?> Find<T>(Dictionary<long, T> rows, long id) where T : class =>
            Failure is { } failure ? Task.FromException<T?>(failure) : Task.FromResult(rows.GetValueOrDefault(id));
    }

    private sealed class RecordingBroadcaster : ILiveBroadcaster
    {
        public List<(string Event, object Payload)> Sent { get; } = [];

        public int FailNext { get; set; }

        private readonly TaskCompletionSource _release = new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>Sends never complete and ignore their token, until <see cref="ReleaseHung"/>.</summary>
        public bool Hang { get; set; }

        /// <summary>Sends never complete on their own but honour their token.</summary>
        public bool HangUntilCancelled { get; set; }

        public int Attempts { get; private set; }

        public Task? LastSend { get; private set; }

        public void ReleaseHung() => _release.TrySetResult();

        public Task BroadcastAsync(string eventName, object payload, CancellationToken cancellationToken)
        {
            Attempts++;
            if (Hang)
            {
                return LastSend = _release.Task;
            }

            if (HangUntilCancelled)
            {
                return LastSend = Task.Delay(Timeout.Infinite, cancellationToken);
            }

            if (FailNext > 0)
            {
                FailNext--;
                return Task.FromException(new IOException("client gone"));
            }

            Sent.Add((eventName, payload));
            return Task.CompletedTask;
        }
    }

    private sealed class CapturingLogger : ILogger<LiveEventDispatcher>
    {
        public List<LogLevel> Levels { get; } = [];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(
            LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            Levels.Add(logLevel);
    }
}
