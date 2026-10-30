using System.Runtime.CompilerServices;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Time.Testing;
using Solia.Bot;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;
using Solia.Shared.Logging;

namespace Solia.Bot.Tests;

/// <summary>
/// The hosted service driving the engine over an <see cref="IBotEventSource"/>: an in-memory run
/// produces the expected trades, shutdown is graceful, an engine error fails loud, and operational
/// milestones reach the sink. Engine P&amp;L math itself lives in <see cref="Trading.PositionManagerTests"/>.
/// </summary>
public class PaperTradingBotTests
{
    private const string Mint = "mint1";
    private const decimal Entry = 0.0001m;
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static BotEvent Tick(int sec, decimal price) =>
        new BotEvent.Price(new PriceTick { Mint = Mint, Ts = T0.AddSeconds(sec), Price = price });

    private static BotEvent Predict(int sec) =>
        new BotEvent.Prediction(PolicyFixtures.Buyable() with { Mint = Mint, Ts = T0.AddSeconds(sec) });

    private static PaperTradingBot Build(
        IBotEventSource source, ITradeStore store, IBotLogSink sink, out PositionManager engine,
        BotOptions? options = null, TimeProvider? time = null)
    {
        options ??= new BotOptions { RealisticFills = false };
        engine = new PositionManager(options, store, sink);
        return new PaperTradingBot(
            NullLogger<PaperTradingBot>.Instance, sink, engine, source, Options.Create(options), time);
    }

    /// <summary>Polls until <paramref name="condition"/> holds — the bot runs on its own task.</summary>
    private static async Task Eventually(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        while (!condition())
        {
            await Task.Delay(10, timeout.Token);
        }
    }

    [Fact]
    public async Task A_run_over_an_in_memory_source_produces_the_expected_trades()
    {
        var store = new InMemoryTradeStore();
        var sink = new RecordingLogSink();
        // Mark, open at the market, then a tick at the +50% take-profit closes the position.
        var source = new InMemoryBotEventSource([Tick(0, Entry), Predict(1), Tick(2, 0.00015m)]);
        PaperTradingBot bot = Build(source, store, sink, out PositionManager engine);

        await bot.StartAsync(default);
        await bot.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));

        ClosedTrade trade = Assert.Single(store.Closed);
        Assert.Equal(ExitReason.TakeProfit, trade.Reason);
        Assert.Equal(1m, trade.Pnl); // size 2 · (0.00015/0.0001) − 2
        Assert.Equal(1, engine.ClosedTradeCount);
        Assert.Empty(engine.OpenPositions);
    }

    [Fact]
    public async Task Start_open_close_and_stop_are_logged_for_the_dashboard()
    {
        var store = new InMemoryTradeStore();
        var sink = new RecordingLogSink();
        var source = new InMemoryBotEventSource([Tick(0, Entry), Predict(1), Tick(2, 0.00015m)]);
        PaperTradingBot bot = Build(source, store, sink, out _);

        await bot.StartAsync(default);
        await bot.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(
            ["bot started", "position opened", "position closed", "bot stopping"],
            sink.Messages);
    }

    [Fact]
    public async Task A_logging_outage_never_stops_trading()
    {
        var store = new InMemoryTradeStore();
        var source = new InMemoryBotEventSource([Tick(0, Entry), Predict(1), Tick(2, 0.00015m)]);
        PaperTradingBot bot = Build(source, store, new ThrowingLogSink(), out _);

        await bot.StartAsync(default);
        await bot.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Single(store.Closed); // the trade still opened and closed despite every log write throwing
    }

    [Fact]
    public async Task Shutdown_cancels_gracefully_without_throwing()
    {
        var sink = new RecordingLogSink();
        var source = new BlockingSource();
        PaperTradingBot bot = Build(source, new InMemoryTradeStore(), sink, out _);

        await bot.StartAsync(default);
        await source.Entered; // park in the feed before stopping, so the test doesn't race the token
        await bot.StopAsync(default); // signals the token; ExecuteAsync must swallow the cancellation
        await bot.ExecuteTask!; // rethrows if the shutdown wasn't graceful

        Assert.True(bot.ExecuteTask!.IsCompletedSuccessfully);
        Assert.Contains("bot stopping", sink.Messages);
    }

    [Fact]
    public async Task An_unexpected_source_error_fails_loud()
    {
        var sink = new RecordingLogSink();
        PaperTradingBot bot = Build(new ThrowingSource(), new InMemoryTradeStore(), sink, out _);

        await bot.StartAsync(default);

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.ExecuteTask!);
        Assert.Contains("bot failed", sink.Messages);
    }

    [Fact]
    public async Task The_bot_session_is_begun_before_the_first_event()
    {
        var store = new InMemoryTradeStore();
        var sink = new RecordingLogSink();
        var sessionStore = new InMemoryBotSessionStore { ChampionId = 1, ChampionVersion = "v1" };
        var options = new BotOptions { RealisticFills = false };
        var sessions = new BotSessions(sessionStore, Options.Create(options), [], sink);
        var source = new InMemoryBotEventSource([Tick(0, Entry), Predict(1), Tick(2, 0.00015m)]);
        var engine = new PositionManager(options, store, sink);
        var bot = new PaperTradingBot(NullLogger<PaperTradingBot>.Instance, sink, engine, source, Options.Create(options), null, sessions);

        await bot.StartAsync(default);
        await bot.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(1, Assert.Single(sessionStore.Sessions).Session.Id);
        Assert.Equal(1, sessions.Current!.Id);
        List<string> messages = [.. sink.Messages];
        Assert.InRange(messages.IndexOf("bot session started"), 0, messages.IndexOf("position opened") - 1);
    }

    [Fact]
    public async Task A_session_store_failure_fails_loud_before_any_trade()
    {
        var store = new InMemoryTradeStore();
        var sink = new RecordingLogSink();
        var options = new BotOptions { RealisticFills = false };
        var sessions = new BotSessions(new ThrowingSessionStore(), Options.Create(options), [], sink);
        var source = new InMemoryBotEventSource([Tick(0, Entry), Predict(1)]);
        var bot = new PaperTradingBot(NullLogger<PaperTradingBot>.Instance, sink, new PositionManager(options, store, sink),
            source, Options.Create(options), null, sessions);

        await bot.StartAsync(default);

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.ExecuteTask!);
        Assert.Empty(store.Open);
        Assert.Contains("bot failed", sink.Messages);
    }

    [Fact]
    public async Task On_a_quiet_feed_the_idle_sweep_closes_a_position_at_its_timeout()
    {
        var time = new FakeTimeProvider(T0);
        var store = new InMemoryTradeStore();
        // Open at t = 1 (deadline t = 901), then the feed goes silent.
        var source = new ThenBlockingSource([Tick(0, Entry), Predict(1)]);
        PaperTradingBot bot = Build(source, store, NullBotLogSink.Instance, out PositionManager engine,
            new BotOptions { TimeoutSweepSeconds = 5, RealisticFills = false }, time);

        await bot.StartAsync(default);
        await source.Entered; // both events applied; the loop now waits on the feed
        Assert.Single(engine.OpenPositions);

        time.Advance(TimeSpan.FromSeconds(900));
        await Eventually(() => store.Closed.Count == 1);

        ClosedTrade trade = store.Closed[0];
        Assert.Equal(ExitReason.Timeout, trade.Reason);
        Assert.Equal(T0.AddSeconds(901), trade.ClosedAt); // at its deadline, not the sweep's time
        Assert.Empty(engine.OpenPositions);
        await bot.StopAsync(default);
    }

    [Fact]
    public async Task The_sweep_extrapolates_event_time_by_idle_wall_time_rather_than_jumping_to_the_wall_clock()
    {
        // Wall clock a day ahead of the feed's timestamps: a jump to wall time would expire everything.
        var time = new FakeTimeProvider(T0.AddDays(1));
        var source = new ThenBlockingSource([Tick(0, Entry), Predict(1)]);
        PaperTradingBot bot = Build(source, new InMemoryTradeStore(), NullBotLogSink.Instance, out PositionManager engine,
            new BotOptions { TimeoutSweepSeconds = 5, RealisticFills = false }, time);

        await bot.StartAsync(default);
        await source.Entered;
        time.Advance(TimeSpan.FromSeconds(10));
        await Eventually(() => engine.Clock > T0.AddSeconds(1));

        Assert.Equal(T0.AddSeconds(11), engine.Clock);
        Assert.Single(engine.OpenPositions); // 10 s idle is far from the 900 s timeout
        await bot.StopAsync(default);
    }

    [Fact]
    public async Task With_the_sweep_disabled_a_quiet_feed_never_moves_the_clock()
    {
        var time = new FakeTimeProvider(T0);
        var store = new InMemoryTradeStore();
        var source = new ThenBlockingSource([Tick(0, Entry), Predict(1)]);
        PaperTradingBot bot = Build(source, store, NullBotLogSink.Instance, out PositionManager engine,
            new BotOptions { TimeoutSweepSeconds = 0, RealisticFills = false }, time);

        await bot.StartAsync(default);
        await source.Entered;
        time.Advance(TimeSpan.FromHours(1));
        await Task.Delay(50);

        Assert.Empty(store.Closed);
        Assert.Equal(T0.AddSeconds(1), engine.Clock);
        await bot.StopAsync(default);
        Assert.True(bot.ExecuteTask!.IsCompletedSuccessfully);
    }

    [Fact]
    public async Task A_store_failure_during_the_idle_sweep_fails_loud()
    {
        var time = new FakeTimeProvider(T0);
        var sink = new RecordingLogSink();
        var source = new ThenBlockingSource([Tick(0, Entry), Predict(1)]);
        PaperTradingBot bot = Build(source, new CloseFailingStore(), sink, out _,
            new BotOptions { TimeoutSweepSeconds = 5, RealisticFills = false }, time);

        await bot.StartAsync(default);
        await source.Entered;
        time.Advance(TimeSpan.FromSeconds(900));

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Contains("bot failed", sink.Messages);
    }

    [Fact]
    public async Task A_non_cancellation_error_raised_by_shutdown_is_a_clean_stop_not_a_crash()
    {
        var sink = new RecordingLogSink();
        var source = new ThrowsOnCancelSource();
        PaperTradingBot bot = Build(source, new InMemoryTradeStore(), sink, out _);

        await bot.StartAsync(default);
        await source.Entered;
        await bot.StopAsync(default);
        await bot.ExecuteTask!;

        Assert.True(bot.ExecuteTask!.IsCompletedSuccessfully);
        Assert.DoesNotContain("bot failed", sink.Messages);
        Assert.Contains("bot stopping", sink.Messages);
    }

    /// <summary>Yields <paramref name="events"/>, then waits for cancellation like a live feed gone quiet.</summary>
    private sealed class ThenBlockingSource(IEnumerable<BotEvent> events) : IBotEventSource
    {
        private readonly TaskCompletionSource _entered = new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>Completes once every event has been handed out and handled.</summary>
        public Task Entered => _entered.Task;

        public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (BotEvent e in events)
            {
                yield return e;
            }

            _entered.SetResult();
            await Task.Delay(Timeout.Infinite, cancellationToken);
        }
    }

    /// <summary>On cancellation, faults with a non-OCE error — as a cancelled Npgsql read can.</summary>
    private sealed class ThrowsOnCancelSource : IBotEventSource
    {
        private readonly TaskCompletionSource _entered = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task Entered => _entered.Task;

        public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            _entered.SetResult();
            try
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                throw new IOException("connection broken by cancel");
            }

            yield break;
        }
    }

    private sealed class CloseFailingStore : ITradeStore
    {
        private readonly InMemoryTradeStore _inner = new();

        public Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default) =>
            _inner.OpenAsync(opening, cancellationToken);

        public Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("db down");
    }

    private sealed class RecordingLogSink : IBotLogSink
    {
        private readonly List<string> _messages = [];

        public IReadOnlyList<string> Messages => _messages;

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            _messages.Add(message);
            return Task.CompletedTask;
        }
    }

    private sealed class ThrowingSessionStore : IBotSessionStore
    {
        public Task<BotSession> BeginAsync(BotSessionRequest request, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("database down");
    }

    private sealed class ThrowingLogSink : IBotLogSink
    {
        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
            throw new InvalidOperationException("sink is down");
    }

    /// <summary>Never yields; waits for cancellation — stands in for a quiet live feed.</summary>
    private sealed class BlockingSource : IBotEventSource
    {
        private readonly TaskCompletionSource _entered = new(TaskCreationOptions.RunContinuationsAsynchronously);

        /// <summary>Completes once <see cref="ReadAsync"/> is parked waiting for cancellation.</summary>
        public Task Entered => _entered.Task;

        public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            _entered.SetResult();
            await Task.Delay(Timeout.Infinite, cancellationToken);
            yield break;
        }
    }

    /// <summary>Faults on the first read with a non-cancellation error.</summary>
    private sealed class ThrowingSource : IBotEventSource
    {
        public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await Task.Yield();
            throw new InvalidOperationException("feed broke");
#pragma warning disable CS0162 // unreachable: required to make this an async iterator
            yield break;
#pragma warning restore CS0162
        }
    }
}
