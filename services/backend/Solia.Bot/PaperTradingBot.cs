using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;
using Solia.Bot.Trading;
using Solia.Shared.Logging;

namespace Solia.Bot;

/// <summary>
/// The paper-trading bot as the long-lived hosted service it is in production (docs/ARCHITECTURE.md §3):
/// it drives the <see cref="PositionManager"/> engine over an injected <see cref="IBotEventSource"/>,
/// one event at a time in logical-time order. Mirrors the data-pipeline <c>Worker</c>: a clean stop on
/// cancellation — including a store/source error surfaced <em>by</em> the cancellation — and
/// <strong>fail-loud</strong> on any other error: an engine/store failure stops the host rather than
/// leaving it up but silently trading nothing.
///
/// <para><b>Idle timeout sweep.</b> The engine's clock is event time, so on a quiet feed nothing would
/// ever expire. Every <see cref="BotOptions.TimeoutSweepSeconds"/> the host checks how long it has been
/// since the last event; once that is at least the interval, it advances the engine's clock by exactly
/// that elapsed wall time (<see cref="PositionManager.AdvanceToAsync"/>) — extrapolating the feed's own
/// timescale rather than jumping to the wall clock, so a feed whose timestamps lag the wall clock isn't
/// then dropped as late. The sweep and the event loop take turns on the (not thread-safe) engine.
/// Disabled when the option is 0 (deterministic replay).</para>
///
/// <para><b>Sessions.</b> With <see cref="BotSessions"/> (the hosted bot), the run first continues the current
/// bot session or opens a new one (issue #169), so its trades are read per regime; a failure there is a bot
/// failure like any other.</para>
///
/// <para>Operational milestones (start/stop) are written to the `logs` table through a
/// <see cref="BestEffortLogWriter"/> — bounded and never thrown, so a logging outage neither aborts
/// trading nor overruns shutdown. Per-trade opens/closes are logged by the engine itself (it owns them).
/// <strong>Paper only</strong>: nothing here reaches a wallet or the chain.</para>
/// </summary>
public sealed class PaperTradingBot : BackgroundService
{
    private readonly ILogger<PaperTradingBot> _logger;
    private readonly BestEffortLogWriter _log;
    private readonly PositionManager _engine;
    private readonly IBotEventSource _source;
    private readonly BotOptions _options;
    private readonly TimeProvider _time;
    private readonly BotSessions? _sessions;

    // Serializes the event loop and the idle sweep on the engine.
    private readonly SemaphoreSlim _engineGate = new(1, 1);

    // Wall timestamp (TimeProvider) of the engine's last clock move — event or sweep.
    private long _lastAdvance;

    public PaperTradingBot(
        ILogger<PaperTradingBot> logger,
        IBotLogSink logSink,
        PositionManager engine,
        IBotEventSource source,
        IOptions<BotOptions> options,
        TimeProvider? timeProvider = null,
        BotSessions? sessions = null)
    {
        _logger = logger;
        _log = new BestEffortLogWriter(logSink, logger);
        _engine = engine;
        _source = source;
        _options = options.Value;
        _time = timeProvider ?? TimeProvider.System;
        _sessions = sessions;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation(
            "paper-trading bot started (startingBalance={Balance}, maxPositions={MaxPositions}, timeoutSweep={Sweep}s)",
            _options.StartingBalance, _options.MaxConcurrentPositions, _options.TimeoutSweepSeconds);
        await _log.WriteAsync(LogSeverity.Info, "bot started",
            new { _options.StartingBalance, _options.MaxConcurrentPositions, _options.TimeoutSweepSeconds }, stoppingToken);

        try
        {
            // Before any event: every trade the bot opens belongs to a session (no session, no trading).
            if (_sessions is not null)
            {
                await _sessions.StartAsync(stoppingToken);
            }

            await RunAsync(stoppingToken);
            if (!stoppingToken.IsCancellationRequested)
            {
                _logger.LogWarning("paper-trading bot event source completed; no more events will be traded");
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Normal shutdown.
        }
        catch (Exception ex) when (stoppingToken.IsCancellationRequested)
        {
            // Cancelling an in-flight store command can surface as a non-OCE error (e.g. a broken
            // Npgsql connection) — that's the shutdown, not a crash.
            _logger.LogInformation(ex, "paper-trading bot: error while stopping, treated as shutdown");
        }
        catch (Exception ex)
        {
            // Fail loudly: rethrowing stops the host instead of leaving a bot that trades nothing.
            _logger.LogCritical(ex, "paper-trading bot failed");
            await _log.WriteAsync(LogSeverity.Error, "bot failed", new { error = ex.Message }, CancellationToken.None);
            throw;
        }

        _logger.LogInformation(
            "paper-trading bot stopping (closedTrades={Closed}, realizedPnl={Pnl}, equity={Equity}, failedEntries={FailedEntries}, " +
            "failedExits={FailedExits}, abandonedEntries={AbandonedEntries})",
            _engine.ClosedTradeCount, _engine.RealizedPnl, _engine.Equity, _engine.FailedEntryCount, _engine.FailedExitCount,
            _engine.AbandonedEntryCount);
        // Not tied to the (already cancelled) stopping token, but bounded by the writer's timeout.
        await _log.WriteAsync(LogSeverity.Info, "bot stopping",
            new
            {
                _engine.ClosedTradeCount,
                _engine.RealizedPnl,
                _engine.Equity,
                _engine.FailedEntryCount,
                _engine.FailedExitCount,
                _engine.AbandonedEntryCount,
            },
            CancellationToken.None);
    }

    /// <summary>
    /// Runs the event loop and the idle sweep until the source completes, either one fails, or the host
    /// stops; the first to end stops the other, and the first failure is the one rethrown.
    /// </summary>
    private async Task RunAsync(CancellationToken stoppingToken)
    {
        using var loops = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken);
        TimeSpan interval = TimeSpan.FromSeconds(_options.TimeoutSweepSeconds);
        // Armed before the event loop starts, so idle time is measured from the very first event.
        using PeriodicTimer? timer = interval > TimeSpan.Zero ? new PeriodicTimer(interval, _time) : null;
        _lastAdvance = _time.GetTimestamp();

        Task events = ConsumeEventsAsync(loops.Token);
        Task sweeps = timer is not null
            ? SweepTimeoutsAsync(timer, interval, loops.Token)
            : Task.Delay(Timeout.Infinite, loops.Token);

        Task first = await Task.WhenAny(events, sweeps);
        await loops.CancelAsync();
        try
        {
            await (first == events ? sweeps : events);
        }
        catch (Exception) when (loops.IsCancellationRequested)
        {
            // The loop we just stopped; the outcome that matters is the first one's.
        }

        await first;
    }

    private async Task ConsumeEventsAsync(CancellationToken cancellationToken)
    {
        await foreach (BotEvent botEvent in _source.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            await _engineGate.WaitAsync(cancellationToken);
            try
            {
                await _engine.HandleAsync(botEvent, cancellationToken);
                _lastAdvance = _time.GetTimestamp();
            }
            finally
            {
                _engineGate.Release();
            }
        }
    }

    private async Task SweepTimeoutsAsync(PeriodicTimer timer, TimeSpan interval, CancellationToken cancellationToken)
    {
        while (await timer.WaitForNextTickAsync(cancellationToken))
        {
            await _engineGate.WaitAsync(cancellationToken);
            try
            {
                TimeSpan idle = _time.GetElapsedTime(_lastAdvance);
                // Before the first event there is no clock to extrapolate — and no position to expire.
                if (idle >= interval && _engine.Clock is DateTimeOffset clock)
                {
                    await _engine.AdvanceToAsync(clock + idle, cancellationToken);
                    _lastAdvance = _time.GetTimestamp();
                }
            }
            finally
            {
                _engineGate.Release();
            }
        }
    }

    public override void Dispose()
    {
        _engineGate.Dispose();
        base.Dispose();
    }
}
