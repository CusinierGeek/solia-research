using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Solia.Bot.Configuration;
using Solia.Bot.Logging;
using Solia.Shared.Logging;

namespace Solia.Bot.Sessions;

/// <summary>
/// The bot session trades are recorded in (issue #169). <see cref="StartAsync"/> runs once when the bot starts: it
/// continues the current session if the regime (effective options + champion) is unchanged, or opens a new one.
/// The champion can also change while the bot runs (a promotion), so <see cref="CurrentIdAsync"/> — asked before
/// each trade is opened — re-checks the regime every <see cref="RefreshInterval"/> and rolls over to a new session
/// when it moved. A trade keeps the session it was opened in, even when it closes in the next one.
/// Not thread-safe: the bot's engine (its only user) runs one event at a time.
/// </summary>
public sealed class BotSessions
{
    public static readonly TimeSpan RefreshInterval = TimeSpan.FromSeconds(30);

    private readonly IBotSessionStore _store;
    private readonly string _optionsJson;
    private readonly string? _label;
    private readonly TimeProvider _time;
    private readonly ILogger _logger;
    private readonly BestEffortLogWriter _log;
    private BotSession? _current;
    private long _checkedAt;

    public BotSessions(IBotSessionStore store, IOptions<BotOptions> options, IEnumerable<BotRegimePart> parts,
        IBotLogSink logSink, TimeProvider? timeProvider = null, ILogger<BotSessions>? logger = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        _store = store;
        _optionsJson = BotSessionFingerprint.OptionsJson(options.Value, parts);
        _label = options.Value.SessionLabel;
        _time = timeProvider ?? TimeProvider.System;
        _logger = logger ?? (ILogger)NullLogger.Instance;
        _log = new BestEffortLogWriter(logSink, _logger);
    }

    /// <summary>The session the bot trades in; null before <see cref="StartAsync"/>.</summary>
    public BotSession? Current => _current;

    /// <summary>Begins (or continues) the session at bot start: every trade still open is now an orphan.</summary>
    public Task<BotSession> StartAsync(CancellationToken cancellationToken = default) =>
        BeginAsync(newRun: true, cancellationToken);

    /// <summary>The current session's id, re-checking the regime once <see cref="RefreshInterval"/> has passed.</summary>
    /// <exception cref="InvalidOperationException"><see cref="StartAsync"/> was never called.</exception>
    public async ValueTask<long> CurrentIdAsync(CancellationToken cancellationToken = default)
    {
        if (_current is null)
        {
            throw new InvalidOperationException("The bot session was never started: no trade can be recorded outside one.");
        }

        if (_time.GetElapsedTime(_checkedAt) >= RefreshInterval)
        {
            await BeginAsync(newRun: false, cancellationToken);
        }

        return _current.Id;
    }

    private async Task<BotSession> BeginAsync(bool newRun, CancellationToken cancellationToken)
    {
        BotSession? previous = _current;
        BotSession session = await _store.BeginAsync(new BotSessionRequest(_optionsJson, _label, newRun), cancellationToken);
        _current = session;
        _checkedAt = _time.GetTimestamp();

        if (session.Decision != BotSessionDecision.Continue || previous is null)
        {
            _logger.LogInformation("bot session #{Session} {Decision} (champion {Model}, config {Hash})",
                session.Id, Describe(session.Decision), session.ModelVersion ?? "none", session.ConfigHash[..12]);
            await _log.WriteAsync(LogSeverity.Info, $"bot session {Describe(session.Decision)}",
                new { session = session.Id, previous = previous?.Id, model = session.ModelVersion, config = session.ConfigHash },
                cancellationToken);
        }

        return session;
    }

    private static string Describe(BotSessionDecision decision) => decision switch
    {
        BotSessionDecision.Continue => "continued",
        BotSessionDecision.Adopt => "adopted",
        _ => "started",
    };
}
