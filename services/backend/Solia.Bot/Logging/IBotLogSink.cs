using Solia.Shared.Logging;

namespace Solia.Bot.Logging;

/// <summary>
/// The bot's operational sink: the shared <see cref="ILogSink"/> contract for the `logs` table, with
/// service = 'bot' (surfaced on the dashboard). A distinct type so the bot's sink can be registered
/// alongside other services' sinks in the backend host. For notable operational events — bot start/stop,
/// position opens/closes — not a replacement for standard <c>ILogger</c> output. Injected into the engine
/// and host with a no-op default (<see cref="NullBotLogSink"/>) so neither needs a database in tests.
/// Callers write through <see cref="BestEffortLogWriter"/>, never directly.
/// </summary>
public interface IBotLogSink : ILogSink;
