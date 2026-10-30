namespace Solia.Bot.Sessions;

/// <summary>What the bot asks of the session store: its regime, and whether this is a fresh bot run.</summary>
/// <param name="OptionsJson">The normalized effective options (<see cref="BotSessionFingerprint.OptionsJson"/>).</param>
/// <param name="Label">Names the session if a new one starts; ignored otherwise.</param>
/// <param name="NewRun">
/// The bot process just started, with an empty book: every trade still open is an orphan of the earlier run.
/// False for a re-check while the bot runs (its open positions are still held).
/// </param>
public sealed record BotSessionRequest(string OptionsJson, string? Label, bool NewRun);

/// <summary>The session the bot now trades in.</summary>
public sealed record BotSession(long Id, BotSessionDecision Decision, long? ModelId, string? ModelVersion, string ConfigHash);

/// <summary>
/// Where bot sessions are kept (table <c>bot_sessions</c>, docs/DATA-MODEL.md). <see cref="BeginAsync"/> reads the
/// champion, applies <see cref="BotSessionRules.Decide"/> to the current session and records the outcome atomically.
/// </summary>
public interface IBotSessionStore
{
    Task<BotSession> BeginAsync(BotSessionRequest request, CancellationToken cancellationToken = default);
}
