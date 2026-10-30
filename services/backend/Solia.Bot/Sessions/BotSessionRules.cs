namespace Solia.Bot.Sessions;

/// <summary>What a bot start (or a champion check) does with the current session.</summary>
public enum BotSessionDecision
{
    /// <summary>Same regime: the current session goes on.</summary>
    Continue,

    /// <summary>
    /// The current session was backfilled without a recorded configuration and its champion is still the one
    /// serving: it is taken over and the configuration recorded.
    /// </summary>
    Adopt,

    /// <summary>No current session, or a different regime: the current one ends and a new one starts.</summary>
    New,
}

/// <summary>The current session as the rule reads it.</summary>
/// <param name="ConfigHash">Null for a session backfilled without a known configuration.</param>
public sealed record CurrentBotSession(long Id, string? ConfigHash, long? ModelId);

/// <summary>When a bot session continues and when a new one starts (issue #169).</summary>
public static class BotSessionRules
{
    /// <summary>
    /// <see cref="BotSessionDecision.Continue"/> when <paramref name="current"/> has the same fingerprint
    /// <paramref name="hash"/> (configuration and champion), <see cref="BotSessionDecision.Adopt"/> when it is a
    /// backfilled session (no fingerprint) served by the same champion, <see cref="BotSessionDecision.New"/>
    /// otherwise — including without any current session.
    /// </summary>
    public static BotSessionDecision Decide(CurrentBotSession? current, string hash, long? championId)
    {
        ArgumentException.ThrowIfNullOrEmpty(hash);
        return current switch
        {
            null => BotSessionDecision.New,
            { ConfigHash: null } when current.ModelId == championId => BotSessionDecision.Adopt,
            { ConfigHash: { } known } when known == hash => BotSessionDecision.Continue,
            _ => BotSessionDecision.New,
        };
    }
}
