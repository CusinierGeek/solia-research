namespace Solia.Bot.Sessions;

/// <summary>
/// An <see cref="IBotSessionStore"/> held in memory — for unit tests and hosts without a database. Applies the same
/// <see cref="BotSessionRules"/> as <see cref="PostgresBotSessionStore"/> against a fixed <see cref="ChampionId"/> /
/// <see cref="ChampionVersion"/>. Ids are sequential from 1. Not thread-safe.
/// </summary>
public sealed class InMemoryBotSessionStore : IBotSessionStore
{
    private readonly List<(BotSession Session, string? Label, bool Ended)> _sessions = [];

    /// <summary>The champion the next <see cref="BeginAsync"/> reads; settable to simulate a promotion.</summary>
    public long? ChampionId { get; set; }

    public string? ChampionVersion { get; set; }

    /// <summary>Every session, oldest first, with its label and whether it has ended.</summary>
    public IReadOnlyList<(BotSession Session, string? Label, bool Ended)> Sessions => _sessions;

    public Task<BotSession> BeginAsync(BotSessionRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();

        int index = _sessions.FindIndex(s => !s.Ended);
        BotSession? current = index < 0 ? null : _sessions[index].Session;
        string hash = BotSessionFingerprint.Hash(request.OptionsJson, ChampionVersion);
        BotSessionDecision decision = BotSessionRules.Decide(
            current is null ? null : new CurrentBotSession(current.Id, current.ConfigHash, current.ModelId), hash, ChampionId);

        if (decision != BotSessionDecision.New)
        {
            return Task.FromResult(_sessions[index].Session with { Decision = decision });
        }

        if (index >= 0)
        {
            _sessions[index] = _sessions[index] with { Ended = true };
        }

        var session = new BotSession(_sessions.Count + 1, decision, ChampionId, ChampionVersion, hash);
        _sessions.Add((session, request.Label, false));
        return Task.FromResult(session);
    }
}
