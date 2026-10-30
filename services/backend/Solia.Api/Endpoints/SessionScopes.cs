using System.Globalization;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Solia.Core.Data;
using Solia.Core.Data.Entities;
using Solia.Core.Http;

namespace Solia.Api.Endpoints;

/// <summary>
/// The trades a read covers, by bot session (issue #169): <see cref="Session"/> is the session whose trades count,
/// or null for every trade (<c>?session=all</c>, or no session recorded yet).
/// </summary>
public sealed record SessionScope(BotSessionRecord? Session)
{
    public static readonly SessionScope All = new((BotSessionRecord?)null);

    /// <summary>The id of the session read, null for every trade.</summary>
    public long? Id => Session?.Id;

    /// <summary>
    /// The starting balance the session's equity curve starts from — the one recorded in its options — or
    /// <paramref name="configured"/> for every trade, or a session recorded without one (backfilled).
    /// </summary>
    public decimal StartingBalance(decimal configured)
    {
        if (Session?.Options is not { } options)
        {
            return configured;
        }

        using JsonDocument doc = JsonDocument.Parse(options);
        return doc.RootElement.TryGetProperty("startingBalance", out JsonElement value) && value.TryGetDecimal(out decimal balance)
            ? balance
            : configured;
    }
}

/// <summary>
/// <c>?session=</c> on the trade reads: <c>current</c> (the default) | <c>all</c> | a session id. The current session is
/// the one without an <c>ended_at</c>; before the bot has ever recorded one, <c>current</c> reads every trade.
/// </summary>
public static class SessionScopes
{
    public const string Current = "current";

    public const string AllKey = "all";

    /// <summary>The scope <paramref name="session"/> names.</summary>
    /// <exception cref="InvalidRequestException">Neither <c>current</c>, <c>all</c> nor a positive id.</exception>
    /// <exception cref="NotFoundException">No session has that id.</exception>
    public static async Task<SessionScope> ResolveAsync(SoliaDbContext db, string? session, CancellationToken cancellationToken)
    {
        string key = string.IsNullOrWhiteSpace(session) ? Current : session.Trim().ToLowerInvariant();
        if (key == AllKey)
        {
            return SessionScope.All;
        }

        IQueryable<BotSessionRecord> sessions = db.BotSessions.AsNoTracking();
        if (key == Current)
        {
            return new SessionScope(await sessions.FirstOrDefaultAsync(s => s.EndedAt == null, cancellationToken));
        }

        if (!long.TryParse(key, NumberStyles.None, CultureInfo.InvariantCulture, out long id) || id < 1)
        {
            throw new InvalidRequestException("session must be current, all or a session id.");
        }

        return new SessionScope(await sessions.FirstOrDefaultAsync(s => s.Id == id, cancellationToken)
            ?? throw new NotFoundException($"No bot session {id}."));
    }

    /// <summary>The trades of <paramref name="scope"/>.</summary>
    public static IQueryable<Trade> In(this IQueryable<Trade> trades, SessionScope scope) =>
        scope.Id is { } id ? trades.Where(t => t.SessionId == id) : trades;
}
