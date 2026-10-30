namespace Solia.Core.Data.Entities;

/// <summary>
/// A run of the paper bot under one configuration and champion — table <c>bot_sessions</c> (migration 0012, issue #169).
/// Written only by the bot's <c>PostgresBotSessionStore</c> (raw SQL); read here.
/// </summary>
public sealed class BotSessionRecord
{
    public long Id { get; set; }

    public DateTimeOffset StartedAt { get; set; }

    /// <summary>Null for the current session.</summary>
    public DateTimeOffset? EndedAt { get; set; }

    /// <summary>Null for a session backfilled without a known configuration.</summary>
    public string? ConfigHash { get; set; }

    /// <summary>The effective bot options at the session's start (JSONB); null when backfilled.</summary>
    public string? Options { get; set; }

    /// <summary>FK → <c>models</c>, <c>ON DELETE SET NULL</c>.</summary>
    public long? ModelId { get; set; }

    public string? ModelVersion { get; set; }

    public string? Label { get; set; }

    /// <summary>Open trades with an id at or below this were left by an earlier bot run: orphans.</summary>
    public long ResumedAfterTradeId { get; set; }
}
