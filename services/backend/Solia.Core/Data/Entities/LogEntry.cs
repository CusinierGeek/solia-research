namespace Solia.Core.Data.Entities;

/// <summary>
/// Operational log surfaced on the dashboard — table <c>logs</c>.
/// Written by the backend/bot log sinks (raw SQL); read here for the dashboard.
/// </summary>
public sealed class LogEntry
{
    public long Id { get; set; }

    /// <summary>DB default <c>now()</c>.</summary>
    public DateTimeOffset Ts { get; set; }

    /// <summary><c>data-pipeline</c> / <c>ai-service</c> / <c>backend</c> / <c>bot</c>.</summary>
    public string Service { get; set; } = null!;

    /// <summary><c>debug</c> / <c>info</c> / <c>warn</c> / <c>error</c> (CHECK lives in SQL).</summary>
    public string Level { get; set; } = null!;

    public string Message { get; set; } = null!;

    /// <summary>Structured fields (JSONB).</summary>
    public string? Context { get; set; }
}
