namespace Solia.Core.Data.Entities;

/// <summary>
/// Immutable registry of every token ever seen (one row per mint) — table <c>tokens_history</c>.
/// The FK target for <see cref="Feature"/>, <see cref="Prediction"/> and <see cref="Trade"/>.
/// Read-oriented: written by the data-pipeline, mapped here for the API's registry reads.
/// </summary>
public sealed class TokenHistory
{
    /// <summary>Token mint address (primary key).</summary>
    public string Mint { get; set; } = null!;

    public string? Symbol { get; set; }

    public string? Name { get; set; }

    public DateTimeOffset? CreatedAt { get; set; }

    /// <summary>When the pipeline first ingested the token (DB default <c>now()</c>).</summary>
    public DateTimeOffset FirstSeenAt { get; set; }

    public DateTimeOffset? LastSeenAt { get; set; }

    /// <summary>Labeled later: <c>pump</c> / <c>rug</c> / <c>neutral</c> (CHECK lives in SQL).</summary>
    public string? FinalOutcome { get; set; }

    /// <summary>Raw source metadata (JSONB).</summary>
    public string? Metadata { get; set; }
}
