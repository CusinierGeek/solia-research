namespace Solia.Core.Data.Entities;

/// <summary>
/// Paper-trading bot activity, one row per (open or closed) position — table <c>trades</c>.
/// Written only by the bot's <c>PostgresTradeStore</c> (raw SQL); read here.
/// Paper trading only — no real orders are ever placed.
/// </summary>
public sealed class Trade
{
    public long Id { get; set; }

    /// <summary>FK → <c>tokens_history</c>.</summary>
    public string Mint { get; set; } = null!;

    /// <summary>FK → <c>predictions</c> (triggering signal), <c>ON DELETE SET NULL</c>.</summary>
    public long? PredictionId { get; set; }

    /// <summary>DB default <c>now()</c>.</summary>
    public DateTimeOffset OpenedAt { get; set; }

    /// <summary>Nullable while the position is open.</summary>
    public DateTimeOffset? ClosedAt { get; set; }

    public decimal? EntryPrice { get; set; }

    public decimal? ExitPrice { get; set; }

    public decimal? Size { get; set; }

    public int? DurationSec { get; set; }

    public decimal? Pnl { get; set; }

    public double? PnlPct { get; set; }

    /// <summary><c>buy</c> / <c>sell</c> / <c>hold</c> (CHECK lives in SQL).</summary>
    public string? Strategy { get; set; }

    public double? Confidence { get; set; }

    /// <summary><c>tp</c> / <c>sl</c> / <c>signal</c> / <c>timeout</c> (CHECK lives in SQL).</summary>
    public string? ExitReason { get; set; }

    /// <summary>Features at entry, for training (JSONB).</summary>
    public string? FeaturesSnapshot { get; set; }

    /// <summary><c>open</c> / <c>closed</c> (DB default <c>open</c>, CHECK lives in SQL).</summary>
    public string Status { get; set; } = null!;

    public TokenHistory? Token { get; set; }

    public Prediction? Prediction { get; set; }
}
