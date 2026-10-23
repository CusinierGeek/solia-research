namespace Solia.Core.Data.Entities;

/// <summary>
/// Model output, one row per inference call — table <c>predictions</c>.
/// Read-oriented here: written by the ai-service/bot, read by the API.
/// </summary>
public sealed class Prediction
{
    public long Id { get; set; }

    /// <summary>FK → <c>tokens_history</c>.</summary>
    public string Mint { get; set; } = null!;

    /// <summary>FK → <c>models</c> (nullable).</summary>
    public long? ModelId { get; set; }

    /// <summary>Inference time (DB default <c>now()</c>).</summary>
    public DateTimeOffset Ts { get; set; }

    /// <summary><c>30s</c> / <c>1m</c> / <c>5m</c> / <c>15m</c> / <c>1h</c> (CHECK lives in SQL).</summary>
    public string Horizon { get; set; } = null!;

    public double? PumpProb { get; set; }

    public double? RugProb { get; set; }

    /// <summary><c>buy</c> / <c>sell</c> / <c>hold</c> (CHECK lives in SQL).</summary>
    public string? Strategy { get; set; }

    public decimal? PositionSize { get; set; }

    public decimal? TakeProfit { get; set; }

    public decimal? StopLoss { get; set; }

    public double? Confidence { get; set; }

    /// <summary>FK → <c>features</c> (input snapshot), <c>ON DELETE SET NULL</c>.</summary>
    public long? FeatureId { get; set; }

    public TokenHistory? Token { get; set; }

    public ModelRecord? Model { get; set; }

    public Feature? Feature { get; set; }
}
