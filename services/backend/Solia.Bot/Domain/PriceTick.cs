namespace Solia.Bot.Domain;

/// <summary>
/// A single observed price for a token at a point in time — what the bot marks open positions
/// against (for TP/SL and timeout checks). Price is <see cref="decimal"/> (money/price = NUMERIC,
/// per the Solia convention).
/// </summary>
public sealed record PriceTick
{
    public required string Mint { get; init; }

    /// <summary>When the price was observed (the event's logical time).</summary>
    public required DateTimeOffset Ts { get; init; }

    /// <summary>Latest price of the token.</summary>
    public required decimal Price { get; init; }
}
