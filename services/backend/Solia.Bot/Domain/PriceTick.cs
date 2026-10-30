namespace Solia.Bot.Domain;

/// <summary>
/// A single observed price for a token at a point in time — what the bot marks open positions
/// against (for TP/SL and timeout checks) — and the pool it was read off, which the simulated fills
/// trade against (see <c>Trading.Fills</c>). Price is <see cref="decimal"/> (money/price = NUMERIC,
/// per the Solia convention).
/// </summary>
public sealed record PriceTick
{
    public required string Mint { get; init; }

    /// <summary>When the price was observed (the event's logical time).</summary>
    public required DateTimeOffset Ts { get; init; }

    /// <summary>Latest price of the token, in SOL.</summary>
    public required decimal Price { get; init; }

    /// <summary>Venue the price was read on (<c>features.venue</c>: pumpfun, pumpswap, raydium, …); null = unknown.</summary>
    public string? Venue { get; init; }

    /// <summary>
    /// SOL in the token's SOL-quoted pool (<c>features.liquidity</c>) — on a bonding curve the <em>real</em> SOL,
    /// without the curve's virtual reserve. Null = unknown.
    /// </summary>
    public decimal? Liquidity { get; init; }

    /// <summary>The pool this price was read off.</summary>
    public Pool Pool => new(Venue, Liquidity);
}
