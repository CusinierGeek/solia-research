namespace Solia.Bot.Domain;

/// <summary>
/// The pool a token trades in, as of one observed price: its venue and the SOL it holds (see
/// <see cref="PriceTick.Venue"/> / <see cref="PriceTick.Liquidity"/>). <c>default</c> = nothing known.
/// </summary>
/// <param name="Venue">Venue name (pumpfun, pumpswap, raydium, …); null = unknown.</param>
/// <param name="LiquiditySol">Real SOL in the SOL-quoted pool; null = unknown.</param>
public readonly record struct Pool(string? Venue, decimal? LiquiditySol);
