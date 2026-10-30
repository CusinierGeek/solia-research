using Solia.Bot.Configuration;
using Solia.Bot.Domain;

namespace Solia.Bot.Trading;

/// <summary>
/// How a paper order fills against the pool it trades in (docs/ML-MODEL.md "Policy layer", "Realistic fills").
/// Pure and deterministic. Applies only with <see cref="BotOptions.RealisticFills"/>; amounts are SOL.
/// <list type="bullet">
///   <item><b>Reserve.</b> The SOL side of a constant-product pool, <c>R</c> = the pool's real SOL
///   (<see cref="Pool.LiquiditySol"/>) + the venue's virtual SOL (<see cref="BotOptions.VirtualSol"/>, non-zero on
///   bonding curves only: they price off virtual reserves, so a fresh curve with almost no real SOL still has a
///   deep, valid price). Unknown liquidity = unknown reserve.</item>
///   <item><b>Gate.</b> A pool is <see cref="IsTradeable"/> when its liquidity is known, at least its venue's
///   <see cref="BotOptions.LiquidityFloorSol"/>, and its reserve is positive.</item>
///   <item><b>Cap.</b> A position commits at most <see cref="BotOptions.MaxPoolFraction"/> of <c>R</c>
///   (<see cref="MaxSize"/>).</item>
///   <item><b>Buy</b> of <c>S</c> SOL at mark <c>P</c>: the venue fee <c>f</c> is taken first, the rest
///   <c>n = S(1 − f)</c> swaps along <c>x·y = k</c> for <c>n/P · R/(R + n)</c> tokens, i.e. an all-in price of
///   <c>P · (R + n) / (R · (1 − f))</c> (<see cref="BuyPrice"/>).</item>
///   <item><b>Sell</b> of a position worth <c>v</c> at mark <c>P</c>: <c>v · R/(R + v)</c> SOL come out of the
///   pool, less the fee, i.e. an exit price of <c>P · R/(R + v) · (1 − f)</c> (<see cref="SellPrice"/>). The
///   proceeds stay below <c>R</c>: an empty or unknown pool pays nothing.</item>
/// </list>
/// <b>Approximation.</b> Each side is priced against the pool as observed, which never contains the bot's own
/// (paper) trade, so a round trip at an unchanged price pays the impact twice where on-chain it would mostly
/// unwind — pessimistic by design, and bounded by the pool cap (2 % of <c>R</c>: ~2 % per side). Triggers
/// (TP/SL) still read the observed mark; only the fills pay impact and fees.
/// </summary>
public static class Fills
{
    /// <summary>Key of a per-venue table's fallback, for every venue not listed (an unknown venue included).</summary>
    public const string DefaultVenue = "default";

    /// <summary>The <see cref="BotOptions.LiquidityFloorSol"/> of <paramref name="venue"/>.</summary>
    public static decimal FloorOf(string? venue, BotOptions options) => ForVenue(options.LiquidityFloorSol, venue, 0m);

    /// <summary>The per-side <see cref="BotOptions.FeePct"/> of <paramref name="venue"/>.</summary>
    public static double FeeOf(string? venue, BotOptions options) => ForVenue(options.FeePct, venue, 0.0);

    /// <summary>The <see cref="BotOptions.VirtualSol"/> of <paramref name="venue"/>.</summary>
    public static decimal VirtualSolOf(string? venue, BotOptions options) => ForVenue(options.VirtualSol, venue, 0m);

    /// <summary>The SOL reserve <c>R</c> an order prices against; null when the pool's liquidity is unknown.</summary>
    public static decimal? ReserveSol(Pool pool, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return pool.LiquiditySol is decimal liquidity && liquidity >= 0m
            ? liquidity + VirtualSolOf(pool.Venue, options)
            : null;
    }

    /// <summary>Whether a position may be opened in <paramref name="pool"/> (the liquidity gate, see the type summary).</summary>
    public static bool IsTradeable(Pool pool, BotOptions options) =>
        ReserveSol(pool, options) is > 0m && pool.LiquiditySol >= FloorOf(pool.Venue, options);

    /// <summary>The largest position <paramref name="pool"/> takes: <see cref="BotOptions.MaxPoolFraction"/> of its reserve (0 if unknown).</summary>
    public static decimal MaxSize(Pool pool, BotOptions options) =>
        ReserveSol(pool, options) is decimal reserve
            ? Money.Truncate(reserve * (decimal)options.MaxPoolFraction)
            : 0m;

    /// <summary>All-in price per token of buying <paramref name="size"/> SOL at <paramref name="mark"/> in a tradeable pool.</summary>
    /// <exception cref="ArgumentException">The pool isn't tradeable (no positive reserve).</exception>
    public static decimal BuyPrice(decimal size, decimal mark, Pool pool, BotOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(size);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(mark);
        decimal reserve = ReserveSol(pool, options) is decimal r && r > 0m
            ? r
            : throw new ArgumentException("A buy needs a pool with a positive SOL reserve.", nameof(pool));

        decimal keep = 1m - (decimal)FeeOf(pool.Venue, options);
        decimal swapped = size * keep;
        return mark * (reserve + swapped) / (reserve * keep);
    }

    /// <summary>
    /// Price per token the <paramref name="entry"/> position sells at, marked at <paramref name="mark"/>, into
    /// <paramref name="pool"/>: 0 when the pool is empty or unknown (nothing can be paid out).
    /// </summary>
    public static decimal SellPrice(TradeOpening entry, decimal mark, Pool pool, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(entry);
        ArgumentOutOfRangeException.ThrowIfNegative(mark);
        if (mark == 0m || ReserveSol(pool, options) is not decimal reserve || reserve <= 0m)
        {
            return 0m;
        }

        decimal value = entry.ValueAt(mark);
        decimal keep = 1m - (decimal)FeeOf(pool.Venue, options);
        return mark * reserve * keep / (reserve + value);
    }

    private static T ForVenue<T>(IReadOnlyDictionary<string, T> table, string? venue, T fallback) =>
        venue is not null && table.TryGetValue(venue.Trim(), out T? value) ? value
        : table.TryGetValue(DefaultVenue, out T? @default) ? @default
        : fallback;
}
