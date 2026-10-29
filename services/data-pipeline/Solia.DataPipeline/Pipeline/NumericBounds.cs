using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// The NUMERIC bounds of migration 0002: prices are <c>NUMERIC(28,24)</c> (|v| &lt; 10^4), amounts
/// <c>NUMERIC(28,12)</c> (|v| &lt; 10^16). PostgreSQL rejects the whole statement on one value that
/// doesn't fit, so the sink writes such a value as NULL instead: one outlier token (e.g. a pool quoted
/// in another token) must not drop every snapshot of its flush.
/// </summary>
internal static class NumericBounds
{
    private const decimal PriceLimit = 10_000m;
    private const decimal AmountLimit = 10_000_000_000_000_000m;

    // PostgreSQL rounds to the column scale before checking, but a decimal (28-29 significant digits)
    // can't hold a value just under a limit with more fractional digits than the scale, so no
    // in-range decimal rounds up to it.
    public static bool FitsPrice(decimal value) => Math.Abs(value) < PriceLimit;

    public static bool FitsAmount(decimal value) => Math.Abs(value) < AmountLimit;

    /// <summary>
    /// The snapshot with every out-of-range NUMERIC value set to null, and the names of the columns
    /// that were nulled (empty, and the same instance returned, when everything fits).
    /// </summary>
    public static FeatureSnapshot Bound(FeatureSnapshot s, out IReadOnlyList<string> nulled)
    {
        List<string>? columns = null;
        decimal? price = Check(s.Price, FitsPrice, "price");
        decimal? volume = Check(s.Volume, FitsAmount, "volume");
        decimal? marketCap = Check(s.MarketCap, FitsAmount, "market_cap");
        decimal? liquidity = Check(s.Liquidity, FitsAmount, "liquidity");
        decimal? netSolFlow = Check(s.NetSolFlow, FitsAmount, "net_sol_flow");
        decimal? avgBuySol = Check(s.AvgBuySol, FitsAmount, "avg_buy_sol");
        decimal? maxBuySol = Check(s.MaxBuySol, FitsAmount, "max_buy_sol");

        nulled = columns ?? (IReadOnlyList<string>)[];
        return columns is null
            ? s
            : s with
            {
                Price = price,
                Volume = volume,
                MarketCap = marketCap,
                Liquidity = liquidity,
                NetSolFlow = netSolFlow,
                AvgBuySol = avgBuySol,
                MaxBuySol = maxBuySol,
            };

        decimal? Check(decimal? value, Func<decimal, bool> fits, string column)
        {
            if (value is { } v && !fits(v))
            {
                (columns ??= []).Add(column);
                return null;
            }

            return value;
        }
    }
}
