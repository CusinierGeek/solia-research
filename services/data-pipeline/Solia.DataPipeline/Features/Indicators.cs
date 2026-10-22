namespace Solia.DataPipeline.Features;

/// <summary>
/// Pure statistical helpers for feature computation. They operate on <see cref="double"/>;
/// callers convert <see cref="decimal"/> prices. Kept separate from state so the math is
/// trivially unit-testable.
/// </summary>
public static class Indicators
{
    /// <summary>EMA smoothing factor for a span (as in a span-N exponential moving average).</summary>
    public static double AlphaFromSpan(int span) => 2.0 / (span + 1);

    public static double Ema(double previous, double value, double alpha) =>
        previous + (alpha * (value - previous));

    /// <summary>Standard deviation of simple returns (p[i]/p[i-1] - 1). 0 for fewer than two prices.</summary>
    public static double Volatility(IReadOnlyList<double> prices)
    {
        if (prices.Count < 2)
        {
            return 0;
        }

        var returns = new double[prices.Count - 1];
        for (int i = 1; i < prices.Count; i++)
        {
            double previous = prices[i - 1];
            returns[i - 1] = previous == 0 ? 0 : (prices[i] / previous) - 1;
        }

        double mean = returns.Average();
        double variance = returns.Sum(r => (r - mean) * (r - mean)) / returns.Length;
        return Math.Sqrt(variance);
    }

    /// <summary>
    /// Simple-average RSI over the last <paramref name="period"/> price deltas, in [0,100].
    /// Null when there isn't enough history (fewer than <paramref name="period"/>+1 prices).
    /// </summary>
    public static double? Rsi(IReadOnlyList<double> prices, int period)
    {
        if (period < 1 || prices.Count <= period)
        {
            return null;
        }

        double gain = 0;
        double loss = 0;
        for (int i = prices.Count - period; i < prices.Count; i++)
        {
            double delta = prices[i] - prices[i - 1];
            if (delta >= 0)
            {
                gain += delta;
            }
            else
            {
                loss -= delta;
            }
        }

        double averageGain = gain / period;
        double averageLoss = loss / period;
        if (averageLoss == 0)
        {
            return averageGain == 0 ? 50 : 100; // flat -> neutral; only-up -> max
        }

        double rs = averageGain / averageLoss;
        return 100 - (100 / (1 + rs));
    }

    /// <summary>Per-second rate of change between two readings. 0 when the interval is non-positive.</summary>
    public static double GrowthRate(double previous, double current, double deltaSeconds) =>
        deltaSeconds <= 0 ? 0 : (current - previous) / deltaSeconds;
}
