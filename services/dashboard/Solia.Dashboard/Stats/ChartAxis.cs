using System.Globalization;

namespace Solia.Dashboard.Stats;

/// <summary>
/// A value axis fitted to a series: about <see cref="TargetTicks"/> ticks on a 1–2–5 step. The charting library
/// only spaces ticks by whole units, so a series is plotted multiplied by <see cref="Scale"/> (a power of ten),
/// which makes a fractional step — a P&amp;L of a few hundredths — a whole <see cref="Ticks"/>; <see cref="Label"/>
/// writes a plotted value back in the series' own units.
/// </summary>
/// <param name="Scale">What a value is multiplied by to be plotted (1 when the step is whole already).</param>
/// <param name="Ticks">The tick step, in plotted units.</param>
/// <param name="Digits">The decimals a tick label needs.</param>
public sealed record ChartAxis(double Scale, int Ticks, int Digits)
{
    internal const int TargetTicks = 5;

    /// <summary>The most decimals a label is rounded to (<see cref="Math.Round(double, int)"/>'s own limit).</summary>
    internal const int MaxDigits = 15;

    /// <summary>The axis for <paramref name="values"/>, spanning zero too when <paramref name="includeZero"/>.</summary>
    public static ChartAxis Fit(IEnumerable<double> values, bool includeZero = false)
    {
        double min = double.MaxValue, max = double.MinValue;
        foreach (double value in values.Where(double.IsFinite))
        {
            min = Math.Min(min, value);
            max = Math.Max(max, value);
        }
        if (min > max)
        {
            min = max = 0;
        }
        if (includeZero)
        {
            min = Math.Min(min, 0);
            max = Math.Max(max, 0);
        }

        // A flat series (one trade, or a balance line) still gets a readable step around its value.
        double range = max - min;
        if (range <= 0)
        {
            range = Math.Abs(max) > 0 ? Math.Abs(max) / 10 : 1;
        }

        double raw = range / TargetTicks;
        int exponent = (int)Math.Floor(Math.Log10(raw));
        double fraction = raw / Math.Pow(10, exponent);
        int nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
        return exponent >= 0
            ? new ChartAxis(1, (int)Math.Min(int.MaxValue, nice * Math.Pow(10, exponent)), 0)
            : new ChartAxis(Math.Pow(10, -exponent), nice, Math.Min(-exponent, MaxDigits));
    }

    /// <summary>A value in plotted units.</summary>
    public double Plot(double value) => value * Scale;

    /// <summary>A plotted value back in the series' units, to the tick's precision: <c>1234</c> at scale 100 → <c>12.34</c>.</summary>
    public string Label(double plotted) =>
        Math.Round(plotted / Scale, Digits).ToString("0.###############", CultureInfo.InvariantCulture);
}
