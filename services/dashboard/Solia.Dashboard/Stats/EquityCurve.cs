using System.Globalization;
using Solia.Dashboard.Api;
using Solia.Dashboard.Formatting;

namespace Solia.Dashboard.Stats;

/// <summary>One closed trade on the curve: the equity right after it closed, and its P&amp;L.</summary>
public sealed record EquityCurvePoint(DateTimeOffset Ts, decimal Equity, decimal Pnl);

/// <summary>Which series of the performance charts a tooltip is for.</summary>
public enum EquitySeriesKind
{
    Equity,
    StartingBalance,
    Pnl,
}

/// <summary>
/// The <c>/stats/performance</c> answer shaped for the performance charts: the points in close order, the series a
/// chart plots, and the tooltip of each point. Pure — no rendering, no library types — so the wrapper around the
/// charting library only draws what this hands it. The axes and plotted times are computed once, on creation.
/// </summary>
/// <param name="Points">The closed trades, oldest first.</param>
/// <param name="StartingBalance">
/// The bot's configured starting balance, the curve's reference line; null when nothing closed. Derived, since the
/// contract doesn't carry it: the latest point's equity is the starting balance plus <em>every</em> closed trade's
/// P&amp;L, which is <c>total_pnl</c> — even when <c>limit</c> cut the older points off.
/// </param>
/// <param name="TotalPnl">The realized P&amp;L over every closed trade.</param>
/// <param name="IsComplete">Whether every closed trade is on the curve (no older one was cut off).</param>
public sealed record EquityCurve(IReadOnlyList<EquityCurvePoint> Points, decimal? StartingBalance, decimal TotalPnl, bool IsComplete)
{
    /// <summary>The time-label spacings the x axis picks from, smallest first; past the last, multiples of it.</summary>
    internal static readonly TimeSpan[] LabelSpacings =
    [
        TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(15), TimeSpan.FromMinutes(30),
        TimeSpan.FromHours(1), TimeSpan.FromHours(3), TimeSpan.FromHours(6), TimeSpan.FromHours(12),
        TimeSpan.FromDays(1), TimeSpan.FromDays(2), TimeSpan.FromDays(3), TimeSpan.FromDays(7), TimeSpan.FromDays(14),
        TimeSpan.FromDays(30), TimeSpan.FromDays(60), TimeSpan.FromDays(90), TimeSpan.FromDays(180), TimeSpan.FromDays(365),
    ];

    /// <summary>About how many time labels the x axis shows: few enough to stay apart at phone width.</summary>
    internal const int TargetLabels = 4;

    /// <summary>
    /// The least gap between two plotted times. The library hands a tooltip the hovered point's time, not its index,
    /// so trades closed at the same instant are plotted this far apart (invisible) to tell their tooltips apart.
    /// </summary>
    internal static readonly TimeSpan MinPlotGap = TimeSpan.FromMilliseconds(1);

    // Computed once: every refresh reads them for the axes, the series and the chart options.
    private readonly DateTime[] _plotTimes = PlotTimes(Points);

    public bool IsEmpty => Points.Count == 0;

    /// <summary>
    /// Maps the backend's curve, fetched with <c>limit = shown + 1</c>: points ordered by <c>ts</c> (ties keep the
    /// backend's order, which is the close order), the latest <paramref name="shown"/> kept, exact decimals kept for
    /// display. One point more than shown means older trades exist, which a count equal to the limit can't tell.
    /// </summary>
    public static EquityCurve From(PerformanceResponse response, int shown)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(shown, 1);
        List<EquityCurvePoint> points = response.Points
            .OrderBy(p => p.Ts)
            .Select(p => new EquityCurvePoint(p.Ts, p.Equity, p.Pnl))
            .ToList();
        bool complete = points.Count <= shown;
        if (!complete)
        {
            points.RemoveRange(0, points.Count - shown);
        }
        decimal? start = points.Count == 0 ? null : points[^1].Equity - response.TotalPnl;
        return new EquityCurve(points, start, response.TotalPnl, complete);
    }

    /// <summary>The axis of the equity panel, fitted to the equity and the starting balance.</summary>
    public ChartAxis EquityAxis { get; } = ChartAxis.Fit(
        Points.Select(p => (double)p.Equity).Concat(StartingBalance is { } start ? [(double)start] : []));

    /// <summary>The axis of the P&amp;L panel, which always shows zero (gains above, losses below).</summary>
    public ChartAxis PnlAxis { get; } = ChartAxis.Fit(Points.Select(p => (double)p.Pnl), includeZero: true);

    /// <summary>The equity after each trade over its plotted UTC time, in <see cref="EquityAxis"/> units.</summary>
    public IReadOnlyList<(DateTime Time, double Value)> EquitySeries => Series(EquityAxis, p => p.Equity);

    /// <summary>The starting balance as a flat line under the same times (empty when nothing closed).</summary>
    public IReadOnlyList<(DateTime Time, double Value)> StartingBalanceSeries =>
        StartingBalance is { } start ? Series(EquityAxis, _ => start) : [];

    /// <summary>Each trade's own P&amp;L, the secondary series, in <see cref="PnlAxis"/> units.</summary>
    public IReadOnlyList<(DateTime Time, double Value)> PnlSeries => Series(PnlAxis, p => p.Pnl);

    /// <summary>The time span the curve covers (zero for one point or none).</summary>
    public TimeSpan Span => IsEmpty ? TimeSpan.Zero : Points[^1].Ts - Points[0].Ts;

    /// <summary>The smallest spacing that keeps the x axis at about <see cref="TargetLabels"/> time labels.</summary>
    public TimeSpan LabelSpacing
    {
        get
        {
            TimeSpan wanted = Span / TargetLabels;
            TimeSpan largest = LabelSpacings[^1];
            return wanted <= largest
                ? LabelSpacings.First(s => s >= wanted)
                : largest * Math.Ceiling(wanted / largest);
        }
    }

    /// <summary>The time-label format: clock time within a day, the date alone once labels are days apart, both between.</summary>
    public string LabelFormat =>
        LabelSpacing >= TimeSpan.FromDays(365) ? "yyyy-MM"
        : LabelSpacing >= TimeSpan.FromDays(1) ? "MM-dd"
        : Span <= TimeSpan.FromDays(1) ? "HH:mm" : "MM-dd HH:mm";

    /// <summary>
    /// How the chart writes a hovered point's time — round-trip, so <see cref="Tooltip"/> can find the trade again
    /// (the library hands the template the series, not the point, index).
    /// </summary>
    public const string TooltipTimeFormat = "o";

    /// <summary>How the chart writes a hovered point's plotted value — round-trip, for the fallback tooltip.</summary>
    public const string TooltipValueFormat = "R";

    /// <summary>
    /// The tooltip of the <paramref name="series"/> point plotted at <paramref name="time"/> (written in
    /// <see cref="TooltipTimeFormat"/>), the UTC timestamp as its subtitle: equity and trade P&amp;L for a trade, the
    /// balance for the reference line. A time the curve can't place (the library wrote it some other way) still
    /// gets a tooltip, from what the library hovered: <paramref name="plotted"/> (in <see cref="TooltipValueFormat"/>)
    /// labelled in the series' units, and the time as written.
    /// </summary>
    public (string Title, string Subtitle) Tooltip(EquitySeriesKind series, string? time, string? plotted)
    {
        if (PointAt(time) is not { } index)
        {
            ChartAxis axis = series == EquitySeriesKind.Pnl ? PnlAxis : EquityAxis;
            // The library writes the value in the current culture; unscaled, rounded off the double's own noise.
            string value = double.TryParse(plotted, NumberStyles.Float, CultureInfo.CurrentCulture, out double y) && Math.Abs(y / axis.Scale) < 1e15
                ? DisplayFormat.Amount((decimal)Math.Round(y / axis.Scale, 12))
                : plotted ?? DisplayFormat.Missing;
            return (value, time ?? "");
        }
        EquityCurvePoint point = Points[index];
        string subtitle = $"{DisplayFormat.Timestamp(point.Ts)} UTC";
        return series == EquitySeriesKind.StartingBalance
            ? ($"Starting balance {DisplayFormat.Amount(StartingBalance)}", subtitle)
            : ($"Equity {DisplayFormat.Amount(point.Equity)} · P&L {DisplayFormat.SignedAmount(point.Pnl)}", subtitle);
    }

    /// <summary>
    /// Whether <paramref name="other"/> would draw the very same chart: a refresh with nothing new maps a new curve
    /// with the same points, which needn't be redrawn.
    /// </summary>
    public bool HasSameData(EquityCurve? other) =>
        other is not null
        && (ReferenceEquals(this, other)
            || (StartingBalance == other.StartingBalance && TotalPnl == other.TotalPnl && IsComplete == other.IsComplete
                && Points.SequenceEqual(other.Points)));

    // The point plotted nearest the parsed time, if within half a gap of it (plotted times are at least a gap apart).
    private int? PointAt(string? time)
    {
        if (!DateTime.TryParseExact(time, TooltipTimeFormat, CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out DateTime at)
            || _plotTimes.Length == 0)
        {
            return null;
        }
        at = at.ToUniversalTime();
        int found = Array.BinarySearch(_plotTimes, at);
        if (found >= 0)
        {
            return found;
        }
        int next = ~found;
        int nearest = next == 0 ? 0
            : next == _plotTimes.Length ? next - 1
            : at - _plotTimes[next - 1] <= _plotTimes[next] - at ? next - 1 : next;
        return (at - _plotTimes[nearest]).Duration() * 2 < MinPlotGap ? nearest : null;
    }

    // Each point's UTC close time, pushed just past the previous one when they tie (or the backend's order dips).
    private static DateTime[] PlotTimes(IReadOnlyList<EquityCurvePoint> points)
    {
        var times = new DateTime[points.Count];
        for (int i = 0; i < times.Length; i++)
        {
            DateTime time = points[i].Ts.UtcDateTime;
            times[i] = i > 0 && time < times[i - 1] + MinPlotGap ? times[i - 1] + MinPlotGap : time;
        }
        return times;
    }

    private List<(DateTime Time, double Value)> Series(ChartAxis axis, Func<EquityCurvePoint, decimal> value) =>
        Points.Select((p, i) => (_plotTimes[i], axis.Plot((double)value(p)))).ToList();
}
