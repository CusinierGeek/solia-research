using System.Globalization;
using System.Numerics;

namespace Solia.Dashboard.Formatting;

/// <summary>
/// How every view writes the backend's values: probabilities as percentages, confidence as a 0–100 score, mints
/// shortened, timestamps as an age. Culture-invariant so the dashboard reads the same whatever the server locale; a
/// missing (or non-finite) value is shown as <see cref="Missing"/>.
/// </summary>
public static class DisplayFormat
{
    /// <summary>The placeholder for a value the backend didn't provide.</summary>
    public const string Missing = "—";

    /// <summary>A 0–1 probability as a percentage with one decimal: <c>0.8123</c> → <c>81.2%</c>.</summary>
    public static string Percent(double? ratio) =>
        ratio is { } r && double.IsFinite(r) ? r.ToString("0.0%", CultureInfo.InvariantCulture) : Missing;

    /// <summary>A 0–100 confidence score, rounded: <c>74.4</c> → <c>74</c>.</summary>
    public static string Score(double? score) =>
        score is { } s && double.IsFinite(s) ? s.ToString("0", CultureInfo.InvariantCulture) : Missing;

    /// <summary>A mint's first and last four characters (<c>So1a…xyz9</c>); short mints are kept whole.</summary>
    public static string ShortMint(string? mint) =>
        string.IsNullOrEmpty(mint) ? Missing : mint.Length <= 11 ? mint : $"{mint[..4]}…{mint[^4..]}";

    /// <summary>A count with thousands separators: <c>1234</c> → <c>1,234</c>.</summary>
    public static string Count(long count) => count.ToString("N0", CultureInfo.InvariantCulture);

    /// <summary>An exact amount or price with its trailing zeros dropped: <c>0.000000123400</c> → <c>0.0000001234</c>.</summary>
    public static string Amount(decimal? value) =>
        value is { } v ? v.ToString("0.############################", CultureInfo.InvariantCulture) : Missing;

    /// <summary>An exact gain or loss, signed: <c>+0.0123</c>, <c>-0.0045</c>, <c>0</c>.</summary>
    public static string SignedAmount(decimal? value) =>
        value is { } v ? (v > 0 ? "+" : "") + Amount(v) : Missing;

    /// <summary>A ratio as a signed percentage with one decimal: <c>0.25</c> → <c>+25.0%</c>, <c>-0.034</c> → <c>-3.4%</c>.</summary>
    public static string SignedPercent(double? ratio) =>
        ratio is { } r && double.IsFinite(r) ? r.ToString("+0.0%;-0.0%;0.0%", CultureInfo.InvariantCulture) : Missing;

    /// <summary>The CSS class colouring a gain or a loss; none for break-even or a missing value.</summary>
    public static string? PnlClass<T>(T? value) where T : struct, INumber<T> => value is { } v ? PnlClass(v) : null;

    /// <inheritdoc cref="PnlClass{T}(T?)"/>
    public static string? PnlClass<T>(T value) where T : struct, INumber<T> =>
        value > T.Zero ? "solia-pnl--gain" : value < T.Zero ? "solia-pnl--loss" : null;

    /// <summary>A duration in seconds, in its two largest units: <c>45s</c>, <c>3m 07s</c>, <c>1h 05m</c>, <c>2d 03h</c>.</summary>
    public static string Duration(int? seconds)
    {
        if (seconds is not { } s || s < 0)
        {
            return Missing;
        }
        var d = TimeSpan.FromSeconds(s);
        return s switch
        {
            < 60 => $"{s}s",
            < 3600 => $"{d.Minutes}m {d.Seconds:00}s",
            < 86400 => $"{d.Hours}h {d.Minutes:00}m",
            _ => $"{(int)d.TotalDays}d {d.Hours:00}h",
        };
    }

    /// <summary>A timestamp to the second in the display zone (see <see cref="DisplayTimeZone"/>): <c>2026-10-27 07:58:00</c>.</summary>
    public static string Timestamp(DateTimeOffset ts, TimeZoneInfo zone) =>
        DisplayTimeZone.ToWallClock(ts, zone).ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);

    /// <summary><see cref="Timestamp(DateTimeOffset, TimeZoneInfo)"/>, or <see cref="Missing"/> when there is none.</summary>
    public static string Timestamp(DateTimeOffset? ts, TimeZoneInfo zone) => ts is { } t ? Timestamp(t, zone) : Missing;

    /// <summary>A timestamp to the tick with its zone's offset, for a hover title: <c>2026-10-27T07:58:00.0000000-04:00</c>.</summary>
    public static string ExactTimestamp(DateTimeOffset ts, TimeZoneInfo zone) =>
        TimeZoneInfo.ConvertTime(ts, zone).ToString("O", CultureInfo.InvariantCulture);

    /// <summary>How long ago <paramref name="ts"/> was, in its largest unit: <c>now</c>, <c>12s</c>, <c>3m</c>, <c>2h</c>, <c>5d</c>.</summary>
    public static string Age(DateTimeOffset ts, DateTimeOffset now)
    {
        TimeSpan age = now - ts;
        return age.TotalSeconds switch
        {
            < 1 => "now", // includes a ts slightly ahead of the dashboard's clock
            < 60 => $"{(int)age.TotalSeconds}s",
            < 3600 => $"{(int)age.TotalMinutes}m",
            < 86400 => $"{(int)age.TotalHours}h",
            _ => $"{(int)age.TotalDays}d",
        };
    }
}
