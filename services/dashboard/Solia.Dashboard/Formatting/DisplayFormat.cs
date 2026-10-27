using System.Globalization;

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
