namespace Solia.Dashboard.Formatting;

/// <summary>
/// The time zone the dashboard shows times in: the viewer's browser's, read once per circuit by the layout and
/// cascaded to every view. The backend, the URL and the API stay in UTC; only what is displayed (and typed into a
/// date filter) is local. UTC until the browser has told its zone, or when it can't be resolved.
/// </summary>
public static class DisplayTimeZone
{
    /// <summary>The browser function returning its IANA zone (defined in <c>App.razor</c>).</summary>
    public const string BrowserFunction = "soliaTimeZone";

    /// <summary>The zone with this IANA id (e.g. <c>America/Toronto</c>); UTC when blank or unknown.</summary>
    public static TimeZoneInfo Resolve(string? ianaId)
    {
        if (string.IsNullOrWhiteSpace(ianaId))
        {
            return TimeZoneInfo.Utc;
        }
        try
        {
            return TimeZoneInfo.FindSystemTimeZoneById(ianaId.Trim());
        }
        catch (Exception ex) when (ex is TimeZoneNotFoundException or InvalidTimeZoneException)
        {
            return TimeZoneInfo.Utc;
        }
    }

    /// <summary>How a column header names the zone: <c>UTC</c>, or <c>local</c> once the browser's zone is known.</summary>
    public static string Label(TimeZoneInfo zone) => zone.BaseUtcOffset == TimeSpan.Zero && !zone.SupportsDaylightSavingTime ? "UTC" : "local";

    /// <summary>A UTC instant as wall-clock time in <paramref name="zone"/> (an unspecified-kind <see cref="DateTime"/>).</summary>
    public static DateTime ToWallClock(DateTimeOffset ts, TimeZoneInfo zone) =>
        DateTime.SpecifyKind(TimeZoneInfo.ConvertTime(ts, zone).DateTime, DateTimeKind.Unspecified);

    /// <summary>
    /// The UTC instant of a wall-clock time in <paramref name="zone"/>. A time skipped or repeated by a DST change
    /// takes the zone's standard offset rather than failing.
    /// </summary>
    public static DateTimeOffset FromWallClock(DateTime wallClock, TimeZoneInfo zone)
    {
        DateTime unspecified = DateTime.SpecifyKind(wallClock, DateTimeKind.Unspecified);
        TimeSpan offset = zone.IsInvalidTime(unspecified) ? zone.BaseUtcOffset : zone.GetUtcOffset(unspecified);
        return new DateTimeOffset(unspecified, offset).ToUniversalTime();
    }
}
