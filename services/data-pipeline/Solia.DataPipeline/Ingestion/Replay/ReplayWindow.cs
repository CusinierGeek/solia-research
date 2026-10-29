namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// A half-open UTC range <c>[StartUtc, EndUtc)</c> to replay. Enumerating <see cref="Hours"/>
/// yields each hourly archive that overlaps it; <see cref="Contains"/> trims events to the exact
/// bounds, so a window that doesn't start or end on the hour returns nothing outside it.
/// </summary>
public readonly record struct ReplayWindow(DateTimeOffset StartUtc, DateTimeOffset EndUtc)
{
    /// <summary>First hour pumpapi archived; earlier hours don't exist.</summary>
    public static readonly DateTimeOffset ArchiveStartUtc = new(2026, 4, 18, 0, 0, 0, TimeSpan.Zero);

    /// <summary>Throws when the window is empty, inverted, unset, or starts before the archive does.</summary>
    public void EnsureValid()
    {
        if (StartUtc >= EndUtc)
        {
            throw new ArgumentException(
                $"Replay window must have StartUtc < EndUtc (got {StartUtc:O} .. {EndUtc:O}).");
        }

        if (StartUtc < ArchiveStartUtc)
        {
            throw new ArgumentException(
                $"Replay window starts at {StartUtc:O}, before the archive begins ({ArchiveStartUtc:O}).");
        }
    }

    public bool Contains(DateTimeOffset timestamp) => timestamp >= StartUtc && timestamp < EndUtc;

    public IEnumerable<DateTimeOffset> Hours()
    {
        DateTimeOffset hour = FloorToHour(StartUtc);
        DateTimeOffset end = EndUtc.ToUniversalTime();
        while (hour < end)
        {
            yield return hour;
            hour = hour.AddHours(1);
        }
    }

    /// <summary>Start of the UTC hour containing <paramref name="value"/>.</summary>
    public static DateTimeOffset FloorToHour(DateTimeOffset value)
    {
        DateTimeOffset utc = value.ToUniversalTime();
        return new DateTimeOffset(utc.Year, utc.Month, utc.Day, utc.Hour, 0, 0, TimeSpan.Zero);
    }
}
