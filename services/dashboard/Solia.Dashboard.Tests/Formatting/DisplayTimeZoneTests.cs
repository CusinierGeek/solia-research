using Solia.Dashboard.Formatting;

namespace Solia.Dashboard.Tests.Formatting;

public class DisplayTimeZoneTests
{
    private static readonly TimeZoneInfo Toronto = TimeZoneInfo.FindSystemTimeZoneById("America/Toronto");

    [Fact]
    public void A_browser_zone_resolves_by_its_iana_id()
    {
        Assert.Equal(Toronto.BaseUtcOffset, DisplayTimeZone.Resolve("America/Toronto").BaseUtcOffset);
        Assert.Equal(Toronto.BaseUtcOffset, DisplayTimeZone.Resolve(" America/Toronto ").BaseUtcOffset);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("Not/AZone")]
    public void A_missing_or_unknown_zone_falls_back_to_utc(string? id) =>
        Assert.Same(TimeZoneInfo.Utc, DisplayTimeZone.Resolve(id));

    [Fact]
    public void The_label_names_utc_and_any_other_zone_local()
    {
        Assert.Equal("UTC", DisplayTimeZone.Label(TimeZoneInfo.Utc));
        Assert.Equal("UTC", DisplayTimeZone.Label(DisplayTimeZone.Resolve("Etc/UTC")));
        Assert.Equal("local", DisplayTimeZone.Label(Toronto));
    }

    [Fact]
    public void An_instant_is_shown_on_the_zone_wall_clock_across_dst()
    {
        // 2026-11-01 06:00 UTC: Toronto falls back from EDT (-4) to EST (-5).
        Assert.Equal(new DateTime(2026, 10, 27, 7, 58, 0), DisplayTimeZone.ToWallClock(new(2026, 10, 27, 11, 58, 0, TimeSpan.Zero), Toronto));
        Assert.Equal(new DateTime(2026, 11, 2, 6, 58, 0), DisplayTimeZone.ToWallClock(new(2026, 11, 2, 11, 58, 0, TimeSpan.Zero), Toronto));
        Assert.Equal(DateTimeKind.Unspecified, DisplayTimeZone.ToWallClock(DateTimeOffset.UnixEpoch, Toronto).Kind);
    }

    [Fact]
    public void A_wall_clock_time_is_read_back_as_its_utc_instant()
    {
        DateTimeOffset utc = new(2026, 10, 27, 11, 58, 0, TimeSpan.Zero);

        Assert.Equal(utc, DisplayTimeZone.FromWallClock(DisplayTimeZone.ToWallClock(utc, Toronto), Toronto));
        Assert.Equal(TimeSpan.Zero, DisplayTimeZone.FromWallClock(new DateTime(2026, 10, 27, 7, 58, 0), Toronto).Offset);
        Assert.Equal(utc, DisplayTimeZone.FromWallClock(new DateTime(2026, 10, 27, 11, 58, 0, DateTimeKind.Utc), TimeZoneInfo.Utc));
    }

    [Fact]
    public void A_wall_clock_time_skipped_or_repeated_by_dst_takes_the_standard_offset()
    {
        // 2026-03-08 02:30 doesn't exist in Toronto (clocks jump 02:00 → 03:00); 2026-11-01 01:30 happens twice.
        Assert.Equal(new DateTimeOffset(2026, 3, 8, 7, 30, 0, TimeSpan.Zero), DisplayTimeZone.FromWallClock(new DateTime(2026, 3, 8, 2, 30, 0), Toronto));
        Assert.Equal(new DateTimeOffset(2026, 11, 1, 6, 30, 0, TimeSpan.Zero), DisplayTimeZone.FromWallClock(new DateTime(2026, 11, 1, 1, 30, 0), Toronto));
    }

    [Fact]
    public void Timestamps_are_written_in_the_display_zone()
    {
        DateTimeOffset ts = new(2026, 10, 27, 11, 58, 0, TimeSpan.Zero);

        Assert.Equal("2026-10-27 07:58:00", DisplayFormat.Timestamp(ts, Toronto));
        Assert.Equal("2026-10-27 11:58:00", DisplayFormat.Timestamp(ts, TimeZoneInfo.Utc));
        Assert.Equal("—", DisplayFormat.Timestamp((DateTimeOffset?)null, Toronto));
        Assert.Equal("2026-10-27T07:58:00.0000000-04:00", DisplayFormat.ExactTimestamp(ts, Toronto));
    }
}
