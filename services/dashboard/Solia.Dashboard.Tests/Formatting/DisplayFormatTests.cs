using System.Globalization;
using Solia.Dashboard.Formatting;

namespace Solia.Dashboard.Tests.Formatting;

public class DisplayFormatTests
{
    [Theory]
    [InlineData(0.8123, "81.2%")]
    [InlineData(0.0, "0.0%")]
    [InlineData(1.0, "100.0%")]
    [InlineData(null, "—")]
    [InlineData(double.NaN, "—")]
    public void Percent(double? ratio, string expected) => Assert.Equal(expected, DisplayFormat.Percent(ratio));

    [Theory]
    [InlineData(74.4, "74")]
    [InlineData(99.6, "100")]
    [InlineData(null, "—")]
    [InlineData(double.PositiveInfinity, "—")]
    public void Score(double? score, string expected) => Assert.Equal(expected, DisplayFormat.Score(score));

    [Theory]
    [InlineData("So1aNaMintAddress1111111111111111111111xyz9", "So1a…xyz9")]
    [InlineData("shortmint", "shortmint")]
    [InlineData("", "—")]
    [InlineData(null, "—")]
    public void ShortMint(string? mint, string expected) => Assert.Equal(expected, DisplayFormat.ShortMint(mint));

    [Theory]
    [InlineData(-3, "now")] // ts ahead of the dashboard's clock
    [InlineData(0.5, "now")]
    [InlineData(12, "12s")]
    [InlineData(125, "2m")]
    [InlineData(7300, "2h")]
    [InlineData(5 * 86400 + 10, "5d")]
    public void Age(double secondsAgo, string expected)
    {
        var now = new DateTimeOffset(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

        Assert.Equal(expected, DisplayFormat.Age(now.AddSeconds(-secondsAgo), now));
    }

    [Fact]
    public void Formats_are_culture_invariant()
    {
        CultureInfo previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-CA");
            Assert.Equal("81.2%", DisplayFormat.Percent(0.8123));
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }

    [Fact]
    public void Amounts_drop_trailing_zeros()
    {
        Assert.Equal("0.0000001234", DisplayFormat.Amount(0.000000123400m));
        Assert.Equal("1.5", DisplayFormat.Amount(1.50m));
        Assert.Equal("12", DisplayFormat.Amount(12.000m));
        Assert.Equal("—", DisplayFormat.Amount(null));
    }

    [Fact]
    public void Counts_and_timestamps_are_invariant()
    {
        CultureInfo previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-CA");
            Assert.Equal("1,234,567", DisplayFormat.Count(1_234_567));
            Assert.Equal("1.5", DisplayFormat.Amount(1.5m));
            Assert.Equal("2026-10-27 11:58:00",
                DisplayFormat.Timestamp(new DateTimeOffset(2026, 10, 27, 7, 58, 0, TimeSpan.FromHours(-4))));
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }
}
