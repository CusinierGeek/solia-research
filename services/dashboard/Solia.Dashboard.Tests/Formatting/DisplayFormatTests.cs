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
    public void Gains_and_losses_are_signed()
    {
        Assert.Equal("+0.0123", DisplayFormat.SignedAmount(0.012300m));
        Assert.Equal("-0.0000000045", DisplayFormat.SignedAmount(-0.0000000045m));
        Assert.Equal("0", DisplayFormat.SignedAmount(0m));
        Assert.Equal("—", DisplayFormat.SignedAmount(null));
    }

    [Theory]
    [InlineData(0.25, "+25.0%")]
    [InlineData(-0.034, "-3.4%")]
    [InlineData(0.0, "0.0%")]
    [InlineData(null, "—")]
    [InlineData(double.NaN, "—")]
    public void SignedPercent(double? ratio, string expected) => Assert.Equal(expected, DisplayFormat.SignedPercent(ratio));

    [Theory]
    [InlineData(0, "0s")]
    [InlineData(45, "45s")]
    [InlineData(187, "3m 07s")]
    [InlineData(3900, "1h 05m")]
    [InlineData(2 * 86400 + 3 * 3600 + 59, "2d 03h")]
    [InlineData(null, "—")]
    [InlineData(-1, "—")]
    public void Duration(int? seconds, string expected) => Assert.Equal(expected, DisplayFormat.Duration(seconds));

    [Fact]
    public void A_missing_timestamp_is_the_placeholder() => Assert.Equal("—", DisplayFormat.Timestamp((DateTimeOffset?)null));

    [Fact]
    public void Formats_are_culture_invariant()
    {
        CultureInfo previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("fr-CA");
            Assert.Equal("81.2%", DisplayFormat.Percent(0.8123));
            Assert.Equal("-3.4%", DisplayFormat.SignedPercent(-0.034));
            Assert.Equal("+1.5", DisplayFormat.SignedAmount(1.5m));
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
