using Solia.Bot.Domain;

namespace Solia.Bot.Tests;

public class MoneyTests
{
    [Theory]
    [InlineData("1.2345678919", "1.234567891")]
    [InlineData("-1.2345678919", "-1.234567891")] // toward zero, both signs
    [InlineData("2", "2")]
    public void Truncate_drops_digits_past_lamport_precision(string amount, string expected)
    {
        Assert.Equal(Parse(expected), Money.Truncate(Parse(amount)));
    }

    [Theory]
    [InlineData("1.2345678916", "1.234567892")]
    [InlineData("0.0000000005", "0")]            // banker's: half to even
    [InlineData("0.0000000015", "0.000000002")]
    [InlineData("-0.000000000000000000000001", "0")] // a rounding residue vanishes
    public void Round_rounds_to_lamport_precision(string amount, string expected)
    {
        Assert.Equal(Parse(expected), Money.Round(Parse(amount)));
    }

    private static decimal Parse(string s) => decimal.Parse(s, System.Globalization.CultureInfo.InvariantCulture);
}
