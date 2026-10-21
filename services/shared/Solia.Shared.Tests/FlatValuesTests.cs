using Solia.Shared.Configuration;

namespace Solia.Shared.Tests;

public class FlatValuesTests
{
    private static Func<string, string?> Lookup(string? value) => _ => value;

    [Theory]
    [InlineData("true", true)]
    [InlineData("TRUE", true)]
    [InlineData("1", true)]
    [InlineData("yes", true)]
    [InlineData("on", true)]
    [InlineData("false", false)]
    [InlineData("0", false)]
    [InlineData("no", false)]
    [InlineData("Off\r", false)]
    public void GetBool_accepts_common_spellings(string raw, bool expected)
    {
        Assert.Equal(expected, FlatValues.GetBool(Lookup(raw), "ENRICH_HOLDERS"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public void GetBool_returns_null_when_unset(string? raw)
    {
        Assert.Null(FlatValues.GetBool(Lookup(raw), "ENRICH_HOLDERS"));
    }

    [Fact]
    public void GetBool_rejects_unknown_values_naming_the_key()
    {
        var ex = Assert.Throws<InvalidOperationException>(
            () => FlatValues.GetBool(Lookup("maybe"), "ENRICH_HOLDERS"));

        Assert.Contains("ENRICH_HOLDERS", ex.Message);
    }
}
