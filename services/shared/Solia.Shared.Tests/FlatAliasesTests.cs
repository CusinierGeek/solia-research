using Microsoft.Extensions.Configuration;
using Solia.Shared.Configuration;

namespace Solia.Shared.Tests;

public class FlatAliasesTests
{
    private static readonly FlatAlias[] Aliases =
    [
        new("SOME_URL", "Section:Url"),
        new("SOME_FLAG", "Section:Flag", IsBool: true),
    ];

    private static ConfigurationManager Config(params (string Key, string Value)[] values)
    {
        var config = new ConfigurationManager();
        config.AddInMemoryCollection(values.Select(v => new KeyValuePair<string, string?>(v.Key, v.Value)));
        return config;
    }

    [Fact]
    public void Flat_keys_override_section_values_and_are_trimmed()
    {
        ConfigurationManager config = Config(
            ("Section:Url", "from-appsettings"),
            ("SOME_URL", "from-env\r"));

        config.AddFlatAliases(Aliases);

        Assert.Equal("from-env", config["Section:Url"]);
    }

    [Theory]
    [InlineData("yes", "True")]
    [InlineData("0", "False")]
    [InlineData(" ON ", "True")]
    public void Flat_bools_are_normalised(string raw, string expected)
    {
        ConfigurationManager config = Config(("SOME_FLAG", raw));

        config.AddFlatAliases(Aliases);

        Assert.Equal(expected, config["Section:Flag"]);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Unset_or_blank_flat_keys_leave_section_values_alone(string raw)
    {
        ConfigurationManager config = Config(("Section:Url", "kept"), ("SOME_URL", raw));

        config.AddFlatAliases(Aliases);

        Assert.Equal("kept", config["Section:Url"]);
    }

    [Fact]
    public void Invalid_flat_bool_fails_naming_the_key()
    {
        ConfigurationManager config = Config(("SOME_FLAG", "maybe"));

        var ex = Assert.Throws<InvalidOperationException>(() => config.AddFlatAliases(Aliases));
        Assert.Contains("SOME_FLAG", ex.Message);
    }
}
