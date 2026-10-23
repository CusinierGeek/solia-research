using Microsoft.Extensions.Configuration;
using Solia.Core.Configuration;

namespace Solia.Core.Tests;

public class ApiFlatEnvAliasesTests
{
    private static ConfigurationManager Config(params (string Key, string Value)[] values)
    {
        var config = new ConfigurationManager();
        config.AddInMemoryCollection(values.Select(v => new KeyValuePair<string, string?>(v.Key, v.Value)));
        return config;
    }

    private static ApiOptions Bound(IConfiguration config)
    {
        ApiOptions options = config.GetSection(ApiOptions.Section).Get<ApiOptions>() ?? new ApiOptions();
        ApiFlatEnvAliases.Apply(config, options);
        return options;
    }

    [Fact]
    public void Backend_url_flat_key_maps_onto_api_options()
    {
        ConfigurationManager config = Config(("BACKEND_URL", "https://api.solia.example"));

        Assert.Equal("https://api.solia.example", Bound(config).BaseUrl);
    }

    [Fact]
    public void Flat_key_overrides_section_value_and_is_trimmed()
    {
        ConfigurationManager config = Config(
            ("Api:BaseUrl", "http://from-appsettings:8080"),
            ("BACKEND_URL", "http://from-env:9090/\r"));

        Assert.Equal("http://from-env:9090/", Bound(config).BaseUrl);
    }

    [Fact]
    public void Unset_flat_key_leaves_section_value_alone()
    {
        ConfigurationManager config = Config(("Api:BaseUrl", "http://from-appsettings:8080"));

        Assert.Equal("http://from-appsettings:8080", Bound(config).BaseUrl);
        Assert.Empty(ApiFlatEnvAliases.Resolve(config));
    }

    [Fact]
    public void Flat_key_from_a_source_added_later_is_still_honoured()
    {
        ConfigurationManager config = Config(("Api:BaseUrl", "http://from-appsettings:8080"));

        config.AddInMemoryCollection([new KeyValuePair<string, string?>("BACKEND_URL", "http://added-later:7070")]);

        Assert.Equal("http://added-later:7070", Bound(config).BaseUrl);
    }
}
