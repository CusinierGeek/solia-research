using Microsoft.Extensions.Configuration;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests;

public class FlatEnvAliasesTests
{
    private static ConfigurationManager Config(params (string Key, string Value)[] values)
    {
        var config = new ConfigurationManager();
        config.AddInMemoryCollection(values.Select(v => new KeyValuePair<string, string?>(v.Key, v.Value)));
        return config;
    }

    [Fact]
    public void Flat_keys_override_section_values()
    {
        ConfigurationManager config = Config(
            ("Pipeline:EnrichHolders", "false"),
            ("PumpApi:WsUrl", "wss://from-appsettings/"),
            ("ENRICH_HOLDERS", "1"),
            ("PUMPAPI_WS_URL", "wss://from-env/\r"),
            ("PUMPAPI_REPLAY_BASE_URL", "https://replay.example"));

        config.AddFlatEnvAliases();

        Assert.True(config.GetSection(PipelineOptions.Section).Get<PipelineOptions>()!.EnrichHolders);
        PumpApiOptions pump = config.GetSection(PumpApiOptions.Section).Get<PumpApiOptions>()!;
        Assert.Equal("wss://from-env/", pump.WsUrl);
        Assert.Equal("https://replay.example", pump.ReplayBaseUrl);
    }

    [Fact]
    public void Unset_flat_keys_leave_section_values_alone()
    {
        ConfigurationManager config = Config(("Pipeline:EnrichHolders", "true"));

        config.AddFlatEnvAliases();

        Assert.True(config.GetSection(PipelineOptions.Section).Get<PipelineOptions>()!.EnrichHolders);
    }

    [Fact]
    public void Invalid_flat_bool_fails_naming_the_key()
    {
        ConfigurationManager config = Config(("ENRICH_HOLDERS", "maybe"));

        var ex = Assert.Throws<InvalidOperationException>(() => config.AddFlatEnvAliases());
        Assert.Contains("ENRICH_HOLDERS", ex.Message);
    }
}
