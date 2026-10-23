using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Solia.Bot.Configuration;

namespace Solia.Bot.Tests;

public class BotConfigurationTests
{
    private static ConfigurationManager Config(params (string Key, string Value)[] values)
    {
        var config = new ConfigurationManager();
        config.AddInMemoryCollection(values.Select(v => new KeyValuePair<string, string?>(v.Key, v.Value)));
        return config;
    }

    [Fact]
    public void Flat_paper_keys_map_onto_bot_options()
    {
        ConfigurationManager config = Config(
            ("Bot:StartingBalance", "10"),
            ("Bot:MaxConcurrentPositions", "5"),
            ("PAPER_STARTING_BALANCE", "25.5"),
            ("PAPER_MAX_CONCURRENT_POSITIONS", "8\r")); // trailing CR from a Windows-edited .env

        config.AddBotFlatEnvAliases();

        BotOptions bot = config.GetSection(BotOptions.Section).Get<BotOptions>()!;
        Assert.Equal(25.5m, bot.StartingBalance);
        Assert.Equal(8, bot.MaxConcurrentPositions);
    }

    [Theory]
    [InlineData("yes", "True")]
    [InlineData("0", "False")]
    public void Flat_bot_enabled_key_maps_onto_the_host_switch(string flat, string expected)
    {
        ConfigurationManager config = Config(("Bot:Enabled", "False"), ("PAPER_BOT_ENABLED", flat));

        config.AddBotFlatEnvAliases();

        Assert.Equal(expected, config[BotFlatEnvAliases.EnabledPath]);
    }

    [Fact]
    public void An_invalid_flat_bot_enabled_key_fails_naming_it()
    {
        ConfigurationManager config = Config(("PAPER_BOT_ENABLED", "maybe"));

        var error = Assert.ThrowsAny<Exception>(() => config.AddBotFlatEnvAliases());
        Assert.Contains("PAPER_BOT_ENABLED", error.Message);
    }

    [Fact]
    public void Unset_flat_keys_leave_section_values_alone()
    {
        ConfigurationManager config = Config(("Bot:StartingBalance", "42"));

        config.AddBotFlatEnvAliases();

        Assert.Equal(42m, config.GetSection(BotOptions.Section).Get<BotOptions>()!.StartingBalance);
    }

    private static BotOptions Resolve(params (string Key, string Value)[] values)
    {
        ConfigurationManager config = Config(values);
        var services = new ServiceCollection();
        services.AddBotOptions(config);
        using ServiceProvider provider = services.BuildServiceProvider();
        return provider.GetRequiredService<IOptions<BotOptions>>().Value;
    }

    [Fact]
    public void Add_bot_options_binds_and_accepts_valid_config()
    {
        BotOptions bot = Resolve(
            ("Bot:StartingBalance", "50"),
            ("Bot:MaxConcurrentPositions", "3"),
            ("Bot:MinPumpProb", "0.7"),
            ("Bot:MaxRugProb", "0.15"));

        Assert.Equal(50m, bot.StartingBalance);
        Assert.Equal(3, bot.MaxConcurrentPositions);
        Assert.Equal(0.7, bot.MinPumpProb);
    }

    [Fact]
    public void Add_bot_options_accepts_empty_config_via_defaults()
    {
        BotOptions bot = Resolve();

        Assert.Equal(new BotOptions().StartingBalance, bot.StartingBalance);
    }

    [Theory]
    [InlineData("Bot:MaxConcurrentPositions", "0")]          // out of range
    [InlineData("Bot:StartingBalance", "-1")]                // non-positive balance
    [InlineData("Bot:MaxPositionFraction", "0")]             // zero sizing cap
    [InlineData("Bot:StopLossPct", "1.5")]                   // stop below a zero price
    [InlineData("Bot:StartingBalance", "100000000000")]      // absurd balance (typo)
    public void Add_bot_options_rejects_invalid_config_at_resolution(string key, string value)
    {
        Assert.Throws<OptionsValidationException>(() => Resolve((key, value)));
    }

    [Fact]
    public void Add_bot_options_accepts_independent_entry_gates()
    {
        BotOptions bot = Resolve(("Bot:MinPumpProb", "0.3"), ("Bot:MaxRugProb", "0.35"));

        Assert.Equal(0.35, bot.MaxRugProb);
    }

    private static IHost BuildHost(params (string Key, string Value)[] values)
    {
        HostApplicationBuilder builder = Host.CreateEmptyApplicationBuilder(new HostApplicationBuilderSettings());
        builder.Configuration.AddInMemoryCollection(
            values.Select(v => new KeyValuePair<string, string?>(v.Key, v.Value)));
        builder.Configuration.AddBotFlatEnvAliases();
        builder.Services.AddBotOptions(builder.Configuration);
        return builder.Build();
    }

    [Fact]
    public async Task Host_fails_to_start_on_invalid_config()
    {
        using IHost host = BuildHost(("PAPER_STARTING_BALANCE", "0"));

        var ex = await Assert.ThrowsAsync<OptionsValidationException>(() => host.StartAsync());
        Assert.Contains(nameof(BotOptions.StartingBalance), ex.Message);
    }

    [Fact]
    public async Task Host_starts_on_valid_config()
    {
        using IHost host = BuildHost(("PAPER_STARTING_BALANCE", "25"));

        await host.StartAsync();
        await host.StopAsync();
    }
}
