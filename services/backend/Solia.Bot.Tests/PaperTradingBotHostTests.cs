using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Solia.Bot;
using Solia.Bot.Configuration;
using Solia.Bot.Intake;
using Solia.Bot.Logging;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests;

/// <summary>
/// Host wiring: bad <see cref="BotOptions"/> fail at startup (not on first use), and
/// <see cref="BotServiceCollectionExtensions.AddPaperTradingBot"/> composes the whole bot — engine,
/// store, sink, event source and hosted service.
/// </summary>
public class PaperTradingBotHostTests
{
    private const string ValidDb = "Host=db;Database=solia;Username=u;Password=p";

    private static IConfiguration Config(Dictionary<string, string?> values) =>
        new ConfigurationBuilder().AddInMemoryCollection(values).Build();

    private static IHost HostWithOptions(Dictionary<string, string?> values)
    {
        HostApplicationBuilder builder = Host.CreateApplicationBuilder();
        builder.Configuration.AddInMemoryCollection(values);
        builder.Services.AddBotOptions(builder.Configuration);
        return builder.Build();
    }

    [Fact]
    public async Task Invalid_options_fail_at_startup()
    {
        using IHost host = HostWithOptions(new() { ["Bot:StartingBalance"] = "0" }); // below the exclusive minimum

        await Assert.ThrowsAsync<OptionsValidationException>(() => host.StartAsync());
    }

    [Fact]
    public async Task Valid_options_start_and_stop_cleanly()
    {
        using IHost host = HostWithOptions(new() { ["Bot:StartingBalance"] = "25" });

        await host.StartAsync();
        await host.StopAsync();
    }

    private static IServiceCollection WithSource(IBotEventSource? source = null)
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddSingleton(source ?? new InMemoryBotEventSource([])); // Sprint 4 registers the live/DB feed first
        return services;
    }

    [Fact]
    public void AddPaperTradingBot_wires_the_engine_store_sink_source_and_hosted_service()
    {
        var source = new InMemoryBotEventSource([]);
        IServiceCollection services = WithSource(source);
        services.AddPaperTradingBot(Config(new() { ["SOLIA_DB_CONNECTION"] = ValidDb }));

        using ServiceProvider provider = services.BuildServiceProvider();

        Assert.IsType<PostgresBotLogSink>(provider.GetRequiredService<IBotLogSink>());
        Assert.IsType<PostgresTradeStore>(provider.GetRequiredService<ITradeStore>());
        Assert.NotNull(provider.GetRequiredService<PositionManager>());
        Assert.Same(source, provider.GetRequiredService<IBotEventSource>());
        Assert.Same(TimeProvider.System, provider.GetRequiredService<TimeProvider>());
        Assert.IsType<PaperTradingBot>(Assert.Single(provider.GetServices<IHostedService>()));
    }

    [Fact]
    public void Without_a_registered_event_source_wiring_fails_fast_instead_of_trading_nothing()
    {
        var services = new ServiceCollection();

        var error = Assert.Throws<InvalidOperationException>(() =>
            services.AddPaperTradingBot(Config(new() { ["SOLIA_DB_CONNECTION"] = ValidDb })));

        Assert.Contains(nameof(IBotEventSource), error.Message);
    }

    [Fact]
    public void A_caller_registered_log_sink_wins_so_a_replay_need_not_write_to_logs()
    {
        IServiceCollection services = WithSource();
        services.AddSingleton<IBotLogSink>(NullBotLogSink.Instance); // e.g. the T6 replay harness
        services.AddPaperTradingBot(Config(new() { ["SOLIA_DB_CONNECTION"] = ValidDb }));

        using ServiceProvider provider = services.BuildServiceProvider();

        Assert.Same(NullBotLogSink.Instance, provider.GetRequiredService<IBotLogSink>());
    }
}
