using Solia.Bot;
using Solia.Bot.Configuration;
using Solia.Bot.Intake;

namespace Solia.Api.Bot;

/// <summary>
/// Hosts the Sprint-3 paper-trading bot in the backend: its validated options, the live
/// <see cref="DbBotEventSource"/> feed, the Postgres trade store / closed-trade counter / `logs` sink
/// (<see cref="BotServiceCollectionExtensions.AddPaperTradingEngine"/>) and <see cref="GatedPaperTradingBot"/>,
/// which runs it only when <see cref="BotHostOptions.Enabled"/>. The feed also needs the EF context and the
/// ai-service client, which <c>Program.cs</c> registers. Callers add
/// <see cref="BotFlatEnvAliases.AddBotFlatEnvAliases"/> to the configuration first.
/// </summary>
public static class BotHostServiceCollectionExtensions
{
    public static IServiceCollection AddHostedPaperTradingBot(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<BotHostOptions>()
            .Bind(configuration.GetSection(BotHostOptions.Section))
            .ValidateDataAnnotations()
            .Validate(o => o.HasKnownHorizon(),
                $"Bot:Horizon must be one of {string.Join(", ", BotHostOptions.Horizons)} (the predictions table's horizons).")
            .ValidateOnStart();

        services.AddSingleton<PostgresPredictionStore>();
        services.AddSingleton<IBotEventSource, DbBotEventSource>();
        services.AddPaperTradingEngine(configuration);
        services.AddSingleton<PaperTradingBot>();
        services.AddHostedService<GatedPaperTradingBot>();
        return services;
    }
}
