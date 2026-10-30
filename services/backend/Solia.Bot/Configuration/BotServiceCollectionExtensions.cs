using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Npgsql;
using Solia.Bot.Intake;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;
using Solia.Bot.Trading;
using Solia.Shared.Configuration;
using Solia.Shared.Data;

namespace Solia.Bot.Configuration;

/// <summary>
/// Wires <see cref="BotOptions"/> into the host: bound from the "Bot" section and validated at startup
/// against the rules declared on <see cref="BotOptions"/>, so bad config fails fast. The host (T6)
/// calls this after registering <see cref="BotFlatEnvAliases.AddBotFlatEnvAliases"/>.
/// </summary>
public static class BotServiceCollectionExtensions
{
    public static IServiceCollection AddBotOptions(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<BotOptions>()
            .Bind(configuration.GetSection(BotOptions.Section))
            .ValidateDataAnnotations()
            .ValidateOnStart();
        return services;
    }

    /// <summary>
    /// Registers <see cref="PostgresTradeStore"/> as the <see cref="ITradeStore"/> and
    /// <see cref="IClosedTradeCounter"/> (recording each trade's <see cref="BotSessions">bot session</see> when
    /// one is registered, as <see cref="AddPaperTradingEngine"/> does), on a data source whose connection string comes from
    /// <see cref="DbConnectionSettings"/> (SOLIA_DB_CONNECTION, else POSTGRES_*). Reuses an
    /// <see cref="NpgsqlDataSource"/> the host already registered.
    /// </summary>
    public static IServiceCollection AddPostgresTradeStore(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddSoliaDataSource(configuration);
        services.AddSingleton<PostgresTradeStore>();
        services.AddSingleton<ITradeStore>(sp => sp.GetRequiredService<PostgresTradeStore>());
        services.AddSingleton<IClosedTradeCounter>(sp => sp.GetRequiredService<PostgresTradeStore>());
        return services;
    }

    /// <summary>
    /// Wires the whole paper-trading bot into the host: <see cref="AddPaperTradingEngine"/> plus the
    /// <see cref="PaperTradingBot"/> hosted service, so the bot runs for the host's whole lifetime. A host
    /// that starts the bot itself (e.g. behind an on/off switch) calls <see cref="AddPaperTradingEngine"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">No <see cref="IBotEventSource"/> is registered.</exception>
    public static IServiceCollection AddPaperTradingBot(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddPaperTradingEngine(configuration);
        services.AddHostedService<PaperTradingBot>();
        return services;
    }

    /// <summary>
    /// Wires everything the paper-trading bot runs on, without hosting it: validated <see cref="BotOptions"/>,
    /// the Postgres trade store, the bot sessions its trades are recorded in (<see cref="BotSessions"/>), the
    /// Postgres operational log sink (service = 'bot') and the <see cref="PositionManager"/> engine.
    /// <para>The caller <strong>must</strong> register an <see cref="IBotEventSource"/> first (the live/DB
    /// feed in Sprint 4, an <see cref="InMemoryBotEventSource"/> for a replay) — there is deliberately no
    /// default, since an empty feed would leave the host up but trading nothing; a missing source throws
    /// here, at startup. The log sink and <see cref="TimeProvider"/> are registered with <c>TryAdd</c>, so a
    /// caller can swap them first (e.g. <see cref="NullBotLogSink"/> for a replay that must not write to
    /// `logs`). Callers add <see cref="BotFlatEnvAliases.AddBotFlatEnvAliases"/> to the configuration first.</para>
    /// </summary>
    /// <exception cref="InvalidOperationException">No <see cref="IBotEventSource"/> is registered.</exception>
    public static IServiceCollection AddPaperTradingEngine(this IServiceCollection services, IConfiguration configuration)
    {
        if (!services.Any(d => d.ServiceType == typeof(IBotEventSource)))
        {
            throw new InvalidOperationException(
                $"Register an {nameof(IBotEventSource)} before wiring the paper-trading bot: " +
                "the bot has no default feed, so it can't start trading nothing.");
        }

        services.AddBotOptions(configuration);
        services.AddPostgresTradeStore(configuration);
        services.TryAddSingleton<IBotLogSink, PostgresBotLogSink>();
        services.TryAddSingleton(TimeProvider.System);
        services.TryAddSingleton<IBotSessionStore, PostgresBotSessionStore>();
        services.TryAddSingleton<BotSessions>();
        services.AddSingleton(sp => new PositionManager(
            sp.GetRequiredService<IOptions<BotOptions>>().Value,
            sp.GetRequiredService<ITradeStore>(),
            sp.GetRequiredService<IBotLogSink>(),
            sp.GetService<ILogger<PositionManager>>()));
        return services;
    }
}
