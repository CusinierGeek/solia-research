using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Npgsql;
using Solia.Bot.Trading;
using Solia.Shared.Configuration;

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
    /// <see cref="IClosedTradeCounter"/>, on a data source whose connection string comes from
    /// <see cref="DbConnectionSettings"/> (SOLIA_DB_CONNECTION, else POSTGRES_*). Reuses an
    /// <see cref="NpgsqlDataSource"/> the host already registered.
    /// </summary>
    public static IServiceCollection AddPostgresTradeStore(this IServiceCollection services, IConfiguration configuration)
    {
        services.TryAddSingleton(_ => NpgsqlDataSource.Create(DbConnectionSettings.Resolve(key => configuration[key])));
        services.AddSingleton<PostgresTradeStore>();
        services.AddSingleton<ITradeStore>(sp => sp.GetRequiredService<PostgresTradeStore>());
        services.AddSingleton<IClosedTradeCounter>(sp => sp.GetRequiredService<PostgresTradeStore>());
        return services;
    }
}
