using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

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
}
