using Microsoft.Extensions.Configuration;
using Solia.Shared.Configuration;

namespace Solia.Bot.Configuration;

/// <summary>
/// The bot's flat PAPER_* keys from .env.example and the <see cref="BotOptions"/> section paths they
/// map onto; the mapping itself is <see cref="FlatAliases"/>. To support a new flat key, add a row to
/// <see cref="Aliases"/>.
/// </summary>
public static class BotFlatEnvAliases
{
    private static readonly FlatAlias[] Aliases =
    [
        new("PAPER_STARTING_BALANCE", $"{BotOptions.Section}:{nameof(BotOptions.StartingBalance)}"),
        new("PAPER_MAX_CONCURRENT_POSITIONS", $"{BotOptions.Section}:{nameof(BotOptions.MaxConcurrentPositions)}"),
    ];

    /// <inheritdoc cref="FlatAliases.AddFlatAliases"/>
    public static IConfigurationBuilder AddBotFlatEnvAliases(this IConfigurationManager configuration) =>
        configuration.AddFlatAliases(Aliases);
}
