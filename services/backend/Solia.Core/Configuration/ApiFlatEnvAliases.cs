using Microsoft.Extensions.Configuration;
using Solia.Shared.Configuration;

namespace Solia.Core.Configuration;

/// <summary>
/// The backend API's flat keys from .env.example (e.g. BACKEND_URL) and the options-section paths they
/// map onto (Api:BaseUrl, AiService:BaseUrl); the mapping itself is <see cref="FlatAliases"/>. To support a new flat key,
/// add a row to <see cref="Aliases"/>. Unlike a one-shot configuration source, <see cref="Resolve"/> is
/// meant to run when <see cref="ApiOptions"/> is resolved, so a flat key supplied by any configuration
/// source — including ones added after startup code ran (e.g. a test host's) — is honoured.
/// </summary>
public static class ApiFlatEnvAliases
{
    private static readonly FlatAlias[] Aliases =
    [
        new("BACKEND_URL", $"{ApiOptions.Section}:{nameof(ApiOptions.BaseUrl)}"),
        new("AI_SERVICE_URL", $"{AiServiceOptions.Section}:{nameof(AiServiceOptions.BaseUrl)}"),
    ];

    /// <summary>The set flat keys in <paramref name="configuration"/>, keyed by the option path they override.</summary>
    public static Dictionary<string, string?> Resolve(IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        return FlatAliases.Resolve(key => configuration[key], Aliases);
    }

    /// <summary>
    /// Overlays the set flat keys onto <paramref name="options"/> (a flat key wins over its section value).
    /// </summary>
    public static void Apply(IConfiguration configuration, ApiOptions options) =>
        Apply(configuration, options, ApiOptions.Section);

    /// <inheritdoc cref="Apply(IConfiguration, ApiOptions)"/>
    public static void Apply(IConfiguration configuration, AiServiceOptions options) =>
        Apply(configuration, options, AiServiceOptions.Section);

    // One typed overload per options class keeps the section tied to its type (no wrong-section binding).
    private static void Apply(IConfiguration configuration, object options, string section)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(section);
        new ConfigurationBuilder()
            .AddInMemoryCollection(Resolve(configuration))
            .Build()
            .GetSection(section)
            .Bind(options);
    }
}
