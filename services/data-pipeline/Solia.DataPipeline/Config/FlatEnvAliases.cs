using Solia.Shared.Configuration;

namespace Solia.DataPipeline.Config;

/// <summary>
/// Maps the flat keys from .env.example (e.g. ENRICH_HOLDERS) onto their options-section
/// paths (Pipeline:EnrichHolders), so options are bound by the normal configuration binder.
/// To support a new flat key, add a row to <see cref="Aliases"/>.
/// </summary>
public static class FlatEnvAliases
{
    private static readonly (string FlatKey, string Path, bool IsBool)[] Aliases =
    [
        ("PUMPAPI_WS_URL", "PumpApi:WsUrl", false),
        ("PUMPAPI_REPLAY_BASE_URL", "PumpApi:ReplayBaseUrl", false),
        ("ENRICH_HOLDERS", "Pipeline:EnrichHolders", true),
        ("STREAM_PROBE", "Pipeline:StreamProbe", true),
        ("INGEST", "Pipeline:Ingest", true),
    ];

    /// <summary>
    /// Adds the aliased values as a configuration source on top of the sources registered so
    /// far, so a flat key wins over appsettings. Throws if a flat bool is set but invalid.
    /// </summary>
    public static IConfigurationBuilder AddFlatEnvAliases(this IConfigurationManager configuration)
    {
        var values = new Dictionary<string, string?>();
        foreach ((string flatKey, string path, bool isBool) in Aliases)
        {
            string? raw = configuration[flatKey];
            if (string.IsNullOrWhiteSpace(raw))
            {
                continue;
            }
            values[path] = isBool
                ? FlatValues.GetBool(key => configuration[key], flatKey)!.Value.ToString()
                : raw.Trim();
        }
        return configuration.AddInMemoryCollection(values);
    }
}
