using Solia.Shared.Configuration;

namespace Solia.DataPipeline.Config;

/// <summary>
/// The data-pipeline's flat keys from .env.example (e.g. ENRICH_HOLDERS) and the options-section
/// paths they map onto (Pipeline:EnrichHolders); the mapping itself is <see cref="FlatAliases"/>.
/// To support a new flat key, add a row to <see cref="Aliases"/>.
/// </summary>
public static class FlatEnvAliases
{
    private static readonly FlatAlias[] Aliases =
    [
        new("PUMPAPI_WS_URL", "PumpApi:WsUrl"),
        new("PUMPAPI_REPLAY_BASE_URL", "PumpApi:ReplayBaseUrl"),
        new("ENRICH_HOLDERS", "Pipeline:EnrichHolders", IsBool: true),
        new("STREAM_PROBE", "Pipeline:StreamProbe", IsBool: true),
        new("INGEST", "Pipeline:Ingest", IsBool: true),
    ];

    /// <inheritdoc cref="FlatAliases.AddFlatAliases"/>
    public static IConfigurationBuilder AddFlatEnvAliases(this IConfigurationManager configuration) =>
        configuration.AddFlatAliases(Aliases);
}
