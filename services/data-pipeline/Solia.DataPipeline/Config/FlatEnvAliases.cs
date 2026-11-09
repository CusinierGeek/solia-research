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
        new("RETENTION_ENABLED", "Retention:Enabled", IsBool: true),
        new("RETENTION_INTERVAL_MINUTES", "Retention:IntervalMinutes"),
        new("RETENTION_FEATURES_TTL_DAYS", "Retention:FeaturesTtlDays"),
        new("RETENTION_LOGS_TTL_DAYS", "Retention:LogsTtlDays"),
        new("RETENTION_PREDICTIONS_TTL_DAYS", "Retention:PredictionsTtlDays"),
        new("RETENTION_BATCH_SIZE", "Retention:BatchSize"),
        new("RETENTION_ESTABLISHED_AGE_HOURS", "Retention:EstablishedAgeHours"),
        new("RETENTION_ESTABLISHED_TTL_DAYS", "Retention:EstablishedTtlDays"),
        new("RETENTION_ESTABLISHED_FULL_RESOLUTION_MINUTES", "Retention:EstablishedFullResolutionMinutes"),
        new("RETENTION_ESTABLISHED_SAMPLE_SECONDS", "Retention:EstablishedSampleSeconds"),
    ];

    /// <inheritdoc cref="FlatAliases.AddFlatAliases"/>
    public static IConfigurationBuilder AddFlatEnvAliases(this IConfigurationManager configuration) =>
        configuration.AddFlatAliases(Aliases);
}
