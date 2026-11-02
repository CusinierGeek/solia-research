using Solia.DataPipeline.Metadata;
using Solia.DataPipeline.Retention;

namespace Solia.DataPipeline.Options;

/// <summary>
/// Binds and validates the pipeline's option sections. Shared by the live host and the one-shot
/// replay / retention commands, so both fail fast on the same bad values.
/// </summary>
public static class PipelineOptionsRegistration
{
    public static IServiceCollection AddPipelineOptions(this IServiceCollection services, IConfiguration configuration)
    {
        services.AddOptions<PipelineOptions>()
            .Bind(configuration.GetSection(PipelineOptions.Section))
            .ValidateDataAnnotations()
            .Validate(o => o.HasConsistentWindow(), "Pipeline:WindowSeconds must be >= Pipeline:SnapshotCadenceSeconds.")
            .Validate(o => o.HasConsistentMomentumWindows(),
                "Pipeline:MomentumShortSeconds < MomentumMediumSeconds < MomentumLongSeconds <= WindowSeconds, and 2 * MomentumShortSeconds <= WindowSeconds.")
            .Validate(o => o.HasConsistentWalletHorizon(), "Pipeline:WalletAgeCapHours must be <= Pipeline:WalletTtlHours.")
            .Validate(o => o.HasConsistentReputationHorizon(),
                "Pipeline:ReputationPositionMaxHours and ReputationOutcomeHorizonSeconds + ReputationOutcomeEmbargoSeconds must be shorter than Pipeline:WalletAgeCapHours.")
            .Validate(o => o.HasExclusiveLiveConsumer(),
                "Pipeline:Ingest and Pipeline:StreamProbe (INGEST / STREAM_PROBE) can't both be enabled: each owns the single live connection.")
            .ValidateOnStart();
        services.AddOptions<PumpApiOptions>()
            .Bind(configuration.GetSection(PumpApiOptions.Section))
            .ValidateDataAnnotations()
            .Validate(o => o.HasValidUrls(),
                "PumpApi:WsUrl must be an absolute ws(s):// URL and PumpApi:ReplayBaseUrl an absolute http(s):// URL.")
            .Validate(o => o.HasConsistentBackoff(), "PumpApi:ReconnectMaxDelaySeconds must be >= ReconnectBaseDelaySeconds.")
            .ValidateOnStart();
        services.AddOptions<MetadataOptions>()
            .Bind(configuration.GetSection(MetadataOptions.Section))
            .ValidateDataAnnotations()
            .Validate(o => o.HasValidGateways(), "Metadata:IpfsGateway and Metadata:ArweaveGateway must be absolute https:// URLs.")
            .ValidateOnStart();
        services.AddOptions<RetentionOptions>()
            .Bind(configuration.GetSection(RetentionOptions.Section))
            .ValidateDataAnnotations()
            .ValidateOnStart();
        return services;
    }
}
