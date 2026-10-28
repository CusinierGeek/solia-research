using Solia.DataPipeline;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Health;
using Solia.DataPipeline.Hosting;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.Shared.Logging;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using Solia.Shared.Data;
using Solia.Shared.Hosting;

// --- Container healthcheck: `--healthcheck` probes /health and exits without starting the host ---
if (HealthProbe.IsRequested(args))
{
    return await HealthProbe.RunAsync(args);
}

var builder = WebApplication.CreateBuilder(args);

// --- Configuration: flat .env.example keys (ENRICH_HOLDERS, ...) map onto option sections ---
builder.Configuration.AddFlatEnvAliases();

// --- Options: bound from config sections, validated at startup (fail fast on bad values) ---
builder.Services.AddOptions<PipelineOptions>()
    .Bind(builder.Configuration.GetSection(PipelineOptions.Section))
    .ValidateDataAnnotations()
    .Validate(o => o.HasConsistentWindow(), "Pipeline:WindowSeconds must be >= Pipeline:SnapshotCadenceSeconds.")
    .Validate(o => o.HasConsistentMomentumWindows(),
        "Pipeline:MomentumShortSeconds < MomentumMediumSeconds < MomentumLongSeconds <= WindowSeconds, and 2 * MomentumShortSeconds <= WindowSeconds.")
    .Validate(o => o.HasExclusiveLiveConsumer(),
        "Pipeline:Ingest and Pipeline:StreamProbe (INGEST / STREAM_PROBE) can't both be enabled: each owns the single live connection.")
    .ValidateOnStart();
builder.Services.AddOptions<PumpApiOptions>()
    .Bind(builder.Configuration.GetSection(PumpApiOptions.Section))
    .ValidateDataAnnotations()
    .Validate(o => o.HasValidUrls(),
        "PumpApi:WsUrl must be an absolute ws(s):// URL and PumpApi:ReplayBaseUrl an absolute http(s):// URL.")
    .Validate(o => o.HasConsistentBackoff(), "PumpApi:ReconnectMaxDelaySeconds must be >= ReconnectBaseDelaySeconds.")
    .ValidateOnStart();

// --- Database: a single pooled data source, owned (and disposed) by the container; the
// connection string is resolved on first use. Plus the operational log sink (`logs` table).
builder.Services.AddSoliaDataSource(builder.Configuration);
builder.Services.AddPostgresLogSink(builder.Configuration, "data-pipeline");

// --- Live stream source (pumpapi WebSocket) ---
builder.Services.AddSingleton<StreamConnectionState>();
builder.Services.AddSingleton<Func<IWebSocket>>(_ => static () => new SystemWebSocket());
builder.Services.AddSingleton<PumpApiStreamSource>();

// --- Ingestion engine (source → rolling features → PostgreSQL) ---
builder.Services.AddSingleton<IDataSource>(sp => sp.GetRequiredService<PumpApiStreamSource>());
builder.Services.AddSingleton<TokenWindowStore>();
builder.Services.AddSingleton<ITokenSink, PostgresTokenSink>();
builder.Services.AddSingleton<PipelineEngine>();

// --- Health (bounded so probes don't hang on an unreachable database) ---
builder.Services.AddHealthChecks()
    .AddCheck<DatabaseHealthCheck>("database", timeout: TimeSpan.FromSeconds(3))
    .AddCheck<StreamHealthCheck>("stream");

// --- Pipeline worker + opt-in live-stream probe ---
builder.Services.AddHostedService<Worker>();
builder.Services.AddHostedService<StreamProbeService>();

var app = builder.Build();

app.MapHealthChecks("/health");
app.MapGet("/", () => "Solia data-pipeline");

await app.RunAsync();
return 0;

// Exposed for WebApplicationFactory in Solia.DataPipeline.Tests.
public partial class Program;
