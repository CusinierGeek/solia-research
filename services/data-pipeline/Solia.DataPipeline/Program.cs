using Npgsql;
using Solia.DataPipeline;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Health;
using Solia.DataPipeline.Hosting;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Logging;
using Solia.DataPipeline.Options;
using Solia.Shared.Configuration;

var builder = WebApplication.CreateBuilder(args);

// --- Configuration: flat .env.example keys (ENRICH_HOLDERS, ...) map onto option sections ---
builder.Configuration.AddFlatEnvAliases();

// --- Options: bound from config sections, validated at startup (fail fast on bad values) ---
builder.Services.AddOptions<PipelineOptions>()
    .Bind(builder.Configuration.GetSection(PipelineOptions.Section))
    .ValidateDataAnnotations()
    .Validate(o => o.HasConsistentWindow(), "Pipeline:WindowSeconds must be >= Pipeline:SnapshotCadenceSeconds.")
    .ValidateOnStart();
builder.Services.AddOptions<PumpApiOptions>()
    .Bind(builder.Configuration.GetSection(PumpApiOptions.Section))
    .ValidateDataAnnotations()
    .Validate(o => o.HasValidUrls(),
        "PumpApi:WsUrl must be an absolute ws(s):// URL and PumpApi:ReplayBaseUrl an absolute http(s):// URL.")
    .Validate(o => o.HasConsistentBackoff(), "PumpApi:ReconnectMaxDelaySeconds must be >= ReconnectBaseDelaySeconds.")
    .ValidateOnStart();

// --- Database: a single pooled data source, owned (and disposed) by the container. The
// connection string comes from IConfiguration (env vars or any other source) on first use.
builder.Services.AddSingleton(sp =>
{
    IConfiguration config = sp.GetRequiredService<IConfiguration>();
    return new NpgsqlDataSourceBuilder(DbConnectionSettings.Resolve(key => config[key])).Build();
});

// --- Operational log sink (writes to the `logs` table) ---
builder.Services.AddSingleton<ILogSink, PostgresLogSink>();

// --- Live stream source (pumpapi WebSocket) ---
builder.Services.AddSingleton<StreamConnectionState>();
builder.Services.AddSingleton<Func<IWebSocket>>(_ => static () => new SystemWebSocket());
builder.Services.AddSingleton<PumpApiStreamSource>();

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

app.Run();

// Exposed for WebApplicationFactory in Solia.DataPipeline.Tests.
public partial class Program;
