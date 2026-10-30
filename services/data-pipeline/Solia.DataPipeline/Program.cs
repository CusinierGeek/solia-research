using Solia.DataPipeline;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Health;
using Solia.DataPipeline.Hosting;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.Shared.Logging;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using Solia.DataPipeline.Retention;
using Solia.Shared.Data;
using Solia.Shared.Hosting;

// --- Container healthcheck: `--healthcheck` probes /health and exits without starting the host ---
if (HealthProbe.IsRequested(args))
{
    return await HealthProbe.RunAsync(args);
}

// --- One-shot historical replay: `replay --from <utc> --to <utc> [...]` runs and exits (no web host) ---
if (ReplayCommand.IsRequested(args))
{
    return await ReplayCommand.RunAsync(args[1..]);
}

// --- One-shot purge: `retention [--dry-run]` runs the retention job once and exits (no web host) ---
if (RetentionCommand.IsRequested(args))
{
    return await RetentionCommand.RunAsync(args[1..]);
}

var builder = WebApplication.CreateBuilder(args);

// --- Configuration: flat .env.example keys (ENRICH_HOLDERS, ...) map onto option sections ---
builder.Configuration.AddFlatEnvAliases();

// --- Options: bound from config sections, validated at startup (fail fast on bad values) ---
builder.Services.AddPipelineOptions(builder.Configuration);

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

// --- Data retention: periodic purge of old features / logs / predictions (Retention:Enabled / RETENTION_ENABLED) ---
builder.Services.AddRetention(hosted: true);

var app = builder.Build();

app.MapHealthChecks("/health");
app.MapGet("/", () => "Solia data-pipeline");

await app.RunAsync();
return 0;

// Exposed for WebApplicationFactory in Solia.DataPipeline.Tests.
public partial class Program;
