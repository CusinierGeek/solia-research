using System.Text.Json;
using Solia.Api.Bot;
using Solia.Api.Endpoints;
using Solia.Api.Hosting;
using Solia.Api.Logging;
using Solia.Api.Middleware;
using Solia.Api.OpenApi;
using Solia.Bot.Configuration;
using Solia.Core.Ai;
using Solia.Core.Data;
using Solia.Core.Http;

var builder = WebApplication.CreateBuilder(args);

// --- Options: bound from config sections + flat .env.example keys (BACKEND_URL, ...), validated at
// startup (fail fast on bad values) ---
builder.Services.AddApiOptions(builder.Configuration);

// --- Secrets: secrets/secrets.enc (SOLIA_SECRETS_FILE) decrypted in memory with SOLIA_MASTER_KEY (AES-256-GCM).
// Registered first: it provides the shared DB data source (db_connection / db_password override plaintext
// config) that the bot, EF context and log sink all reuse. ---
builder.Services.AddSoliaSecrets(builder.Configuration);

// --- Paper-trading bot (Bot:* / PAPER_*): validated options (the equity curve starts from the configured
// starting balance), the DB/ai-service feed, the Postgres trade store and the hosted bot, run only when
// Bot:Enabled / PAPER_BOT_ENABLED is set. Paper only. ---
builder.Configuration.AddBotFlatEnvAliases();
builder.Services.AddHostedPaperTradingBot(builder.Configuration);

// --- JSON: snake_case on the wire (docs/API-CONTRACTS.md: total_pnl, page_size, ...) ---
builder.Services.ConfigureHttpJsonOptions(options =>
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower);

// --- Data: pooled EF Core context mapped over the Sprint-1 schema (read endpoints, stats, bot host).
// Shares the pooled, secrets-aware NpgsqlDataSource with the log sink and the bot. ---
builder.Services.AddSoliaDbContext(builder.Configuration);

// --- ai-service: typed HttpClient (AiService:BaseUrl / AI_SERVICE_URL) with timeout, retry + jitter and
// a circuit breaker; failures surface as AiServiceException (mapped to the error envelope) ---
builder.Services.AddSoliaAiClient(builder.Configuration);

// --- Logging: console (default) + best-effort mirror of notable entries into the `logs` table ---
builder.Services.AddBackendLogSink(builder.Configuration);
builder.Logging.AddPostgresLogSink();

// --- Redaction: every logger scrubs known secret values (and the master key) before any provider sees them ---
builder.Logging.AddSecretRedaction();

// --- OpenAPI document (served in Development), with the public base URL as its server ---
builder.Services.AddOpenApi(options => options.AddDocumentTransformer<ApiServerDocumentTransformer>());

var app = builder.Build();

// First in the pipeline so every failure below becomes the documented error envelope.
app.UseMiddleware<ErrorShapeMiddleware>();

if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
    app.UseSwaggerUI(options => options.SwaggerEndpoint("/openapi/v1.json", "Solia backend API v1"));
}

// Liveness only; real DB/ai-service readiness checks land in T8.
app.MapGet(ApiRoutes.Health, () => Results.Ok(new { status = "ok" }));

// Versioned public surface: read/query endpoints and the AI-proxy endpoints (/predict, /model/*).
app.MapGroup(ApiRoutes.BasePath)
    .MapReadEndpoints()
    .MapPredictEndpoints()
    .MapModelEndpoints();

app.Run();

// Exposed for WebApplicationFactory in Solia.Api.Tests.
public partial class Program;
