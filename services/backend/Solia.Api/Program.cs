using Solia.Api.Hosting;
using Solia.Api.Logging;
using Solia.Api.Middleware;
using Solia.Api.OpenApi;
using Solia.Core.Http;

var builder = WebApplication.CreateBuilder(args);

// --- Options: bound from config sections + flat .env.example keys (BACKEND_URL, ...), validated at
// startup (fail fast on bad values) ---
builder.Services.AddApiOptions(builder.Configuration);

// --- Logging: console (default) + best-effort mirror of notable entries into the `logs` table ---
builder.Services.AddBackendLogSink(builder.Configuration);
builder.Logging.AddPostgresLogSink();

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

// Versioned public surface — endpoints land in T4/T5. Empty seam for now.
app.MapGroup(ApiRoutes.BasePath);

app.Run();

// Exposed for WebApplicationFactory in Solia.Api.Tests.
public partial class Program;
