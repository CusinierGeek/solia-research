using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using MudBlazor.Services;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components;
using Solia.Dashboard.Configuration;

var builder = WebApplication.CreateBuilder(args);

// --- Options: Backend:BaseUrl (+ flat BACKEND_URL), validated at startup — a missing/invalid URL fails fast ---
builder.Services.AddDashboardOptions(builder.Configuration);

// --- Backend REST client: typed ISoliaApiClient, bounded by timeouts, one retry on transient failures ---
builder.Services.AddSoliaApiClient();

// --- UI: Blazor Server (interactive server render mode) + MudBlazor component library ---
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddMudServices();

// --- Health: /health/live is dependency-free (the backend being down must not restart the dashboard) ---
builder.Services.AddHealthChecks();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseAntiforgery();

app.MapHealthChecks("/health/live", new HealthCheckOptions { Predicate = _ => false });

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();

// Exposed for WebApplicationFactory in Solia.Dashboard.Tests.
public partial class Program;
