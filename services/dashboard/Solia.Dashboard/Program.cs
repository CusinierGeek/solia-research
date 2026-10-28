using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using MudBlazor.Services;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components;
using Solia.Dashboard.Configuration;
using Solia.Dashboard.Live;
using Solia.Shared.Hosting;

// --- Container healthcheck: `--healthcheck` probes /health/live and exits without starting the host (no backend
// call, no options validation). Liveness only: the backend being down must not mark the dashboard unhealthy. ---
if (HealthProbe.IsRequested(args))
{
    return await HealthProbe.RunAsync(args, path: DashboardRoutes.HealthLive);
}

var builder = WebApplication.CreateBuilder(args);

// --- Options: Backend:BaseUrl (+ flat BACKEND_URL), validated at startup — a missing/invalid URL fails fast ---
builder.Services.AddDashboardOptions(builder.Configuration);

// --- Backend REST client: typed ISoliaApiClient, bounded by timeouts, one retry on transient failures ---
builder.Services.AddSoliaApiClient();

// --- Live feed: one app-wide SignalR connection to {BaseUrl}/hubs/live, auto-reconnecting (never blocks startup) ---
builder.Services.AddLiveFeed();

// --- Clock: injected so views that show ages (and their tests) don't read the system clock directly ---
builder.Services.AddSingleton(TimeProvider.System);

// --- UI: Blazor Server (interactive server render mode) + MudBlazor component library ---
builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();
builder.Services.AddMudServices();

// --- Health: /health/live is dependency-free (the backend being down must not restart the dashboard) ---
builder.Services.AddHealthChecks();

var app = builder.Build();

// No UseHttpsRedirection/UseHsts: the container serves plain HTTP on its published port (TLS, if any, is
// terminated in front of it), so a redirect or an HSTS header would break the page and the SignalR circuit.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
}
app.UseStatusCodePagesWithReExecute("/not-found", createScopeForStatusCodePages: true);

app.UseAntiforgery();

app.MapHealthChecks(DashboardRoutes.HealthLive, new HealthCheckOptions { Predicate = _ => false });

app.MapStaticAssets();
app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

await app.RunAsync();
return 0;

// Exposed for WebApplicationFactory in Solia.Dashboard.Tests.
public partial class Program;
