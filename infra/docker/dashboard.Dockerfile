# syntax=docker/dockerfile:1
# Solia dashboard — Blazor Server live UI over the backend REST API + /hubs/live (paper trading only).
# Build context = repo root (the project references services/shared/Solia.Shared):
#   docker build -f infra/docker/dashboard.Dockerfile -t solia-dashboard .

# ---- build: restore (cached on project files only), then publish Release ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Project files first: editing a .cs/.razor file doesn't invalidate the restore layer.
COPY services/shared/Solia.Shared/Solia.Shared.csproj services/shared/Solia.Shared/
COPY services/dashboard/Solia.Dashboard/Solia.Dashboard.csproj services/dashboard/Solia.Dashboard/
RUN dotnet restore services/dashboard/Solia.Dashboard/Solia.Dashboard.csproj

# Sources of the published project + its reference only (no tests). Publish includes the static web
# assets (wwwroot, MudBlazor's _content, the _framework/blazor.web.js script).
COPY services/shared/Solia.Shared/ services/shared/Solia.Shared/
COPY services/dashboard/Solia.Dashboard/ services/dashboard/Solia.Dashboard/
RUN dotnet publish services/dashboard/Solia.Dashboard/Solia.Dashboard.csproj \
        -c Release --no-restore -o /app /p:UseAppHost=false

# ---- runtime: ASP.NET Core runtime only (no SDK, no sources), non-root ----
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Configuration comes from environment variables only: BACKEND_URL (internal http://backend:8080) is
# required — there is no default, so a missing or invalid URL fails startup (non-zero exit).
# Production environment, and appsettings.Development.json isn't published.
# Kestrel listens on 8081 over plain HTTP: no HTTPS redirect / HSTS, so the published port and the
# Blazor SignalR circuit (WebSockets) work as-is; TLS termination is a reverse proxy's job (future).
ENV ASPNETCORE_ENVIRONMENT=Production \
    ASPNETCORE_HTTP_PORTS=8081 \
    DOTNET_NOLOGO=1

COPY --from=build /app .

# Built-in non-root user of the .NET images (UID 1654).
USER $APP_UID
EXPOSE 8081

# No curl in the runtime image: the binary probes its own /health/live (Solia.Shared HealthProbe), on
# the port Kestrel is configured with (ASPNETCORE_HTTP_PORTS). Dependency-free, so the backend being
# down — which a dashboard restart can't fix — doesn't mark the container unhealthy.
# The probe is a short-lived second runtime: workstation GC and no tiered PGO keep it small.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --start-interval=2s --retries=3 \
    CMD ["env", "DOTNET_gcServer=0", "DOTNET_TieredPGO=0", "dotnet", "Solia.Dashboard.dll", "--healthcheck"]

ENTRYPOINT ["dotnet", "Solia.Dashboard.dll"]
