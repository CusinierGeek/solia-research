# syntax=docker/dockerfile:1
# Solia data-pipeline — .NET 10 worker (ASP.NET Core host for /health).
# Build context = repo root (the project references services/shared/Solia.Shared):
#   docker build -f infra/docker/data-pipeline.Dockerfile -t solia-data-pipeline .

# ---- build: restore (cached on project files only), then publish Release ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Project files first: editing a .cs file doesn't invalidate the restore layer.
COPY services/shared/Solia.Shared/Solia.Shared.csproj services/shared/Solia.Shared/
COPY services/data-pipeline/Solia.DataPipeline/Solia.DataPipeline.csproj services/data-pipeline/Solia.DataPipeline/
RUN dotnet restore services/data-pipeline/Solia.DataPipeline/Solia.DataPipeline.csproj

# Sources of the published project + its reference only (no test projects).
COPY services/shared/Solia.Shared/ services/shared/Solia.Shared/
COPY services/data-pipeline/Solia.DataPipeline/ services/data-pipeline/Solia.DataPipeline/
RUN dotnet publish services/data-pipeline/Solia.DataPipeline/Solia.DataPipeline.csproj \
        -c Release --no-restore -o /app /p:UseAppHost=false

# ---- runtime: ASP.NET Core runtime only (no SDK, no sources), non-root ----
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Configuration comes from environment variables only (.env.example keys, e.g. POSTGRES_*,
# PUMPAPI_*, INGEST): Production environment, and appsettings.Development.json isn't published.
# Kestrel listens on the base image's ASPNETCORE_HTTP_PORTS (8080).
ENV ASPNETCORE_ENVIRONMENT=Production \
    DOTNET_NOLOGO=1

COPY --from=build /app .

# Mount point of the optional replay-archive mirror volume, owned by the app user so a fresh named
# volume (which copies this directory's ownership) is writable. Empty in the image: archives only
# land there on `replay --mirror /replay-archives`.
RUN mkdir /replay-archives && chown $APP_UID /replay-archives

# Built-in non-root user of the .NET images (UID 1654).
USER $APP_UID
EXPOSE 8080

# No curl in the runtime image: the binary probes its own /health (Solia.Shared HealthProbe), on
# the port Kestrel is configured with. /health is 503 only when the database is unreachable; a
# pumpapi stream drop (INGEST or STREAM_PROBE on) is Degraded (200), so an external outage — which
# a restart can't fix — doesn't mark the container unhealthy.
# The probe is a short-lived second runtime: workstation GC and no tiered PGO keep its memory and
# CPU small (the Web SDK's server GC would size heaps per core for a single GET).
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["env", "DOTNET_gcServer=0", "DOTNET_TieredPGO=0", "dotnet", "Solia.DataPipeline.dll", "--healthcheck"]

ENTRYPOINT ["dotnet", "Solia.DataPipeline.dll"]
