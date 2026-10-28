# syntax=docker/dockerfile:1
# Solia backend — ASP.NET Core REST API + paper-trading bot host + AES secrets (paper only).
# Build context = repo root (the projects reference services/shared/Solia.Shared):
#   docker build -f infra/docker/backend.Dockerfile -t solia-backend .

# ---- build: restore (cached on project files only), then publish Release ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Project files first: editing a .cs file doesn't invalidate the restore layer.
COPY services/shared/Solia.Shared/Solia.Shared.csproj services/shared/Solia.Shared/
COPY services/backend/Solia.Core/Solia.Core.csproj services/backend/Solia.Core/
COPY services/backend/Solia.Secrets/Solia.Secrets.csproj services/backend/Solia.Secrets/
COPY services/backend/Solia.Bot/Solia.Bot.csproj services/backend/Solia.Bot/
COPY services/backend/Solia.Api/Solia.Api.csproj services/backend/Solia.Api/
RUN dotnet restore services/backend/Solia.Api/Solia.Api.csproj

# Sources of the published project + its references only (no tests, no Solia.Secrets.Cli).
COPY services/shared/Solia.Shared/ services/shared/Solia.Shared/
COPY services/backend/Solia.Core/ services/backend/Solia.Core/
COPY services/backend/Solia.Secrets/ services/backend/Solia.Secrets/
COPY services/backend/Solia.Bot/ services/backend/Solia.Bot/
COPY services/backend/Solia.Api/ services/backend/Solia.Api/
RUN dotnet publish services/backend/Solia.Api/Solia.Api.csproj \
        -c Release --no-restore -o /app /p:UseAppHost=false

# ---- runtime: ASP.NET Core runtime only (no SDK, no sources), non-root ----
FROM mcr.microsoft.com/dotnet/aspnet:10.0 AS runtime
WORKDIR /app

# Configuration comes from environment variables only (.env.example keys, e.g. POSTGRES_*,
# AI_SERVICE_URL, PAPER_*): Production environment, and appsettings.Development.json isn't published.
# Kestrel listens on the base image's ASPNETCORE_HTTP_PORTS (8080).
ENV ASPNETCORE_ENVIRONMENT=Production \
    DOTNET_NOLOGO=1

COPY --from=build /app .

# Secrets (docs/SECRETS.md): nothing is baked in. secrets.enc is read from the default relative path
# secrets/secrets.enc, i.e. /app/secrets mounted read-only at runtime; SOLIA_MASTER_KEY comes from the
# environment. SOLIA_SECRETS_FILE is deliberately left unset: no mount => plaintext config, while an
# explicit SOLIA_SECRETS_FILE that is missing, or a wrong key, fails startup (non-zero exit).
# No VOLUME instruction: it would leave an anonymous volume per run; compose mounts it (`:ro`).

# Built-in non-root user of the .NET images (UID 1654).
USER $APP_UID
EXPOSE 8080

# No curl in the runtime image: the binary probes its own /health/live (Solia.Shared HealthProbe), on
# the port Kestrel is configured with. Liveness only (no DB / ai-service check), so a dependency outage
# — which a restart can't fix — doesn't mark the container unhealthy; /health/ready is for readiness.
# The probe is a short-lived second runtime: workstation GC and no tiered PGO keep it small.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["env", "DOTNET_gcServer=0", "DOTNET_TieredPGO=0", "dotnet", "Solia.Api.dll", "--healthcheck"]

ENTRYPOINT ["dotnet", "Solia.Api.dll"]
