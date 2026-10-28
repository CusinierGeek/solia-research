# syntax=docker/dockerfile:1
# Solia db-migrator — one-shot job: applies pending services/db/migrations/*.sql, then exits.
# Exit 0 = schema up to date (incl. nothing to apply); non-zero = bad SQL, checksum drift,
# unreachable DB or lock wait timed out, so services gated on it with
# `service_completed_successfully` don't start (compose service `db-migrate`).
# Build context = repo root (the project references services/shared/Solia.Shared):
#   docker build -f infra/docker/db-migrator.Dockerfile -t solia-db-migrator .

# ---- build: restore (cached on project files only), then publish Release ----
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

# Project files first: editing a .cs file doesn't invalidate the restore layer.
COPY services/shared/Solia.Shared/Solia.Shared.csproj services/shared/Solia.Shared/
COPY services/db/Solia.DbMigrator/Solia.DbMigrator.csproj services/db/Solia.DbMigrator/
RUN dotnet restore services/db/Solia.DbMigrator/Solia.DbMigrator.csproj

# Sources of the published project + its reference only (no test projects).
COPY services/shared/Solia.Shared/ services/shared/Solia.Shared/
COPY services/db/Solia.DbMigrator/ services/db/Solia.DbMigrator/
RUN dotnet publish services/db/Solia.DbMigrator/Solia.DbMigrator.csproj \
        -c Release --no-restore -o /app /p:UseAppHost=false

# ---- runtime: .NET runtime only (console app: no ASP.NET, no SDK, no sources), non-root ----
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app

# Connection from the same env vars as the other services (POSTGRES_HOST=postgres, POSTGRES_PORT,
# POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD, or SOLIA_DB_CONNECTION). MIGRATIONS_DIR is explicit,
# so a missing directory fails loudly instead of probing for a repo layout that isn't there.
ENV MIGRATIONS_DIR=/app/migrations \
    DOTNET_NOLOGO=1

COPY --from=build /app .
# The SQL is the schema's source of truth: shipped as-is, read-only for the runtime user.
COPY services/db/migrations/*.sql /app/migrations/

# Built-in non-root user of the .NET images (UID 1654).
USER $APP_UID

# No HEALTHCHECK / EXPOSE: a job that exits, not a server. Re-running it is a no-op, and concurrent
# runs are serialized by a Postgres advisory lock.
ENTRYPOINT ["dotnet", "Solia.DbMigrator.dll"]
