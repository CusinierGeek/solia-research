# infra

Docker & orchestration assets. See the root [`docker-compose.yml`](../docker-compose.yml) and [Sprint 6](../docs/sprints/SPRINT-06-docker.md).

- `docker/*.Dockerfile` — per-service build definitions (filled in Sprint 6; `data-pipeline`, `ai-service`, `backend` done).

## Image conventions

- **Build context = repo root** (`docker build -f infra/docker/<service>.Dockerfile .`): .NET services
  reference `services/shared`. The root [`.dockerignore`](../.dockerignore) keeps the context small
  and secret-free (`bin/`, `obj/`, `.venv/`, `.git/`, test outputs, docs, and `.env` / `secrets/` /
  keys at **any** depth — `**/` patterns, since Dockerfiles COPY whole service folders).
- **Multi-stage**: SDK stage restores from project files only (cached until a `.csproj` changes),
  then publishes Release; the final stage is the runtime image only — no SDK, sources or tests.
- **Non-root**: .NET images run as the built-in `app` user (`USER $APP_UID`, UID 1654).
- **Config from env vars only** (`.env.example` keys); `ASPNETCORE_ENVIRONMENT=Production` (image
  default, pinned in compose so `.env` can't switch containers to Development) and
  `appsettings.Development.json` is never published.
- **Healthcheck without curl**: .NET services listen on `8080` in-container (the base image's
  `ASPNETCORE_HTTP_PORTS`) and probe themselves with `dotnet <Service>.dll --healthcheck`
  (`Solia.Shared.Hosting.HealthProbe`, exit 0 on 2xx; follows `ASPNETCORE_URLS`/`ASPNETCORE_HTTP_PORTS`
  if overridden). Run it via `env DOTNET_gcServer=0 DOTNET_TieredPGO=0` to keep the probe process light.
  The backend probes `/health/live` (liveness, no dependency); `/health/ready` stays for readiness.
- **Secrets (backend)**: nothing baked in. `secrets.enc` is read from `/app/secrets/secrets.enc` (the
  default relative path under `WORKDIR /app`), mounted **read-only** at runtime; `SOLIA_MASTER_KEY`
  comes from the environment. No mount ⇒ plaintext config; explicit `SOLIA_SECRETS_FILE` missing, no
  key or wrong key ⇒ the container exits non-zero ([`docs/SECRETS.md`](../docs/SECRETS.md)).
  `Solia.Secrets.Cli` is not in the image.
- **CI**: the `docker` job in [`ci.yml`](../.github/workflows/ci.yml) builds every image on each PR
  (no push), asserts `id -u` ≠ 0, and optionally waits for the HEALTHCHECK against Postgres
  (failing fast if the container turns unhealthy or exits). Add a
  new image as one `matrix.include` entry.

Today only `postgres` runs: `docker compose up -d postgres`.
