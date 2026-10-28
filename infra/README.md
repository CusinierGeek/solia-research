# infra

Docker & orchestration assets. See the root [`docker-compose.yml`](../docker-compose.yml) and [Sprint 6](../docs/sprints/SPRINT-06-docker.md).

- `docker/*.Dockerfile` — per-service build definitions (`data-pipeline`, `ai-service`, `backend`, `dashboard`, `db-migrator`).

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
- **Dashboard**: listens on **8081** (`ASPNETCORE_HTTP_PORTS=8081`) over plain HTTP — no HTTPS
  redirect / HSTS, so the published port and the Blazor SignalR circuit (WebSockets) work as-is; TLS
  termination is a future reverse proxy's job. `BACKEND_URL` (internal `http://backend:8080`) is
  required: missing or invalid ⇒ the container exits non-zero. Its HEALTHCHECK probes the
  dependency-free `/health/live` (backend down ≠ unhealthy). `RequiresAspNetWebAssets` is pinned in
  the `.csproj` so the csproj-only restore layer still pulls `_framework/blazor.web.js`.
- **Secrets (backend)**: nothing baked in. `secrets.enc` is read from `/app/secrets/secrets.enc` (the
  default relative path under `WORKDIR /app`), mounted **read-only** at runtime; `SOLIA_MASTER_KEY`
  comes from the environment. No mount ⇒ plaintext config; explicit `SOLIA_SECRETS_FILE` missing, no
  key or wrong key ⇒ the container exits non-zero ([`docs/SECRETS.md`](../docs/SECRETS.md)).
  `Solia.Secrets.Cli` is not in the image.
- **CI**: the `docker` job in [`ci.yml`](../.github/workflows/ci.yml) builds every image on each PR
  (no push), asserts `id -u` ≠ 0, and optionally waits for the HEALTHCHECK against Postgres
  (failing fast if the container turns unhealthy or exits). Add a
  new image as one `matrix.include` entry.

## Compose

- Requires **Docker Engine 25+** (healthcheck `start_interval`) and **Docker Compose 2.24+**.
- **Config = `.env` via compose interpolation**: each service lists exactly the variables it reads
  (`${VAR:-default}`), so a key reaches only the containers that need it — `SOLIA_MASTER_KEY` goes to
  the backend alone. A `.env` key that compose doesn't list reaches **no** container. For anything
  else (nested .NET keys such as `Logging__LogLevel__Default`, `Bot__*`, `Pipeline__*`), create the
  optional, gitignored `config/<service>.env` (e.g. `config/backend.env`): `env_file` with
  `required: false`, loaded by that service only; the values pinned in compose win over it. `cp .env.example .env` works as-is; host env vars win
  over `.env`; without a `.env` every default applies. `.env.example` marks each key `[compose]` or
  `[local dev]`.
- `docker compose up -d --build` — the whole stack, in order: postgres → `db-migrate` (one-shot) →
  data-pipeline + ai-service → backend → dashboard (the backend waits for db-migrate and ai-service,
  not the pipeline). Only `8080` (backend) and `8081` (dashboard) are
  published. Topology: [`docs/ARCHITECTURE.md` §7](../docs/ARCHITECTURE.md#7-deployment-topology-docker-compose).
- `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres` — local dev:
  Postgres alone, published on `127.0.0.1:5432` for `dotnet run` / pytest.
- Compose healthchecks only tune timing; the probe command is each image's `HEALTHCHECK`.
- `.env` can't misroute the containers: service hosts, in-container URLs/paths are pinned,
  `SOLIA_DB_CONNECTION` is blanked (it wins over `POSTGRES_*`) and every container runs as Production.
- **Secrets mount**: `./secrets` → `/app/secrets` (ro). The backend runs as UID 1654, so
  `secrets/secrets.enc` must be readable by it (`chmod 644 secrets/secrets.enc`, or
  `chown 1654 secrets/secrets.enc` to keep mode 600). Create `./secrets` yourself: if it's missing
  Docker creates it root-owned on the host (the backend then just runs on plaintext config).
  Creating the file and injecting the key: [`docs/SECRETS.md` § Docker](../docs/SECRETS.md#docker-compose).
- The `compose` job in [`ci.yml`](../.github/workflows/ci.yml) validates both files and asserts the
  topology (services, published ports, startup conditions, restart policies, volumes, service hosts,
  the exact variable set of each service, `.env.example` coverage, optional `config/<service>.env` per service only, git/docker ignore of `.env`/`secrets/`/`config/*.env`);
  `compose-up` starts the real stack from `.env.example` on a fresh volume (`INGEST=false`), waits
  until db-migrate exits 0 and every service is healthy, then checks the secrets path: a wrong key
  makes the backend exit non-zero without leaking the key or a value, the right key brings it back.
