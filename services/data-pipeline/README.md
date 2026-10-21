# data-pipeline (.NET 10 worker)

Ingests Pump.fun activity from **pumpapi.io** — live WebSocket **and** historical replay — normalizes events, computes features across the full token lifecycle (pre/post-bonding), and persists them to PostgreSQL.

- **Design:** [DATA-SOURCES.md](../../docs/DATA-SOURCES.md) · [DATA-PIPELINE.md](../../docs/DATA-PIPELINE.md)
- **Sprint:** [Sprint 1](../../docs/sprints/SPRINT-01-data-pipeline.md)

## Status

Skeleton (T2): runnable host with configuration, structured logging + a `logs`-table sink,
a `/health` endpoint, and graceful shutdown. Ingestion, parsing, features and persistence
land in later Sprint-1 tasks (T3–T7).

## Run locally

```bash
# 1. Start PostgreSQL + apply migrations (see services/db/README.md)
docker compose up -d postgres
POSTGRES_HOST=localhost dotnet run --project services/db/Solia.DbMigrator

# 2. Run the worker (the launch profile listens on http://localhost:8090)
POSTGRES_HOST=localhost dotnet run --project services/data-pipeline/Solia.DataPipeline
```

- `GET http://localhost:8090/health` → `Healthy` when the database is reachable, `Unhealthy`
  (HTTP 503) otherwise. The database check times out after 3 s.
- A startup row appears in `logs` (`service = 'data-pipeline'`).

## Configuration

Bound from `appsettings.json` (or any configuration source, e.g. `Pipeline__WindowSeconds`), with the
flat env vars from `.env.example` mapped onto their section keys and taking precedence
(`Config/FlatEnvAliases.cs`). The database settings are resolved by the shared
`Solia.Shared` library, the same code `Solia.DbMigrator` uses. Values are trimmed; a value that is set
but invalid (bad port, `ENRICH_HOLDERS=maybe`) fails startup with the variable's name.

| Env var | Default | Meaning |
|---------|---------|---------|
| `PUMPAPI_WS_URL` | `wss://stream.pumpapi.io/` | live stream |
| `PUMPAPI_REPLAY_BASE_URL` | `https://replay.pumpapi.io` | historical replay base |
| `ENRICH_HOLDERS` | `false` | paid holder enrichment (off = approximate from stream); accepts `true/false`, `1/0`, `yes/no`, `on/off` |
| `SOLIA_DB_CONNECTION` | — | full Npgsql connection string (wins over `POSTGRES_*`) |
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | `localhost/5432/solia/solia/change_me` | DB parts |

`Pipeline` section knobs: `SnapshotCadenceSeconds` (1–3600), `WindowSeconds` (1–86400, and
≥ the cadence), `MaxTrackedTokens` (1–1,000,000), `EnrichHolders`. `PumpApi` URLs must be absolute
(`ws(s)://` for the stream, `http(s)://` for replay). Invalid values stop the worker at startup.

## Tests

```bash
dotnet test Solia.slnx
```

`Solia.DataPipeline.Tests` covers flat-key mapping, options validation, log levels and the host's
`/health` behaviour (against unreachable databases, so no Postgres is needed).

> The autonomous `SolanaRpcSource` fallback (future) reuses the Pump.fun IDL decoder from the old SolGemFinder project.
