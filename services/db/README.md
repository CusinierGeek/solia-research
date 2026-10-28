# db (PostgreSQL 16)

Schema and migrations — the **single source of truth** for Solia's database. Raw SQL is
authoritative (the DB is polyglot: the Python ai-service reads the same tables), so the
schema is defined in SQL, not in any ORM model.

- **Schema reference:** [DATA-MODEL.md](../../docs/DATA-MODEL.md)
- **Migrations:** `migrations/*.sql`, applied in filename order.
- **Runner:** `Solia.DbMigrator` (minimal, Npgsql-only).

## Layout

```
services/db/
├── migrations/            # ordered SQL migrations (source of truth)
│   ├── 0001_init.sql
│   └── 0002_bounded_numeric.sql
└── Solia.DbMigrator/      # applies migrations idempotently
```

## Run the migrations locally

1. Start PostgreSQL (from the repo root — the dev override publishes the compose Postgres 16 on localhost;
   in the full stack the `db-migrate` job applies the migrations before any service starts):
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres   # publishes 127.0.0.1:5432
   ```
2. Run the migrator (from the repo root):
   ```bash
   # Local dev defaults assume Postgres on localhost. Override via env if needed.
   POSTGRES_HOST=localhost dotnet run --project services/db/Solia.DbMigrator
   ```

Running it again is a **no-op** — already-applied files are skipped.

## In Docker (one-shot job)

`infra/docker/db-migrator.Dockerfile` ships the migrator with `migrations/*.sql` baked in
(`MIGRATIONS_DIR=/app/migrations`, non-root). It applies pending migrations and **exits** — 0 when the
schema is up to date, non-zero on bad SQL, checksum drift, an unreachable database or a lock wait that
timed out — so the app services can be gated on it with `service_completed_successfully`. (The root
`docker-compose.yml` doesn't run it yet: that wiring lands with the Sprint 6 compose task.) It reads the
same `POSTGRES_*` variables as the other services:

```bash
docker build -f infra/docker/db-migrator.Dockerfile -t solia-db-migrator .
docker run --rm --network <compose-network> -e POSTGRES_HOST=postgres solia-db-migrator
```

## Configuration (environment variables)

| Variable | Default | Notes |
|----------|---------|-------|
| `SOLIA_DB_CONNECTION` | — | Full Npgsql connection string; takes precedence over the parts below. |
| `POSTGRES_HOST` | `localhost` | Use `postgres` inside Docker (service name). |
| `POSTGRES_PORT` | `5432` | |
| `POSTGRES_DB` | `solia` | |
| `POSTGRES_USER` | `solia` | |
| `POSTGRES_PASSWORD` | `change_me` | |
| `MIGRATOR_LOCK_TIMEOUT_SECONDS` | `300` | How long to wait for another runner's migration lock before exiting non-zero (`0` = don't wait). |
| `MIGRATIONS_DIR` | auto-resolved | Path to `migrations/` (the first CLI arg takes precedence). An explicit path that doesn't exist is an error; otherwise `services/db/migrations` is searched for upward from the working and executable directories. |

## How the runner works

- Takes a Postgres advisory lock first, so several containers starting at once apply migrations one at a time.
  The wait is bounded (`MIGRATOR_LOCK_TIMEOUT_SECONDS`): behind a stuck runner it exits non-zero instead of hanging.
- Bootstraps a `schema_migrations (filename, checksum, applied_at)` table.
- Applies each not-yet-applied `*.sql` file **inside a transaction**, then records its name + SHA-256 checksum
  (computed over LF-normalized text, so CRLF/LF checkouts agree; `.gitattributes` also pins `*.sql` to LF).
- On re-run, applied files are skipped; if an already-applied file's checksum changed, the runner **fails loudly**.
- Postgres errors are reported with SQLSTATE, detail, hint and position.

## Adding a new migration

Create a new, higher-numbered file (e.g. `0002_add_x.sql`). **Never edit an already-applied
migration** — add a new one instead (the checksum guard enforces this).

Statements that can't run in a transaction (e.g. `CREATE INDEX CONCURRENTLY`) go in their own
file whose **first line** is `-- solia:no-transaction`. Such a file must contain a **single
statement** (Postgres runs a multi-statement batch as one implicit transaction). If it fails
part-way — or the run dies after the statement but before its `schema_migrations` row is written
(the two can't be atomic without a transaction) — it is re-run on the next run, so write it to be
re-runnable (e.g. `CREATE INDEX CONCURRENTLY IF NOT EXISTS`). A failing no-transaction file says so in
the migrator's error output.
