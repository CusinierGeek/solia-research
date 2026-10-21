# Sprint 1 — Data pipeline

**Milestone:** Data pipeline · **Depends on:** — · **Service:** `services/data-pipeline`, `services/db`

## Goal
Stand up PostgreSQL with the full schema, and a Python service that ingests PumpPortal WebSocket events, computes features, and persists them.

## Scope
- PostgreSQL schema + migrations for all tables in [`../DATA-MODEL.md`](../DATA-MODEL.md).
- `DataSource` abstraction + `PumpPortalWebSocketSource` + `ReplaySource`.
- Normalization, rolling windows, feature computation ([`../DATA-PIPELINE.md`](../DATA-PIPELINE.md)).
- Idempotent persistence to `tokens_live`, `tokens_history`, `features`.
- Health endpoint, structured logging to `logs`.

## Out of scope
- Any ML (Sprint 2). Any .NET (Sprint 4). Full Docker wiring (Sprint 6) — a local Postgres container is fine here.

## Task breakdown
- [ ] Choose migration tool (Alembic or raw SQL) and set up `services/db/migrations`.
- [ ] Write DDL for all 8 tables + indexes.
- [ ] Project skeleton (`pyproject.toml`/`requirements.txt`, lint: ruff, format: black, tests: pytest).
- [ ] DB access layer (asyncpg or SQLAlchemy async).
- [ ] `RawEvent` model + `DataSource` protocol.
- [ ] `PumpPortalWebSocketSource`: connect, subscribe, reconnect w/ backoff, resubscribe.
- [ ] `ReplaySource`: read recorded events from a file.
- [ ] Normalizer: source payload → `RawEvent`.
- [ ] Rolling-window manager (bounded memory, eviction of inactive tokens).
- [ ] Feature computation (all features in the pipeline doc).
- [ ] Persistence (idempotent upserts, batching).
- [ ] Health endpoint + structured logging.
- [ ] Record a small real event sample for `ReplaySource` fixtures.

## Deliverables
- `services/db/migrations/*`
- `services/data-pipeline/` runnable service (`python -m solia_pipeline`)
- Unit tests (feature math) + integration test (ReplaySource → ephemeral Postgres)
- Service README with run instructions

## Definition of Done
- [ ] `docker run postgres` + migrations apply cleanly.
- [ ] Running against PumpPortal (or replay) produces rows in `features`.
- [ ] Reconnect verified (kill/restore WS).
- [ ] Unit + integration tests green in CI.
- [ ] Feature definitions documented and matched to the spec.

## Kickoff prompt
> We're implementing **Sprint 1 — Data pipeline** of the Solia project. Read `docs/ARCHITECTURE.md`, `docs/DATA-MODEL.md`, and `docs/DATA-PIPELINE.md`. Create the PostgreSQL migrations for all tables, then build the Python `data-pipeline` service: a swappable `DataSource` with a PumpPortal WebSocket adapter and a Replay adapter, event normalization, rolling windows, feature computation, and idempotent persistence. Add ruff/black/pytest, unit tests for feature math, and an integration test using the Replay source against an ephemeral Postgres. Work on a branch `sprint/01-data-pipeline` and open a PR following `CONTRIBUTING.md`. Ask me before finalizing the exact PumpPortal subscription payloads.
