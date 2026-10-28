# Sprint 6 — Docker & orchestration

**Milestone:** Docker & orchestration · **Depends on:** Sprints 1–5 · **Service:** `infra/`, root compose

## Goal
`docker compose up` brings the whole system online: Postgres, data-pipeline, ai-service, backend, dashboard — on an internal network with persistent volumes.

## Scope
- Production-ready Dockerfiles for each service (multi-stage, slim).
- `docker-compose.yml` with service graph, healthchecks, depends_on, internal network, volumes (`pgdata`, `models`, `secrets` ro).
- Automatic DB migration on startup (init job or entrypoint).
- `.env`-driven configuration; secrets via master key injection.

## Out of scope
- Kubernetes / cloud deploy (future). CI publish of images (touch in Sprint 8).

## Task breakdown
Each line is one GitHub issue / PR (milestone *Sprint 6*).
- [x] `infra/docker/data-pipeline.Dockerfile` (.NET 10 SDK→aspnet multi-stage, non-root) + image CI build — #96. *(The original brief said "python slim": the data-pipeline is a .NET worker.)*
- [x] `infra/docker/ai-service.Dockerfile` (Python 3.11 slim, CPU torch, cached deps, non-root) — #97.
- [x] `infra/docker/backend.Dockerfile` (.NET 10 SDK→runtime multi-stage, read-only secrets mount) — #98.
- [x] `infra/docker/dashboard.Dockerfile` (.NET 10 multi-stage, Blazor Server on 8081) — #99.
- [x] Migration bootstrap: one-shot `db-migrate` job (`infra/docker/db-migrator.Dockerfile`) — #100.
- [x] Finalize root `docker-compose.yml`: healthchecks + ordered startup, internal `solia` network, `pgdata` / `models` volumes, publish only backend (8080) and dashboard (8081) — #101.
- [x] `.env`-driven config (per-service variable lists) + master-key injection (backend only) — #102.
- [x] End-to-end compose smoke test (health, topology, non-root, persistence, opt-in data flow) in CI — #103.
- [x] Quickstart docs, `make up|down|logs|smoke` shortcuts (`infra/scripts/solia.sh` without make), this DoD — #104.

## Deliverables
- All Dockerfiles + working compose — `infra/docker/`, [`docker-compose.yml`](../../docker-compose.yml), [`infra/README.md`](../../infra/README.md).
- Updated README quickstart — [`README.md` § Quick start](../../README.md#quick-start).

## Definition of Done
Verified by `infra/scripts/smoke.sh` (`make smoke`), which CI's `compose-up` job runs on every PR.
- [x] `cp .env.example .env && docker compose up -d` starts all services healthy (`db-migrate` exits 0; every long-running service healthy).
- [x] Dashboard reachable (`:8081`); data flows end-to-end in containers — pumpapi → data-pipeline → Postgres → backend `/api/v1/tokens/live`, checked by the opt-in `SMOKE_DATA_FLOW=1` (needs the live stream, so skipped in CI). Known issue: live feature writes intermittently hit a NUMERIC overflow (`22003`, migration 0002 bounds) and drop batches — tracked as a pipeline follow-up, not a container issue.
- [x] Volumes persist across `down`/`up` (DB rows and a `models` marker file survive).
- [x] Images reasonably sized (.NET images ~300–360 MB, ai-service ~2.1 GB with CPU torch); every container runs non-root.

## Kickoff prompt
*(Historical — the sprint is done; kept as the original brief.)*
> We're implementing **Sprint 6 — Docker & orchestration** of Solia. Read `docs/ARCHITECTURE.md` (deployment topology) and the existing `docker-compose.yml` + `infra/docker/*` placeholders. Write multi-stage, slim, non-root Dockerfiles for data-pipeline (.NET 10), ai-service (Python 3.11), backend (.NET 10), and dashboard (.NET 10); finalize `docker-compose.yml` with healthchecks, ordered startup, an internal network, and persistent volumes (pgdata, models, secrets ro); add automatic DB migration on startup; publish only backend:8080 and dashboard:8081. Verify `docker compose up -d` brings everything up healthy end-to-end. Branch `sprint/06-docker`, PR per `CONTRIBUTING.md`.
