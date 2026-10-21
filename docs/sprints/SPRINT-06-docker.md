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
- [ ] `infra/docker/data-pipeline.Dockerfile` (python slim, non-root).
- [ ] `infra/docker/ai-service.Dockerfile` (torch base, cache wheels).
- [ ] `infra/docker/backend.Dockerfile` (.NET 10 SDK→runtime multi-stage).
- [ ] `infra/docker/dashboard.Dockerfile` (.NET 10 multi-stage).
- [ ] Finalize root `docker-compose.yml` (replace the skeleton).
- [ ] Healthchecks + ordered startup.
- [ ] Migration bootstrap.
- [ ] Volumes + internal network; publish only dashboard (8081) and backend (8080).
- [ ] `make up` / compose docs in README.

## Deliverables
- All Dockerfiles + working compose.
- Updated README quickstart.

## Definition of Done
- [ ] `cp .env.example .env && docker compose up -d` starts all services healthy.
- [ ] Dashboard reachable; data flows end-to-end in containers.
- [ ] Volumes persist across `down`/`up`.
- [ ] Images reasonably sized; containers run non-root.

## Kickoff prompt
> We're implementing **Sprint 6 — Docker & orchestration** of Solia. Read `docs/ARCHITECTURE.md` (deployment topology) and the existing `docker-compose.yml` skeleton + `infra/docker/*` stubs. Write multi-stage, slim, non-root Dockerfiles for data-pipeline, ai-service, backend (.NET 10), and dashboard (.NET 10); finalize `docker-compose.yml` with healthchecks, ordered startup, an internal network, and persistent volumes (pgdata, models, secrets ro); add automatic DB migration on startup; publish only backend:8080 and dashboard:8081. Verify `docker compose up -d` brings everything up healthy end-to-end. Branch `sprint/06-docker`, PR per `CONTRIBUTING.md`.
