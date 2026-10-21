# infra

Docker & orchestration assets. See the root [`docker-compose.yml`](../docker-compose.yml) and [Sprint 6](../docs/sprints/SPRINT-06-docker.md).

- `docker/*.Dockerfile` — per-service build definitions (skeletons today, filled in Sprint 6).

Today only `postgres` runs: `docker compose up -d postgres`.
