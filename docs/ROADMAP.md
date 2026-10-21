# Roadmap — 8 sprints

Each sprint is a GitHub **milestone**, implemented on a feature branch and merged via PR (see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)). Sprints are mostly sequential; some can overlap once contracts are frozen.

| # | Sprint | Outcome | Depends on |
|---|--------|---------|------------|
| 1 | [Data pipeline](sprints/SPRINT-01-data-pipeline.md) | PostgreSQL schema + pumpapi ingestion (.NET, live+replay) + features flowing | — |
| 2 | [AI model](sprints/SPRINT-02-ai-model.md) | Transformer trains on historical data; inference works offline | 1 |
| 3 | [Paper-trading bot](sprints/SPRINT-03-paper-bot.md) | Bot simulates trades from predictions, records outcomes | 1, 2 |
| 4 | [Backend API](sprints/SPRINT-04-backend.md) | ASP.NET Core REST API + AI client + AES secrets, hosts bot | 1, 2, 3 |
| 5 | [Dashboard](sprints/SPRINT-05-dashboard.md) | Blazor Server live UI over the API | 4 |
| 6 | [Docker & orchestration](sprints/SPRINT-06-docker.md) | Full `docker compose up` brings everything online | 1–5 |
| 7 | [Continuous training](sprints/SPRINT-07-continuous-training.md) | Retrain-after-N, versioning, promote/rollback | 2, 3, 6 |
| 8 | [Optimization & QA](sprints/SPRINT-08-optimization-qa.md) | Perf, retention, metrics, test coverage, CI hardening | all |

## Dependency graph

```
S1 ─▶ S2 ─▶ S3 ─▶ S4 ─▶ S5
        └────┴────┴────┴──▶ S6 ─▶ S7 ─▶ S8
```

## Working rhythm

- One sprint ≈ one focused conversation. Start it by pasting that sprint's **Kickoff prompt**.
- Freeze the contracts (`API-CONTRACTS.md`, `DATA-MODEL.md`) before building dependents; change them via PR.
- Every PR must satisfy the sprint's **Definition of Done** and pass CI.
