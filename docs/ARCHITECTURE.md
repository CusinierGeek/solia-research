# Architecture

## 1. Goals & constraints

- **Modular**: each concern is an independent, containerized service with a clear contract.
- **Swappable data source**: the ingestion layer hides the provider behind a `DataSource` interface (primary: **pumpapi.io**; fallbacks: PumpPortal, Solana RPC+IDL) so the provider can change without touching downstream code. See [`DATA-SOURCES.md`](DATA-SOURCES.md).
- **Full token lifecycle**: data covers pre- **and** post-bonding (after graduation to PumpSwap/Raydium), required for correct pump/rug labels.
- **Paper-trading only**: no key ever signs a real transaction. The "bot" simulates fills against observed market prices.
- **Continuous learning**: new paper trades are labeled outcomes that feed periodic re-training.
- **Secure by default**: secrets live in an AES-encrypted file decrypted only at runtime (see [`SECRETS.md`](SECRETS.md)).
- **CI/CD ready**: every service builds in Docker; PRs run lint + tests.

## 2. Component overview

```
                         ┌──────────────────────────────────────────┐
                         │               PostgreSQL 16                │
                         │  tokens_live tokens_history features       │
                         │  predictions trades training_events        │
                         │  models logs                               │
                         └───────▲───────────────▲──────────────▲─────┘
                                 │ write          │ read/write   │ read
         ┌───────────────────────┴──┐   ┌─────────┴─────────┐   │
         │  data-pipeline (.NET)     │   │  ai-service (Py)  │   │
         │  • pumpapi WS + replay    │   │  • FastAPI        │   │
         │  • normalization          │   │  • Transformer    │   │
         │  • feature engineering     │   │  • train/infer    │   │
         └───────────────────────────┘   │  • retrain loop   │   │
                                          └───────▲───────────┘   │
                                                  │ HTTP /predict  │
                         ┌────────────────────────┴───────────────┴───┐
                         │            backend — ASP.NET Core (.NET 10)  │
                         │  • REST API (dashboard + control)            │
                         │  • AI client (calls ai-service)              │
                         │  • Paper-trading bot (hosted service)        │
                         │  • Secrets (AES) / config                    │
                         └───────▲──────────────────────────────┬──────┘
                                 │ SignalR / REST               │ REST
                         ┌───────┴───────────┐                  │
                         │ dashboard          │                  │
                         │ Blazor Server      │                  │
                         └────────────────────┘          (external callers)
```

## 3. Responsibilities per service

### data-pipeline (.NET worker)
- Connects to the **pumpapi.io** WebSocket (primary); can replay historical archives through the same path for training backfill.
- Normalizes raw events (live + replay) into a canonical schema; tracks tokens across graduation (pre/post-bonding).
- Maintains rolling windows per token and computes features (see [`DATA-PIPELINE.md`](DATA-PIPELINE.md)).
- Persists `tokens_live`, `tokens_history`, and `features`.
- Exposes a thin health endpoint; no business API.
- Reuses the SolGemFinder Pump.fun IDL decoder only for the autonomous `SolanaRpcSource` fallback.

### ai-service (Python, PyTorch + FastAPI)
- Owns the model: architecture, training, inference, versioning.
- `POST /predict` returns pump/rug probabilities, strategy, sizing, TP/SL, confidence.
- Hosts the **retrain orchestrator**: triggered after N new trades, trains a candidate, compares against the current champion, promotes or rolls back.
- Reads features/trades from PostgreSQL; writes to `predictions`, `models`, `training_events`.

### backend (ASP.NET Core, .NET 10)
- Public REST API consumed by the dashboard and external callers (see [`API-CONTRACTS.md`](API-CONTRACTS.md)).
- Calls `ai-service` for predictions and relays them.
- Hosts the **paper-trading bot** as a `BackgroundService`.
- Manages secrets (AES) and application config.

### dashboard (Blazor Server, .NET 10)
- Live UI over the backend: a typed, read-only REST client (`/api/v1/…`: tokens, predictions, trades, stats, logs,
  model status — never `/predict` or `/model/retrain`) and **one app-wide SignalR client** of the backend hub
  `/hubs/live` (`prediction_created`, `trade_opened`, `trade_closed`, `log_written`, `stats_changed`), auto-reconnecting
  with backoff. Delivery is at-most-once, so every (re)connect makes the live views re-fetch over REST.
- Views: overview (win-rate, total P&L, model status), live predictions, prediction history, trade history,
  performance (equity curve + win-rate donut, MudBlazor charts), bot logs.
- Degrades instead of failing: the backend down → each view shows an error state with a retry (or keeps its last data
  under a stale warning); the hub down → an app-bar LIVE / RECONNECTING / OFFLINE indicator, REST keeps working. See
  [`services/dashboard/README.md`](../services/dashboard/README.md).

### PostgreSQL
- Single source of truth. Schema in [`DATA-MODEL.md`](DATA-MODEL.md); migrations in `services/db/migrations`.

## 4. Data flow (happy path)

1. `data-pipeline` receives a trade event → normalizes → updates rolling windows → writes `features`.
2. `backend` (or a scheduler) requests a prediction for a token → calls `ai-service /predict` with the latest feature vector.
3. `ai-service` runs inference → persists a `predictions` row → returns the result.
4. `backend` paper-trading bot decides buy/sell based on the strategy + confidence → writes `trades`.
5. When a trade closes, the realized outcome is stored and counted toward the retrain threshold.
6. After N new closed trades, `ai-service` retrain orchestrator trains a candidate model, evaluates, and promotes/rolls back; result logged in `training_events` and `models`.
7. `dashboard` reflects everything live.

> The above is the *logical* flow. The **latency-critical live prediction path is wired differently** (no DB, no HTTP on the critical path) — see §5.

## 5. Inference paths & latency

Live "is this coin about to pump?" prediction is latency-sensitive (seconds matter), so it runs on a **different path** from training. **PostgreSQL is deliberately off the hot path.**

### Fast path — live prediction (hot)

```
event (pumpapi) ─► pipeline updates in-memory features ─► model inference ─► bot decision
                                  └─► async batched write to Postgres (history/training only)
```

- Features come from the pipeline's **in-memory rolling state**, never re-read from Postgres.
- Inference is **co-located in the pipeline** via an **ONNX** export of the trained model (ONNX Runtime for .NET) — **no HTTP, no DB** on the critical path. Our compute is ~single-digit to ~20 ms; end-to-end latency is dominated by the data source's own delivery lag, not by us.
- Postgres writes (`features`, `predictions`, `trades`) happen **asynchronously** — for history/dashboard/training, never to make the decision.
- A cheap **pre-filter** limits which tokens are scored each tick (e.g. young, accelerating volume) so throughput holds.

### Slow path — training / retraining (cold)

```
Postgres (accumulated features + trades) ─► ai-service reads in bulk ─► train ─► new champion (ONNX artifact)
```

- The Python **ai-service** owns training: it reads Postgres directly (large batch reads) and writes `models` / `training_events`.
- It produces the **ONNX artifact** that the fast path loads. No real-time constraint.

### REST `ai-service /predict` (moderate cadence)

The HTTP `/predict` endpoint (features in the request → prediction) stays for non-urgent, moderate-cadence scoring and tooling. It is **not** the per-event sniping path — that is the ONNX fast path. Both coexist.

> Decision date: the model format (ONNX) and the inference wiring are finalized in **Sprint 2**; perf tuning in **Sprint 8**.

## 6. Communication

- **backend ↔ ai-service**: HTTP/JSON (REST). Contracts in `API-CONTRACTS.md`.
- **backend ↔ dashboard**: REST (`/api/v1`, read endpoints) + a SignalR hub (`/hubs/live`, server → client push).
  Separately, browsers talk to the dashboard over Blazor Server's own SignalR circuit.
- **services ↔ db**: direct SQL — EF Core on the .NET side (backend + data-pipeline); SQLAlchemy/asyncpg on the Python side (ai-service).
- **internal network**: a dedicated Docker bridge network; only the dashboard (and optionally backend) are published to the host.

## 7. Deployment topology (Docker Compose)

`docker compose up -d` (or `make up`) starts everything on the internal `solia` bridge network; only
the backend and the dashboard are published on the host.

```
       host :8081                         host :8080
           │                                  │
       dashboard ───────── HTTP ─────────▶ backend ──── HTTP ────▶ ai-service ── models (volume)
                                              │                        │
                                              ▼                        │
  pumpapi.io ──WS──▶ data-pipeline ────▶  postgres  ◀──────────────────┘
                                              ▲   └─ pgdata (volume)
                               db-migrate ────┘  (one-shot, before every app service)
```

All six containers share the `solia` network and address each other by service name; pumpapi.io is
the only outbound dependency (data-pipeline).

| Container      | Image base              | Published | Starts after (condition)                     | Volume            | Restart          |
|----------------|-------------------------|-----------|----------------------------------------------|-------------------|------------------|
| postgres       | `postgres:16-alpine`    | — (internal) | —                                         | `pgdata`          | unless-stopped   |
| db-migrate     | `dotnet/runtime:10.0`   | —         | postgres (healthy)                           | —                 | no (one-shot)    |
| data-pipeline  | `dotnet/aspnet:10.0`    | —         | db-migrate (exited 0)                        | —                 | unless-stopped   |
| ai-service     | `python:3.11-slim`      | —         | db-migrate (exited 0)                        | `models` (`MODEL_DIR`) | unless-stopped |
| backend        | `dotnet/aspnet:10.0`    | 8080      | db-migrate (exited 0), ai-service (healthy)  | `./secrets` → `/app/secrets` (ro) | unless-stopped |
| dashboard      | `dotnet/aspnet:10.0`    | 8081      | backend (healthy)                            | —                 | unless-stopped   |

- `db-migrate` is a job, not a server: it applies `services/db/migrations` and exits; no service
  touches the database before it succeeds.
- The backend reads only the database and ai-service, so it doesn't wait for the data-pipeline: a
  failing pipeline never keeps 8080/8081 down.
- Every long-running container has a healthcheck (the image's `HEALTHCHECK`, timing tuned in
  compose so cold starts don't flap). The data-pipeline ingests (`INGEST=true` by default).
- Service hosts are pinned in compose (`POSTGRES_HOST=postgres`, `AI_SERVICE_URL=http://ai-service:8000`,
  `BACKEND_URL=http://backend:8080`) and `SOLIA_DB_CONNECTION` is blanked (it would win over
  `POSTGRES_*`); every container runs as `ASPNETCORE_ENVIRONMENT=Production`. Each service gets only
  the `.env` keys `docker-compose.yml` lists for it: a new `.env` key reaches no container until it is
  listed there. Other per-service settings (nested .NET keys like `Logging__LogLevel__Default`,
  `Bot__*`) go in the optional, gitignored `config/<service>.env`, loaded by that service only.
- Requires Docker Engine 25+ (healthcheck `start_interval`) and Docker Compose 2.24+. Config: `.env` via compose interpolation, per-service variable lists (`SOLIA_MASTER_KEY` → backend only), plus optional `config/<service>.env`.
- Every container runs non-root; `pgdata` and `models` survive `docker compose down` (only
  `down -v` / `make reset` deletes them). `infra/scripts/smoke.sh` (`make smoke`, CI `compose-up`)
  asserts this whole topology on the real stack.
- Local dev: `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres`
  (`make dev-db`) publishes Postgres on `127.0.0.1:5432` for `dotnet run` / pytest
  (`POSTGRES_HOST=localhost`).

See [`../infra/`](../infra/), [`../docker-compose.yml`](../docker-compose.yml) and
[`../docker-compose.dev.yml`](../docker-compose.dev.yml).

## 8. Cross-cutting concerns

- **Config**: 12-factor via env vars (`.env`), plus AES secrets file for sensitive values.
- **Logging**: structured JSON logs; a subset persisted to the `logs` table for the dashboard.
- **Observability**: health endpoints per service; metrics hooks reserved for Sprint 8.
- **Testing**: unit tests per service + integration tests against an ephemeral Postgres (see each sprint's DoD).

## 9. Explicit non-goals (v1)

- No real-money execution, wallet signing, or private-key custody.
- No multi-exchange support (Pump.fun only).
- No distributed training (single-node PyTorch).
