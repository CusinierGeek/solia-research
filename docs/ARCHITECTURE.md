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
- Live UI over the backend API + SignalR push.
- Views: live predictions, prediction history, trade history, win-rate, total P&L, performance curve, bot logs.

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

## 5. Communication

- **backend ↔ ai-service**: HTTP/JSON (REST). Contracts in `API-CONTRACTS.md`.
- **backend ↔ dashboard**: in-process + SignalR (Blazor Server) and REST.
- **services ↔ db**: direct SQL — EF Core on the .NET side (backend + data-pipeline); SQLAlchemy/asyncpg on the Python side (ai-service).
- **internal network**: a dedicated Docker bridge network; only the dashboard (and optionally backend) are published to the host.

## 6. Deployment topology (Docker Compose)

| Container      | Published | Depends on            | Volume            |
|----------------|-----------|-----------------------|-------------------|
| postgres       | — (internal) | —                  | `pgdata`          |
| data-pipeline  | —         | postgres              | —                 |
| ai-service     | —         | postgres              | `models`          |
| backend        | 8080      | postgres, ai-service  | `secrets` (ro)    |
| dashboard      | 8081      | backend               | —                 |

See [`../infra/`](../infra/) and [`../docker-compose.yml`](../docker-compose.yml).

## 7. Cross-cutting concerns

- **Config**: 12-factor via env vars (`.env`), plus AES secrets file for sensitive values.
- **Logging**: structured JSON logs; a subset persisted to the `logs` table for the dashboard.
- **Observability**: health endpoints per service; metrics hooks reserved for Sprint 8.
- **Testing**: unit tests per service + integration tests against an ephemeral Postgres (see each sprint's DoD).

## 8. Explicit non-goals (v1)

- No real-money execution, wallet signing, or private-key custody.
- No multi-exchange support (Pump.fun only).
- No distributed training (single-node PyTorch).
