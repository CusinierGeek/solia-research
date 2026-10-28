# Solia

> Predictive trading research system for Pump.fun tokens — ML pump/rug probability, autonomous **paper-trading** bot, continuous training, and a live dashboard.

**Status:** 🧪 Runs end-to-end in Docker Compose (Sprints 1–6): live ingestion, model service, backend + paper bot, dashboard. Next: continuous training, then optimization & QA — see [`docs/ROADMAP.md`](docs/ROADMAP.md). Each task is implemented in its own branch/PR.

> ⚠️ **Disclaimer.** Solia is a **research and educational** project. It performs **paper trading only** (simulated orders, no real funds, no on-chain execution). Nothing here is financial advice. Crypto markets — and Pump.fun memecoins in particular — are extremely high risk.

---

## What it does

Solia ingests live Pump.fun market data, builds time-series features, and uses a Transformer model to estimate, per token and per time horizon:

- **pump probability** (0–1)
- **rug probability** (0–1)
- a **recommended strategy** (buy / sell / hold) with **position sizing**, **dynamic take-profit**, **dynamic stop-loss**, and a **confidence score** (0–100%)

A paper-trading bot acts on those recommendations, records every trade, and feeds the results back into a continuous-training loop that re-trains and versions the model once enough new data has accumulated.

Prediction horizons: **30s, 1m, 5m, 15m, 1h**.

---

## Architecture at a glance

```
pumpapi.io WS ──▶ data-pipeline (.NET) ─▶ PostgreSQL ──▶ ai-service (FastAPI + PyTorch)
                                             │                    ▲   │
                                             │                    │   └─▶ predictions
                                             ▼                    │
                                    backend (ASP.NET Core) ◀───────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                                          ▼
                 dashboard (Blazor Server)                paper-trading bot
```

Full detail: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Services

| Path                     | Stack                         | Role |
|--------------------------|-------------------------------|------|
| `services/data-pipeline` | .NET 10 worker                | pumpapi.io WS + historical replay ingestion, feature engineering, DB writes |
| `services/ai-service`    | Python 3.11 + PyTorch + FastAPI | Transformer model, training, inference API, retrain orchestrator |
| `services/backend`       | ASP.NET Core (.NET 10)        | REST API, orchestration, encrypted secrets, paper-trading bot host |
| `services/dashboard`     | Blazor Server (.NET 10)       | Live UI: predictions, trades, win-rate, P&L, logs |
| `services/db`            | PostgreSQL 16                 | Schema + migrations |
| `infra/`                 | Docker / Compose              | Containers, internal network, persistent volumes |

## Tech choices

- **.NET 10** (backend, dashboard, data-pipeline), **Python 3.11** (ai-service only), **PostgreSQL 16**, **PyTorch**, **FastAPI**, **Blazor Server**
- Data source: **pumpapi.io** free WebSocket + free historical replay, behind a swappable `DataSource` abstraction (fallbacks: PumpPortal, Solana RPC+IDL). See [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md)
- Secrets: **AES-encrypted file** (see [`docs/SECRETS.md`](docs/SECRETS.md))

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, deployment topology |
| [`docs/DATA-MODEL.md`](docs/DATA-MODEL.md)     | PostgreSQL tables & relationships |
| [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) | Providers compared, cost strategy, pumpapi decision |
| [`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md) | Ingestion, feature engineering |
| [`docs/ML-MODEL.md`](docs/ML-MODEL.md)         | Transformer spec, losses, outputs, training loop |
| [`docs/API-CONTRACTS.md`](docs/API-CONTRACTS.md) | REST contracts between services |
| [`docs/SECRETS.md`](docs/SECRETS.md)           | AES secrets handling |
| [`docs/ROADMAP.md`](docs/ROADMAP.md)           | The 8 sprints overview |
| [`docs/sprints/`](docs/sprints/)               | One detailed backlog file per sprint |

## How we work

Each sprint = one milestone = one (or more) PR, reviewed before merge. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the branch/PR/review process. Each sprint file ends with a **"Kickoff prompt"** you can paste into a fresh conversation to implement that sprint.

## Quick start

Requires Docker Engine 25+ and Docker Compose 2.24+. From a fresh clone:

```bash
cp .env.example .env      # 1. config — works as-is; change POSTGRES_PASSWORD for anything shared
                          # 2. (optional) encrypted secrets — see below
docker compose up -d      # 3. builds the images on first run, then starts the whole stack
                          # 4. open http://localhost:8081 (dashboard); the REST API is on :8080
```

The first start builds every image (a few minutes; the ai-service image pulls PyTorch). Startup is
ordered — postgres → `db-migrate` (applies the schema, then exits) → data-pipeline + ai-service →
backend (waits for ai-service, not the pipeline) → dashboard, so the dashboard answers once the
backend is healthy; `make ps` shows where a slow start is. The paper bot
is off by default (`PAPER_BOT_ENABLED=false` in `.env`).

**(Optional) secrets.** Without `secrets/secrets.enc` the stack runs on the plaintext `.env`. To keep
secrets encrypted: create the file and the master key with `Solia.Secrets.Cli` on the host, put it in
`./secrets/` (readable by UID 1654), and inject `SOLIA_MASTER_KEY` (host env, or `.env`) — step by
step in [`docs/SECRETS.md` § Docker](docs/SECRETS.md#docker-compose).

Day-to-day:

| Shortcut                    | Plain compose                                  | What it does |
|-----------------------------|------------------------------------------------|--------------|
| `make up`                   | `docker compose up -d --build --wait`          | (re)build and start, wait until healthy |
| `make down`                 | `docker compose down`                          | stop; DB and models volumes are **kept** |
| `make logs` (`ARGS=backend`)| `docker compose logs -f --tail=100 [service]`  | follow the logs |
| `make ps`                   | `docker compose ps -a`                         | container status (incl. the exited `db-migrate`) |
| `make reset`                | `docker compose down -v --remove-orphans`      | stop **and delete** the volumes (DB data, trained models) — asks first |
| `make smoke`                | `infra/scripts/smoke.sh`                       | end-to-end smoke test, in its own compose project |
| `make dev-db`               | `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d postgres` | Postgres alone on `127.0.0.1:5432` for `dotnet run` / pytest |

No `make` (Windows)? The targets are thin wrappers: run the same commands from Git Bash with
`infra/scripts/solia.sh <command>` (e.g. `infra/scripts/solia.sh logs backend`,
`infra/scripts/solia.sh reset --yes`), or use the plain compose column.

How the containers fit together (images, startup order, volumes, network, ports):
[`infra/README.md`](infra/README.md) and [`docs/ARCHITECTURE.md` §7](docs/ARCHITECTURE.md#7-deployment-topology-docker-compose).
