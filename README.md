# Solia

> Predictive trading research system for Pump.fun tokens — ML pump/rug probability, autonomous **paper-trading** bot, continuous training, and a live dashboard.

**Status:** 🏗️ Planning / scaffolding. No application code yet — this repository currently contains the architecture, data model, and the sprint backlog. Each sprint is implemented in its own branch/PR.

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

## Quick start (once services exist)

```bash
cp .env.example .env    # fill in values
docker compose up -d    # postgres + services
```

Nothing is runnable yet — follow the sprints in order.
