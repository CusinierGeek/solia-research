# Solia

> A research system that tried to find a tradeable edge on Pump.fun tokens with machine learning, using
> **paper trading only**. The answer turned out to be **no**, and this repository documents how that was measured.

**Status: concluded (November 2026).** The system runs end to end: live ingestion, a model service, a paper-trading bot,
continuous training and a dashboard. Its research question is answered. No further strategy work is planned.

> ⚠️ **Disclaimer.** Research and educational project. **Paper trading only**: simulated orders, no real funds, no wallet
> signing, no on-chain execution. Nothing here is financial advice. Pump.fun memecoins are extremely high risk, and the
> findings below show why.

---

## Findings

The full write-up, with numbers and links to every report, is in [`docs/FINDINGS.md`](docs/FINDINGS.md). In short:

| What was tried | Result after costs, out of sample |
|---|---|
| Transformer predicting pumps and rugs | High AUC, but the "pump" signal measures volatility, not direction |
| Paper bot on the model's best setting | +17 % per trade in backtests, then +2 %, +15 % and −13 % on later data; about zero net over two paper sessions |
| Rules on tokens whose liquidity is burned | Lose −6 to −8 % per trade on every period |
| Explainable rug-risk score (LightGBM + SHAP) | Better than the usual "LP burned" rule, but 7.7 % rugs remain in the "safe" set |
| Holding established tokens (1–6 h old) | −28 % on average after 24 h; most losses are sudden |
| Copying persistently winning wallets | Winners persist, but copies lose −9 % to −18 % per trade |

**Why:** Pump.fun is a **negative-sum market**: closed positions lost 66–170 k SOL per period studied. The consistent
winners profit from **speed**: they buy in the token's creation block and sell within seconds. Many pools are controlled
by their creator, who drains them **in a single transaction**. A model acting a few seconds later on public data ends up on
the wrong side of both.

**How results were judged:**
- Real costs: fees, slippage on the pool's reserves, latency.
- Settings chosen on one period and confirmed unchanged on a later one.
- Confidence intervals resampled by token.
- For the last studies, a pass/fail criterion committed to git **before** the test data was read.

## What it does

```
pumpapi.io WS ─▶ data-pipeline (.NET) ─▶ PostgreSQL ─▶ ai-service (FastAPI + PyTorch)
                                             │                 │
                                   backend (ASP.NET Core) ◀────┘
                                   ├─ paper-trading bot
                                   └─ dashboard (Blazor Server)
```

- **data-pipeline**: ingests the free pumpapi.io stream and its hourly replay archives through the same code path, and
  writes 10-second feature snapshots per token.
- **ai-service**: multi-horizon Transformer (pump/rug probability, expected return), training, inference, and a
  continuous-training loop with a champion/challenger gate. It also holds the research tools: walk-forward ablations,
  `exit_sim` (bot-faithful exit simulation), `model_value`, `rug_guard`, `survivor_study` and `wallet_study`.
- **backend**: REST API, AES-encrypted secrets, and the paper-trading bot.
- **dashboard**: live predictions, trades, P&L and logs.

| Path | Stack |
|------|-------|
| `services/data-pipeline` | .NET 10 worker |
| `services/ai-service` | Python 3.11, PyTorch, FastAPI, LightGBM |
| `services/backend` | ASP.NET Core (.NET 10) |
| `services/dashboard` | Blazor Server (.NET 10) |
| `services/db` | PostgreSQL 16, raw SQL migrations |
| `infra/` | Docker / Compose |

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | **What was tested and what it showed** |
| [`docs/ML-MODEL.md`](docs/ML-MODEL.md) | Model, labels, evaluation, and every study in detail |
| [`docs/reports/`](docs/reports/) | Raw reports and the scripts that produced them |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, deployment |
| [`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md) | Ingestion, replay, feature engineering |
| [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) | Data providers and cost strategy |
| [`docs/DATA-MODEL.md`](docs/DATA-MODEL.md) | PostgreSQL tables |
| [`docs/API-CONTRACTS.md`](docs/API-CONTRACTS.md) | REST contracts between services |
| [`docs/SECRETS.md`](docs/SECRETS.md) | AES secrets handling |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | How the project was organized, and where it ended |

## Quick start

Requires Docker Engine 25+ and Docker Compose 2.24+.

```bash
cp .env.example .env      # works as-is; change POSTGRES_PASSWORD for anything shared
docker compose up -d      # builds the images on first run, then starts the stack
                          # dashboard: http://localhost:8081, REST API: http://localhost:8080
```

The paper bot is off by default (`PAPER_BOT_ENABLED=false`). Encrypted secrets are optional; see
[`docs/SECRETS.md`](docs/SECRETS.md#docker-compose). `make up | down | logs | ps | reset | smoke` wrap the usual compose
commands. Without `make`, run `infra/scripts/solia.sh <command>` from Git Bash.

**Historical data.** The model trains on history replayed from pumpapi's free hourly archives, through the same feature
path as live:

```bash
docker compose run --rm data-pipeline replay --from 2026-11-01T00:00Z --to 2026-11-01T03:00Z
```

Add `--mirror /replay-archives` to keep the archives (~400 MB per hour). Details:
[`docs/DATA-PIPELINE.md`](docs/DATA-PIPELINE.md#historical-replay-training-data).

**Tests.** `dotnet test` (add `--filter "Category!=Integration"` to skip Testcontainers) and, in `services/ai-service`,
`pytest`.

## About this repository

This is the public copy of a private research repository. Issue and PR numbers in commit messages and docs (`#260`,
`#271`, …) refer to that private tracker. The code is released under the [MIT license](LICENSE): use it freely, keeping
in mind it was written to answer a research question, not as a product.
