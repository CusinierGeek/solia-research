# Roadmap — 8 sprints and two research phases (concluded)

> **Status (November 2026): concluded.** Sprints 1–8 and the feature-enrichment phase were delivered. The research
> phase that followed ("Rug guard & established tokens", then the winning-wallets studies) found no tradeable edge, so
> the remaining strategy tasks were closed and the planned live-trading sprint was cancelled. See
> [`FINDINGS.md`](FINDINGS.md).

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

## Future phase — Feature enrichment & tuning

After Sprint 2 ships a trained baseline **and** the evaluation harness, a dedicated phase enriches
the model's inputs and tunes it — each change justified by **measured lift (ablation)**, never
added blind. On this problem, feature quality beats model complexity. Candidates (prefer free,
stream-derivable signals): price/volume **acceleration**, buy pressure by SOL volume (net flow),
holder **concentration proxy** from traders, promoting `burnedLiquidity` to a column, `venue`
one-hot; plus hyperparameter/threshold tuning and probability calibration.

Tracked by milestone **"Feature enrichment & tuning"** (epic #34). Not a numbered sprint — runs
when the baseline exists and we can measure improvements.

## Research phase — Rug guard & established tokens, winning wallets

After the paper bot's setting failed on new data (November 2026), the work moved away from racing pool drains: an
explainable rug-risk score (#260), a study of established tokens (#261), and two studies of winning wallets and copy
trading (#269, #271). Each was judged against a criterion fixed in advance; none passed. Results:
[`FINDINGS.md`](FINDINGS.md).

## Cancelled — Live trading

A gated live-trading sprint was planned, conditional on a paper edge that is "demonstrably not luck". That gate was never
met, so the sprint was cancelled and its plan removed. The paper-only rule stands.
