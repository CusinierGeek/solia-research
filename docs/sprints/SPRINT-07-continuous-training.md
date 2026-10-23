# Sprint 7 — Continuous training

**Milestone:** Continuous training · **Depends on:** Sprints 2, 3, 6 · **Service:** `services/ai-service`

## Goal
Automatically retrain after N new closed trades, version models, compare candidate vs champion, and promote or roll back.

## Scope
- `retrain.py` orchestrator + trigger on `RETRAIN_EVERY_N_TRADES`.
- Dataset refresh (historical + new live trades).
- Candidate training (warm-start from champion, configurable).
- Evaluation vs champion on a recent held-out window.
- Promote/reject/rollback decision + `models`/`training_events` updates.
- `/model/retrain` and `/model/status` wired through backend.
- Checkpoint management + detailed logs.

## Out of scope
- Distributed training. Hyperparameter search (optional stretch).

## Task breakdown
- [ ] Trade counter since last training (query `training_events` + `trades`).
- [ ] Trigger (scheduler/background task) at threshold + manual endpoint.
  ⚠️ The backend proxies `POST /model/retrain` with `{ "trigger": "manual" }`, but the `training_events.trigger` CHECK only allows `initial` / `scheduled` / `n_trades`: add `manual` (migration `0003`) or map it before inserting.
- [ ] Dataset builder refresh incorporating live trades.
- [ ] Candidate training run (warm start option).
- [ ] Evaluation harness: candidate vs champion, primary metric + margin.
- [ ] Promotion logic (set `is_champion`, flip old champion) + rollback path.
- [ ] Record `training_events` (trigger, samples, metrics, decision) + new `models` row.
- [ ] Checkpoint retention/cleanup policy.
- [ ] Logs + dashboard surfacing via `/model/status`.
- [ ] Tests: threshold logic, promote/reject/rollback on synthetic metrics.

## Deliverables
- Working retrain orchestrator (auto + manual).
- Versioned models with promote/rollback.
- Tests for the decision logic.

## Definition of Done
- [ ] After N simulated closed trades, a retrain runs automatically.
- [ ] A better candidate is promoted; a worse one is rejected; rollback restores the prior champion.
- [ ] `/model/status` reflects version, metrics, trades-since-train.
- [ ] Every run is recorded in `training_events`.
- [ ] Tests green in CI.

## Kickoff prompt
> We're implementing **Sprint 7 — Continuous training** of Solia. Read `docs/ML-MODEL.md`, `docs/DATA-MODEL.md`, `docs/API-CONTRACTS.md` and the Sprint 2 ai-service. Build `retrain.py`: count new closed trades, trigger at `RETRAIN_EVERY_N_TRADES` (plus a manual endpoint), refresh the dataset with live trades, train a candidate (warm-start option), evaluate candidate vs champion on a recent held-out window, then promote/reject/rollback and record `models` + `training_events`. Wire `/model/retrain` and `/model/status`. Add checkpoint retention and tests for the decision logic on synthetic metrics. Branch `sprint/07-continuous-training`, PR per `CONTRIBUTING.md`.
