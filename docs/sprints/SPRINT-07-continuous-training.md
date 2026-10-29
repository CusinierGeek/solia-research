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
- [x] Trade counter since last training (query `training_events` + `trades`) — `retrain.count_trades_since_last_train` (#133), same rule as the backend.
- [x] Trigger (scheduler/background task) at threshold + manual endpoint — `retrain_service.py` (#135).
  ✅ Resolved by migration `0010` (#130): the CHECK now allows `manual`, `decision` gains `failed` for a crashed/errored run, and `decision` / `finished_at` must be set together. Only `promoted` / `rejected` reset the trades-since-last-train count (`failed` and `rollback` trained nothing).
- [x] Dataset builder refresh incorporating live trades — each run re-reads `features` (historical replay #117 + live) with a refreshed time split (#132). Closed paper trades trigger the retrain; they are never labels.
- [x] Candidate training run (warm start option) — `candidate.py` (#132).
- [x] Evaluation harness: candidate vs champion, primary metric + margin — walk-forward windows × seeds, guard rails, P&L veto (`promotion.py`, #131; `retrain.train_and_evaluate`, #133).
- [x] Promotion logic (set `is_champion`, flip old champion) + rollback path (#131, #133).
- [x] Record `training_events` (trigger, samples, metrics, decision) + new `models` row (#130, #133).
- [x] Checkpoint retention/cleanup policy — `retention.py` (#134).
- [x] Logs + dashboard surfacing via `/model/status` — `logs` rows per step; `trades_since_last_train`, `retrain_threshold`, `latest_run` (#135).
- [x] Tests: threshold logic, promote/reject/rollback on synthetic metrics, plus the end-to-end test `tests/test_continuous_training_e2e.py` (#136).

## Deliverables
- Working retrain orchestrator (auto + manual).
- Versioned models with promote/rollback.
- Tests for the decision logic.

## Definition of Done
- [x] After N simulated closed trades, a retrain runs automatically.
- [x] A better candidate is promoted; a worse one is rejected; rollback restores the prior champion.
- [x] `/model/status` reflects version, metrics, trades-since-train.
- [x] Every run is recorded in `training_events`.
- [x] Tests green in CI (the end-to-end test runs in the Python job, against Testcontainers PostgreSQL).

## Kickoff prompt
> We're implementing **Sprint 7 — Continuous training** of Solia. Read `docs/ML-MODEL.md`, `docs/DATA-MODEL.md`, `docs/API-CONTRACTS.md` and the Sprint 2 ai-service. Build `retrain.py`: count new closed trades, trigger at `RETRAIN_EVERY_N_TRADES` (plus a manual endpoint), refresh the dataset with live trades, train a candidate (warm-start option), evaluate candidate vs champion on a recent held-out window, then promote/reject/rollback and record `models` + `training_events`. Wire `/model/retrain` and `/model/status`. Add checkpoint retention and tests for the decision logic on synthetic metrics. Branch `sprint/07-continuous-training`, PR per `CONTRIBUTING.md`.
