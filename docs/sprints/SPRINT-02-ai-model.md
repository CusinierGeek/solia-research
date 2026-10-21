# Sprint 2 — AI model (Transformer)

**Milestone:** AI model · **Depends on:** Sprint 1 · **Service:** `services/ai-service`

## Goal
Build the Transformer, a dataset/labeling pipeline over the Sprint-1 data, train an initial model offline, and run inference.

## Scope
- `model.py`, `dataset.py`, `features.py`, `train.py`, `inference.py`, `config.yaml`.
- Multi-task, multi-horizon heads per [`../ML-MODEL.md`](../ML-MODEL.md).
- BCE + Focal loss, AdamW, temporal split, checkpointing.
- Offline evaluation (ROC/PR-AUC, calibration, simulated P&L).

## Out of scope
- FastAPI serving endpoints beyond a minimal inference entrypoint (full API shared with Sprint 4 integration).
- Continuous retraining (Sprint 7).

## Task breakdown
- [ ] Project skeleton (deps: torch, numpy, pandas, pyyaml, scikit-learn; ruff/black/pytest).
- [ ] `features.py`: feature spec mirrored from the pipeline (+ parity test).
- [ ] `dataset.py`: windowing, labeling (pump/rug thresholds), temporal split, DataLoader.
- [ ] `model.py`: input projection + positional encoding + encoder stack + pooling + heads.
- [ ] Loss module (BCE + Focal + Huber, weighted).
- [ ] `train.py`: AdamW, cosine+warmup, grad clip, early stop, checkpoint + `models` registry row.
- [ ] `inference.py`: load champion, produce the prediction contract output.
- [ ] Confidence calibration.
- [ ] Evaluation script + metrics report.
- [ ] `config.yaml` with all hyperparameters/thresholds.

## Deliverables
- Trained initial checkpoint + a `models` row (`is_champion=true`).
- Metrics report (AUC, calibration, simulated P&L).
- Unit tests (forward pass shapes, loss, dataset windowing) + parity test with pipeline features.

## Definition of Done
- [ ] `python train.py` trains end-to-end on Sprint-1 data and writes a checkpoint.
- [ ] `python inference.py --mint ...` returns the full output contract.
- [ ] Metrics beat a naive baseline; calibration reasonable.
- [ ] Tests green in CI.

## Kickoff prompt
> We're implementing **Sprint 2 — AI model** of Solia. Read `docs/ML-MODEL.md`, `docs/DATA-MODEL.md`, `docs/DATA-SOURCES.md`, `docs/DATA-PIPELINE.md`, `docs/API-CONTRACTS.md`. Train on the pipeline's PostgreSQL data, backfilled from the pumpapi historical replay window (pre/post-bonding). Build the `ai-service` Python package: `features.py` (mirroring the pipeline, with a parity test), `dataset.py` (windowing + pump/rug labeling + temporal split), `model.py` (time-series Transformer with multi-task multi-horizon heads), a weighted BCE+Focal+Huber loss, `train.py` (AdamW, cosine warmup, checkpointing + models-registry row), `inference.py`, and `config.yaml`. Train an initial model on Sprint-1 data and produce an evaluation report (ROC/PR-AUC, calibration, simulated P&L). Branch `sprint/02-ai-model`, PR per `CONTRIBUTING.md`. Confirm the labeling thresholds with me first.
