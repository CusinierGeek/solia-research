# ML model specification

## Problem framing

A **multi-task, multi-horizon time-series** model. Given a sequence of feature snapshots for a token, predict — **for each horizon** (30s, 1m, 5m, 15m, 1h):

1. **pump probability** `p_pump ∈ [0,1]`
2. **rug probability** `p_rug ∈ [0,1]`

and, as derived heads:

3. **strategy** `{buy, sell, hold}` (classification)
4. **position size** (regression, 0–1 fraction of budget)
5. **take-profit** multiplier (regression)
6. **stop-loss** multiplier (regression)
7. **confidence** `∈ [0,100]` (derived from calibrated head uncertainty)

## Architecture

- **Type**: Transformer encoder for time series (no decoder).
- **Input**: `[batch, seq_len, n_features]`. `seq_len` = rolling window of snapshots (e.g. 64); `n_features` ≈ the feature set in [`DATA-MODEL.md`](DATA-MODEL.md).
- **Blocks**:
  - Input projection (`Linear(n_features → d_model)`) + learned/positional encoding.
  - `N` × TransformerEncoderLayer (multi-head self-attention + FFN, pre-norm).
  - Masked mean / attention pooling over the sequence → context vector.
- **Heads** (per horizon): small MLPs
  - `pump_head`, `rug_head` → sigmoid
  - `strategy_head` → softmax(3)
  - `sizing_head`, `tp_head`, `sl_head` → bounded regression (sigmoid/softplus)
- Default hyperparams (tunable in `config.yaml`): `d_model=128`, `n_heads=8`, `n_layers=4`, `dropout=0.1`, `seq_len=64`.

## Losses

Total loss = weighted sum:

- **pump / rug**: BCE **+ Focal Loss** (class imbalance — pumps/rugs are rare).
  `L = α · BCE + (1-α) · Focal(γ)`
- **strategy**: cross-entropy.
- **sizing / TP / SL**: smooth L1 (Huber).
- Per-horizon weights and loss weights configurable.

## Optimizer & training

- **Optimizer**: AdamW (`lr`, `weight_decay` in config).
- **Scheduler**: cosine with warmup.
- **Regularization**: dropout, gradient clipping, early stopping on val AUC.
- **Mixed precision** optional.
- **Split**: temporal split (train on past, validate on more recent) to avoid leakage. **Never** shuffle across the time boundary within a token.

## Labeling

Labels are derived from realized outcomes:
- `pump` label at horizon `h` = price rose ≥ threshold within `h`.
- `rug` label = liquidity/price collapsed ≥ threshold within `h`.
- Strategy/sizing/TP/SL targets derived from closed **paper trades** (`trades` table) — this is what makes continuous learning work.

Thresholds are defined in `config.yaml` and documented alongside the dataset builder (Sprint 2).

## Confidence

Confidence (0–100) is derived from head agreement + calibrated probability margin (e.g. temperature-scaled). Precise formula finalized in Sprint 2; must be **monotonic** with historical hit-rate.

## Outputs contract

The inference response (see [`API-CONTRACTS.md`](API-CONTRACTS.md)) returns, per requested horizon, all seven outputs plus the `model_version` used.

## Continuous training loop

1. Count new **closed** trades since last training (`training_events`).
2. When `>= RETRAIN_EVERY_N_TRADES`: build an updated dataset (historical + live trades).
3. Train a **candidate** from the champion checkpoint (warm start) or from scratch (configurable).
4. Evaluate candidate vs champion on a held-out recent window.
5. **Promote** candidate to champion only if it beats the champion by a margin on the primary metric (e.g. val AUC / expected P&L); otherwise **reject** / **rollback**.
6. Record everything in `training_events` and `models`; log to `logs`.

Checkpoints, versioning, comparison, and rollback are first-class — see Sprint 7.

## Files (ai-service, built in Sprints 2 & 7)

| File | Role |
|------|------|
| `model.py` | Transformer + heads |
| `dataset.py` | Dataset/DataLoader, windowing, labeling |
| `features.py` | Shared feature definitions / parity with pipeline |
| `train.py` | Initial training entrypoint |
| `inference.py` | Load champion, run predictions |
| `retrain.py` | Continuous-training orchestrator |
| `api.py` | FastAPI app (`/predict`, `/model/status`, `/model/retrain`) |
| `config.yaml` | All hyperparameters & thresholds |

## Evaluation metrics

- pump/rug: ROC-AUC, PR-AUC, precision/recall at operating threshold, calibration (ECE).
- strategy: accuracy, confusion matrix.
- economic: simulated win-rate and expected P&L on the validation window (the metric that actually matters).
