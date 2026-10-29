# ai-service (Python 3.11 · PyTorch · FastAPI)

The Transformer model: dataset building, training, inference (+ ONNX export for the .NET fast
path), an internal prediction API, and continuous retraining (Sprint 7).

- **Design:** [ML-MODEL.md](../../docs/ML-MODEL.md) · [API-CONTRACTS.md](../../docs/API-CONTRACTS.md) · [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) (§5 inference paths)
- **Sprint:** [Sprint 2](../../docs/sprints/SPRINT-02-ai-model.md)

## Status

Sprint 2 complete: feature spec, dataset/labeling, Transformer model, training + checkpoints,
inference + ONNX export, evaluation, and the internal prediction API. Sprint 7 adds continuous
retraining (`retrain.py` + `/model/retrain` + the background trigger, see below).

## Dev setup

```bash
cd services/ai-service
python -m venv .venv && . .venv/Scripts/activate   # or .venv/bin/activate on Linux/macOS
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

ruff check .
black --check .
pytest
```

Python **3.11**. CPU PyTorch is fine for development. Dependencies are pinned exactly in
`pyproject.toml` (the versions the suite is tested with); bump them deliberately. Run the tools
from this directory so they pick up its `pyproject.toml` settings — CI does the same.

### GPU (NVIDIA) image

The default image is CPU-only (`requirements.lock`, torch `+cpu`). On a host with an NVIDIA GPU
(driver with CUDA 13.0 support; Docker Desktop's WSL 2 backend or the NVIDIA Container Toolkit), the
`docker-compose.gpu.yml` override builds the ai-service with `TORCH_VARIANT=cu130`
(`requirements-cu130.lock`, tagged `solia-ai-service-cu130`) and reserves the GPUs for it:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

Nothing else changes: training — `python -m solia_ai.retrain` and the background trigger alike —
picks `cuda` whenever torch sees a GPU (`--device` overrides it). Only the model runs on the GPU;
reading `features` and building the dataset still use host RAM. `requirements-cu130.lock` is the
CPU lock with torch swapped for its CUDA build (regenerate it after `requirements.lock`, see its
header); `tests/test_requirements_lock.py` keeps the two in step.

## Configuration

- `solia_ai/config.yaml` — the single source of model hyperparameters, training settings, loss
  weights (`loss.alpha` for `L = α·BCE + (1-α)·Focal`), prediction horizons, and labeling
  thresholds (see `docs/ML-MODEL.md`). It ships inside the package, so installed copies find it;
  point `SOLIA_AI_CONFIG` at another file to override it. `load_config()` validates it and raises
  `ConfigError` naming the file and the offending key.
- **Database** comes from the environment, not the YAML, with the same contract as the .NET
  services: `SOLIA_DB_CONNECTION` (an Npgsql connection string, `Host=...;Username=...;...`)
  wins, otherwise `POSTGRES_*` (see `.env.example`). Blank values count as unset and credentials
  are escaped. The local-dev default password is only used for a local host; a remote database
  without `POSTGRES_PASSWORD` fails at startup. Built by `database_url()` in `solia_ai/config.py`.

## Feature spec & DB access

- `solia_ai/features.py` — `FEATURE_COLUMNS` is the ordered model input spec. Two parity tests
  keep it in lockstep with the `features` table: one replays `CREATE TABLE` / `ALTER TABLE` across
  **every** migration (any column type), the other reads `information_schema` from a migrated
  database. `build_feature_matrix` imputes NULLs and non-finite values to `0.0`, encodes
  `is_migrated` as `0/1`, and raises on a missing column or a non-numeric value.
- `solia_ai/db.py` — read-only `read_features` / `read_trades` into pandas, ordered by
  `(mint, time)`. Time bounds are half-open and must be timezone-aware (normalized to UTC);
  `chunksize` streams through a server-side cursor so memory stays bounded.
- The database tests run against Testcontainers PostgreSQL. Locally they are skipped when Docker
  isn't running; in CI (`CI` set) a missing Docker fails the build instead.

## Dataset & labeling

`solia_ai/dataset.py` slides `seq_len` windows over each token's `features` rows (windows are
views over rows stored once, so memory doesn't grow with `seq_len`) and labels each horizon from
the realized future:

- **pump** — the *highest* price in the horizon rose by ≥ `pump_threshold`;
- **rug** — the *lowest* price **or** liquidity fell by ≥ `rug_threshold`;
- **expected_return** — the price at the horizon end vs. the anchor.

Pump and rug are independent heads (a violent swing can be both). A horizon is masked when the
data doesn't cover it or the last row inside it is stale (`max(max_gap_fraction·H,
min_gap_seconds)`), except when a rug has already happened in the observed part — that's how a
token that collapses and stops trading still gets its rug label. The temporal split embargoes the
longest horizon; `positive_weights` (BCE `pos_weight`), `balanced_sampler` and `make_loaders`
handle the class imbalance.

## Model

`solia_ai/model.py` — `SoliaTransformer`: a pre-norm Transformer encoder over a token's recent
feature rows, with per-horizon `pump` / `rug` (logits) and `expected_return` heads.

- **Input normalization is part of the model**: signed `log1p`, then per-feature standardization
  fitted on the training rows (`model.fit_normalizer(rows)`), stored as buffers so checkpoints and
  the ONNX export apply the same transform.
- **Young tokens**: sequences of `min_seq_len .. seq_len` rows, left-padded with a padding mask
  (`dataset.pad_window`); attention and the masked-mean pooling ignore padding.
- **Checkpoints carry their horizons and feature spec** and refuse to load into a mismatched model.
- `predict()` returns probabilities and restores the caller's train/eval mode.
- Shape knobs (`d_model`, `n_heads`, `n_layers`, `ff_dim`, `dropout`, `seq_len`, `min_seq_len`)
  come from `config.yaml`.

## Evaluation

`solia_ai/evaluate.py` scores a checkpoint over the held-out validation window and reports, per
horizon (on labeled samples only): **pump / rug** ROC-AUC, PR-AUC and calibration error (ECE);
**expected_return** information coefficient (Spearman rank IC), Pearson correlation and MAE; and a
simulated **win-rate / P&L** from a deliberately simple `ReferencePolicy` (enter when `pump_prob`
is high and `rug_prob` low). That policy is a *yardstick* for the signal only — the real, tunable
policy lives in the bot (Sprints 3–4). AUCs/correlations are `null` when undefined (a single class
or a constant series), and the JSON report nulls any non-finite metric.

```bash
python -m solia_ai.evaluate --champion --out report.json   # or --checkpoint path/to/model.pt
```

## Prediction API

`solia_ai/api.py` — the internal FastAPI the backend calls (`docs/API-CONTRACTS.md`):

- `POST /predict` — **stateless**: the caller sends the token's recent feature window (objects
  keyed by feature name, or arrays in `FEATURE_COLUMNS` order) and gets back, per requested
  horizon, `pump_prob` / `rug_prob` / `expected_return` / `confidence` plus the `model_version`.
- `GET /model/status` — the loaded champion's version, parameter count and training metrics.
- `GET /health` — liveness (never touches the database).

The champion is loaded lazily and cached, so the process starts before a model is promoted;
`/predict` and `/model/status` return `503` until one exists. Errors use the shared
`{"error": {"code", "message"}}` shape.

```bash
uvicorn solia_ai.api:app --port 8000
```

## Continuous retraining

`retrain.py` is the orchestrator (`python -m solia_ai.retrain`, see `docs/ML-MODEL.md`); inside the API,
`retrain_service.py` serves `POST /model/retrain` (`{"trigger": "manual"}` → `202 {"training_event_id"}`,
`409` while one runs) and the background trigger: `RETRAIN_SCHEDULER_ENABLED` (default `false`; `true` in
compose), `RETRAIN_CHECK_INTERVAL_SECONDS` (default 300), `RETRAIN_EVERY_N_TRADES` (default 500).
