# ai-service (Python 3.11 · PyTorch · FastAPI)

The Transformer model: dataset building, training, inference (+ ONNX export for the .NET fast
path), and an internal prediction API. Continuous retraining lands in Sprint 7.

- **Design:** [ML-MODEL.md](../../docs/ML-MODEL.md) · [API-CONTRACTS.md](../../docs/API-CONTRACTS.md) · [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) (§5 inference paths)
- **Sprint:** [Sprint 2](../../docs/sprints/SPRINT-02-ai-model.md)

## Status

Scaffold (T1): package, pinned deps, `config.yaml`, lint/format/test tooling. Dataset, model,
training, inference/ONNX and the API land in T2–T7.

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

## Planned modules (Sprint 2)

`features.py` (DB read + feature-spec parity), `dataset.py` (windowing + labeling),
`model.py` (Transformer), `train.py` (training + checkpoints), `inference.py` (+ ONNX export),
`api.py` (FastAPI). `retrain.py` (continuous training) arrives in Sprint 7.
