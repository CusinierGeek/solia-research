"""#188 night run: ablation of the five computed-but-disabled groups on the bounded training read.

The ablation CLI reads the whole table (no lookback / max_rows), so this driver reuses the
training read and run_ablation directly. Data settings: stage 1's choice when it beat the
baseline, else stride 60 s / 8 M rows (stage 1: +0.020 pump/5m PR-AUC vs baseline, just under the margin, at half its cost).
"""

import json
import sys
from pathlib import Path

import torch
import yaml

from solia_ai.ablation import Variant, plan_variants, run_ablation
from solia_ai.config import load_config, validate_config
from solia_ai.db import DataSettings, make_engine, read_training_features
from solia_ai.features import BASELINE_GROUPS
from solia_ai.overnight import stage1_choice

REPORTS = Path("/app/models/reports")
GROUPS = ["buy_pressure", "momentum", "concentration", "burn", "venue"]

base = load_config()
choice = None
stage1 = REPORTS / "overfitting-data.json"
if stage1.is_file():
    report = json.loads(stage1.read_text(encoding="utf-8"))
    picked = stage1_choice(report, base)
    if picked != {k: base["data"][k] for k in picked}:
        choice = picked
        print(f"stage 1 winner: {choice}", file=sys.stderr)
if choice is None:
    choice = {"window_stride_seconds": 60, "max_rows": 8_000_000}
    print(f"stage 1 had no winner: using {choice}", file=sys.stderr)

config = {**base, "data": {**base["data"], **choice}}
validate_config(config, "ablation-188 config")
(REPORTS / "ablation-188-confirm.config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

variants = plan_variants(["momentum"])
variants.append(Variant("+momentum+concentration", tuple(BASELINE_GROUPS) + ("momentum", "concentration"), "momentum+concentration", "add"))
print("variants: " + ", ".join(v.name for v in variants), file=sys.stderr)

df, bounds = read_training_features(make_engine(), DataSettings.from_config(config))
print(bounds.describe(), file=sys.stderr)
result = run_ablation(
    df,
    config,
    variants=variants,
    n_windows=3,
    seeds=[0, 1, 2],
    device="cuda" if torch.cuda.is_available() else "cpu",
    progress=lambda msg: print(msg, file=sys.stderr, flush=True),
)
print(result.render())
print(f"report written to {result.write(REPORTS / 'ablation-188-confirm.json')}")
