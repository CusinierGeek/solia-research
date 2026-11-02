#!/bin/sh
# #172 stage 2 (regularization) on the groups #188 enabled, at 8M rows / stride 60 (resumable: rerun this script).
set -e
R=/app/models/reports
[ -f $R/overfitting-regularization.config.yaml ] && [ ! -f $R/overfitting-regularization.config.night.yaml ] && \
  mv $R/overfitting-regularization.config.yaml $R/overfitting-regularization.config.night.yaml
python - <<'EOF'
import yaml
from solia_ai.config import load_config
from solia_ai.overnight import stage2_config
cfg = stage2_config(load_config(), {"window_stride_seconds": 60, "max_rows": 8_000_000})
open("/app/models/reports/overfitting-regularization.config.yaml", "w").write(yaml.safe_dump(cfg, sort_keys=False))
print("stage 2 config:", cfg["features"], cfg["data"], flush=True)
EOF
export SOLIA_AI_CONFIG=$R/overfitting-regularization.config.yaml
exec python -u -m solia_ai.search --space overfitting-regularization --strategy random --trials 10 \
  --windows 2 --seeds 0 1 --objective pr_auc \
  --out $R/overfitting-regularization.json --diff-out $R/overfitting-regularization.diff
