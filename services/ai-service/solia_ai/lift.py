"""Paired-lift primitives shared by the ablation harness, the search and the promotion gate.

Kept torch-free (numpy only) so :mod:`solia_ai.promotion` can use them without importing training.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

PREDICTIVE_METRICS = ("roc_auc", "pr_auc", "ece")
# Calibration error: lower is better, so its lift is negated.
LOWER_IS_BETTER = frozenset({"ece"})


def lift(value: float, baseline: float, metric: str) -> float:
    """``value`` − ``baseline`` oriented so positive is better (``metric`` may be a full
    ``<head>/<horizon>/<metric>`` key)."""
    sign = -1.0 if metric.rsplit("/", 1)[-1] in LOWER_IS_BETTER else 1.0
    return sign * (value - baseline)


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    """Mean ± standard deviation over the defined values (``None`` ones are skipped; the spread of
    fewer than two values is undefined)."""
    defined = np.array([v for v in values if v is not None], dtype=np.float64)
    n = int(defined.size)
    return {
        "n": n,
        "mean": float(defined.mean()) if n else None,
        "std": float(defined.std(ddof=1)) if n > 1 else None,
    }
