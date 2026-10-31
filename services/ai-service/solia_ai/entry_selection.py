"""Entry selection: the windows the bot would buy first, and whether they are any good.

Averaged ranking metrics (ROC-AUC / PR-AUC) count every window once, so a handful of near-duplicate
windows of **one** token can hide a confidently wrong entry rule — or fake a good one. Retrain #3's
candidate (#179, ``docs/reports/inverted-1h-179.txt``) is the case: a pump head that ranks well
(top 1 % at 1 h, precision ×7 the base rate over 362 tokens) whose top windows *passing the rug
gate* were 417 windows of 11 tokens, 85 % of them one token that rugged — precision 0.

Per horizon, the **entry selection** is the top ``top_fraction`` of the labeled windows by
``pump_prob`` among those passing the rug gate (``rug_prob <= rug_threshold``, the reference
policy's — the bot's ``MaxRugProb``); the **pump-only** selection is the same without the rug gate.
Each is described by its precision against the horizon's base rate and by its **breadth**: distinct
tokens and the share of the most frequent one. Everything is a share or a count over a rank, so no
fixed ``pump_prob`` threshold is involved (the probability scale shifts with every model).

Torch-free (numpy only): :mod:`solia_ai.evaluate` computes the selections, :mod:`solia_ai.ablation`
flattens them into run metrics and :mod:`solia_ai.promotion` guards on them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

# Run-metric prefixes: ``entry/<horizon>/<metric>`` (rug-gated) and ``entry_pump/...`` (no gate).
GATED = "entry"
PUMP_ONLY = "entry_pump"
ENTRY_METRICS = (
    "windows",  # selected windows
    "tokens",  # distinct tokens among them
    "top_token_share",  # share of the selected windows held by the most frequent token
    "precision",  # share of the selected windows that pumped
    "base_rate",  # share of all labeled windows that pumped (buy everything)
    "median_return",  # median realized return of the selected windows
    "pool_tokens",  # distinct tokens the selection was drawn from (after the rug gate)
)
# The rug gate the selections were measured under (the reference policy's).
RUG_THRESHOLD_KEY = "policy/rug_threshold"


def entry_key(prefix: str, horizon: str, metric: str) -> str:
    return f"{prefix}/{horizon}/{metric}"


def entry_keys(horizons: Iterable[str]) -> list[str]:
    """Every entry-selection run metric, rug-gated first, then pump-only."""
    horizons = list(horizons)
    return [
        entry_key(prefix, h, m)
        for prefix in (GATED, PUMP_ONLY)
        for h in horizons
        for m in ENTRY_METRICS
    ]


def select_entries(
    pump_prob: np.ndarray,
    rug_prob: np.ndarray,
    pumped: np.ndarray,
    returns: np.ndarray,
    mints: np.ndarray,
    *,
    top_fraction: float,
    rug_threshold: float | None,
) -> dict[str, Any]:
    """The top ``top_fraction`` of the windows by ``pump_prob`` among those with ``rug_prob <=
    rug_threshold`` (every window when ``rug_threshold`` is ``None``), described by
    :data:`ENTRY_METRICS`. Arrays are one horizon's labeled windows, aligned row for row.

    The selection holds ``ceil(top_fraction x pool)`` windows (at least one of a non-empty pool);
    ties keep the input order. An empty pool selects nothing: its rates are ``None``.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
    pump_prob = np.asarray(pump_prob, dtype=np.float64)
    pumped = np.asarray(pumped, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    mints = np.asarray(mints, dtype=object)
    n = int(pump_prob.size)
    pool = np.arange(n)
    if rug_threshold is not None:
        pool = pool[np.asarray(rug_prob, dtype=np.float64) <= rug_threshold]
    out: dict[str, Any] = {
        "top_fraction": top_fraction,
        "rug_threshold": rug_threshold,
        "pool_windows": int(pool.size),
        "pool_tokens": len(set(mints[pool])),
        "base_rate": float(pumped.mean()) if n else None,
    }
    if pool.size == 0:
        return {
            **out,
            "windows": 0,
            "tokens": 0,
            "top_token_share": None,
            "precision": None,
            "median_return": None,
        }
    # Float slack, so 1 % of 300 windows selects 3, not 4.
    k = max(1, math.ceil(top_fraction * pool.size - 1e-9))
    chosen = pool[np.argsort(-pump_prob[pool], kind="stable")[:k]]
    _, counts = np.unique(mints[chosen].astype(str), return_counts=True)
    return {
        **out,
        "windows": int(k),
        "tokens": int(counts.size),
        "top_token_share": float(counts.max() / k),
        "precision": float(pumped[chosen].mean()),
        "median_return": float(np.median(returns[chosen])),
    }


def flatten(entry: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, float | None]:
    """:attr:`solia_ai.evaluate.EvaluationReport.entry` (``{horizon: {"gated"|"pump_only": ...}}``)
    as run metrics (:func:`entry_keys`); ``None`` where undefined."""
    out: dict[str, float | None] = {}
    for horizon, selections in entry.items():
        for prefix, kind in ((GATED, "gated"), (PUMP_ONLY, "pump_only")):
            for m in ENTRY_METRICS:
                value = selections[kind][m]
                out[entry_key(prefix, horizon, m)] = None if value is None else float(value)
    return out


def summarize_selection(
    runs: Sequence[Mapping[str, float | None]], prefix: str, horizon: str, *, min_tokens: int
) -> dict[str, Any]:
    """Mean of each :data:`ENTRY_METRICS` over the runs whose selection is **measurable**: at
    least one window, drawn from at least ``min_tokens`` distinct tokens (a smaller pool could
    never be broad enough to judge). ``defined`` is false — never a pass — when no run is."""

    def measurable(m: Mapping[str, float | None]) -> bool:
        windows = m.get(entry_key(prefix, horizon, "windows"))
        pool = m.get(entry_key(prefix, horizon, "pool_tokens"))
        return bool(windows) and pool is not None and pool >= min_tokens

    usable = [m for m in runs if measurable(m)]
    out: dict[str, Any] = {"defined": bool(usable), "n_runs": len(usable)}
    for metric in ENTRY_METRICS:
        values = [v for m in usable if (v := m.get(entry_key(prefix, horizon, metric))) is not None]
        out[metric] = float(np.mean(values)) if values else None
    return out
