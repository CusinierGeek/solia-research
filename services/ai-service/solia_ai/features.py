"""Canonical model feature spec.

The single source of truth for *which* columns of the ``features`` table (written by the .NET
pipeline) feed the model, and in what order. Parity tests check this list against the schema the
migrations produce, so the two can never drift silently. NULLs are imputed to 0 (absence of
signal); a missing column or a non-numeric value is an error, never silently zeroed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# Ordered model inputs: the numeric features plus the migrated flag. Keep this list and the
# `features` table (services/db/migrations) in lockstep — see tests/test_features.py.
FEATURE_COLUMNS: list[str] = [
    "price",
    "volume",
    "market_cap",
    "holders",
    "liquidity",
    "buy_events",
    "sell_events",
    "buy_sell_ratio",
    "token_age_sec",
    "trending_score",
    "social_mentions",
    "holders_growth_rate",
    "volume_growth_rate",
    "tx_creation_rate",
    "volatility",
    "ema",
    "rsi",
    "price_derivative",
    "is_migrated",
]

# Feature groups for ablation (solia_ai.ablation): every model input belongs to exactly one group
# — see tests/test_features.py. An enrichment task adds its columns to FEATURE_COLUMNS *and* a new
# group here, so its lift can be measured against the baseline before it is accepted.
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": ("price", "market_cap", "ema", "rsi", "price_derivative", "volatility"),
    "liquidity": ("liquidity", "is_migrated"),
    "flow": (
        "volume",
        "buy_events",
        "sell_events",
        "buy_sell_ratio",
        "volume_growth_rate",
        "tx_creation_rate",
    ),
    "holders": ("holders", "holders_growth_rate"),
    "context": ("token_age_sec", "trending_score", "social_mentions"),
}

# The groups making up the baseline model: today's 19 columns. Groups outside it are candidates.
BASELINE_GROUPS: tuple[str, ...] = ("price", "liquidity", "flow", "holders", "context")


def group_columns(groups: Iterable[str]) -> list[str]:
    """The columns of ``groups``, in :data:`FEATURE_COLUMNS` order.

    Raises:
        ValueError: on a group name not declared in :data:`FEATURE_GROUPS`.
    """
    groups = list(groups)
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature group(s): {', '.join(unknown)} "
            f"(known: {', '.join(FEATURE_GROUPS)})"
        )
    wanted = {col for g in groups for col in FEATURE_GROUPS[g]}
    return [col for col in FEATURE_COLUMNS if col in wanted]


def group_mask(groups: Iterable[str]) -> np.ndarray:
    """A ``[F]`` boolean mask over :data:`FEATURE_COLUMNS`: ``True`` where the column is in an
    enabled group. Disabled columns are zeroed by the ablation harness (no signal)."""
    enabled = set(group_columns(groups))
    return np.array([col in enabled for col in FEATURE_COLUMNS], dtype=bool)


# Columns of the `features` table that are deliberately NOT model inputs (identity, bookkeeping,
# categorical/JSONB we don't encode yet). `venue` encoding and the JSONB signals are future work.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "id",
        "mint",
        "ts",
        "venue",
        "pool_address",
        "onchain_signals",
        "behavioral_signals",
        "raw",
    }
)


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Project ``features`` rows onto :data:`FEATURE_COLUMNS` as a ``float32`` matrix.

    Columns are taken in canonical order; booleans and ``NUMERIC``/``Decimal`` values are coerced
    to numbers. NULLs and non-finite values (±inf, or magnitudes beyond float32) become ``0.0``.

    Raises:
        ValueError: if a feature column is missing (e.g. a frame from another table, or a renamed
            column) or holds a value that isn't numeric.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"feature frame is missing column(s): {', '.join(missing)}")

    numeric: dict[str, pd.Series] = {}
    for col in FEATURE_COLUMNS:
        values = df[col]
        coerced = pd.to_numeric(values, errors="coerce")
        invalid = coerced.isna() & values.notna()
        if invalid.any():
            sample = values[invalid].iloc[0]
            raise ValueError(f"feature column '{col}' holds a non-numeric value: {sample!r}")
        numeric[col] = coerced.astype("float64")

    with np.errstate(over="ignore"):  # out-of-range values become inf here, then imputed below
        matrix = pd.DataFrame(numeric, columns=FEATURE_COLUMNS).to_numpy(dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
