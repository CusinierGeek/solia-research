"""Canonical model feature spec.

The single source of truth for *which* columns of the ``features`` table (written by the .NET
pipeline) feed the model, and in what order. Parity tests check this list against the schema the
migrations produce, so the two can never drift silently. NULLs are imputed to 0 (absence of
signal) unless :data:`NULL_FILL` gives the column a neutral value; a missing column or a
non-numeric value is an error, never silently zeroed. Rows written before a group's migration
(NULL where the pipeline now always writes a value) are refused for that group, see
:func:`check_group_coverage`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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
    "net_sol_flow",
    "buy_volume_ratio",
    "avg_buy_sol",
    "max_buy_sol",
    "large_buy_count",
    "return_short",
    "return_medium",
    "return_long",
    "price_acceleration",
    "ema_slope",
    "volume_acceleration",
    "volume_burst_ratio",
    "price_history_sec",
    "unique_traders",
    "top_trader_share",
    "trader_hhi",
    "new_trader_ratio",
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
    # FE-T2 (#111): SOL-weighted buy pressure, SOL-quoted trades only.
    "buy_pressure": (
        "net_sol_flow",
        "buy_volume_ratio",
        "avg_buy_sol",
        "max_buy_sol",
        "large_buy_count",
    ),
    # FE-T3 (#112): momentum & acceleration over the pipeline's short/medium/long windows.
    "momentum": (
        "return_short",
        "return_medium",
        "return_long",
        "price_acceleration",
        "ema_slope",
        "volume_acceleration",
        "volume_burst_ratio",
        "price_history_sec",  # how far back the returns can look: "no history" vs a flat price
    ),
    # FE-T4 (#113): holder-concentration proxy from trade signers (txSigner) over the window.
    "concentration": (
        "unique_traders",
        "top_trader_share",
        "trader_hhi",
        "new_trader_ratio",
    ),
}

# The groups making up the baseline model: the original 19 columns. Groups outside it are
# candidates, disabled (zeroed) by default until their ablation lift clears the margin — the
# model trains on ``features.groups`` from the config, which defaults to this baseline.
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


def column_mask(columns: Sequence[str], enabled: Iterable[str]) -> np.ndarray:
    """A ``[len(columns)]`` boolean mask: ``True`` where the column is in ``enabled``.

    Raises:
        ValueError: on an enabled column that isn't one of ``columns``.
    """
    enabled = set(enabled)
    if unknown := sorted(enabled - set(columns)):
        raise ValueError(f"enabled column(s) not in the feature spec: {', '.join(unknown)}")
    return np.array([col in enabled for col in columns], dtype=bool)


def group_mask(groups: Iterable[str]) -> np.ndarray:
    """A ``[F]`` boolean mask over :data:`FEATURE_COLUMNS`: ``True`` where the column is in an
    enabled group. The model zeroes disabled columns (no signal)."""
    return column_mask(FEATURE_COLUMNS, group_columns(groups))


# NULL imputation where 0 would mean something: no SOL trade in the window is "no pressure either
# way" (0.5), not "only sells" (0.0); a burst ratio of 1 is steady volume, 0 would be a collapse.
# No recent signed trader is -1 for new_trader_ratio: 0 is a real reading ("all returning"). The
# concentration shares live in (0, 1], so their default 0 already marks "no signed volume".
NULL_FILL: dict[str, float] = {
    "buy_volume_ratio": 0.5,
    "volume_burst_ratio": 1.0,
    "new_trader_ratio": -1.0,
}

# Per group, the columns the pipeline always writes since the group's migration: a NULL there
# means the row predates it (never computed), not "no signal" — such rows must be re-replayed
# before the group is trained or ablated on, or they'd teach the model a fake zero.
ALWAYS_WRITTEN: dict[str, tuple[str, ...]] = {
    "buy_pressure": ("net_sol_flow", "large_buy_count"),  # migration 0005
    "momentum": ("volume_burst_ratio", "price_history_sec"),  # migration 0006
    "concentration": ("unique_traders",),  # migration 0007
}


def check_group_coverage(df: pd.DataFrame, groups: Iterable[str]) -> None:
    """Refuse ``df`` if an enabled group has rows written before its migration.

    Raises:
        ValueError: naming the group and how many rows lack it (re-replay them first).
    """
    for group in dict.fromkeys(groups):
        cols = [c for c in ALWAYS_WRITTEN.get(group, ()) if c in df.columns]
        if not cols:
            continue
        uncomputed = int(df[cols].isna().any(axis=1).sum())
        if uncomputed:
            raise ValueError(
                f"{uncomputed} of {len(df)} feature rows predate the '{group}' group "
                f"(NULL {', '.join(cols)}): re-replay them before enabling it"
            )


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
    to numbers. NULLs become their :data:`NULL_FILL` value (default ``0.0``); non-finite values
    (±inf, or magnitudes beyond float32) become ``0.0``.

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
        numeric[col] = coerced.astype("float64").fillna(NULL_FILL.get(col, 0.0))

    with np.errstate(over="ignore"):  # out-of-range values become inf here, then imputed below
        matrix = pd.DataFrame(numeric, columns=FEATURE_COLUMNS).to_numpy(dtype=np.float32)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
