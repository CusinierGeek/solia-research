"""Rug guard: an explainable LightGBM risk score on the rug-guard labels, judged on a criterion
fixed before the run (#260).

**Dataset** (``build``). Anchors: each token's first ``features`` row of every
``anchor_seconds`` (15 min) bucket, labeled by :mod:`solia_ai.rug_labels` (``sudden_<H>``,
``gradual_<H>``, NaN = unknown) and joined with the row's own risk columns (:data:`INPUT_COLUMNS`,
NULL kept as NaN: LightGBM routes missing values on its own, so "no creator history" never reads as
a real 0) and the pool's **seed**: the liquidity of the anchor's pool at its first observation
(:class:`SeedTracker`, point-in-time — the anchor itself is an observation of that pool; unknown for
pools first seen in the first :data:`SEED_WARMUP` of the data, which may predate it). Rows are read
in time chunks (days, cut at the segment edges), each labeled with its own 24 h of future and
written to one parquet file, so a crash loses one chunk.

**Split** (:data:`DEFAULT_SPLIT`): train, validation (early stopping), period 1, period 2, in time
order, each starting at least the longest horizon after the previous one ends — no label of an
earlier segment reads a price of a later one.

**Models** (``evaluate``): one LightGBM binary classifier per kind x horizon, early-stopped on the
validation segment. Reasons: each score's top risk-raising feature contributions (TreeSHAP,
``pred_contrib``), :func:`reasons`.

**Criterion** (:class:`Criterion`, docs/ML-MODEL.md "Rug guard"), on ``sudden_24h``, survivors =
each token's first anchor of the period at ``token_age_sec >= 6 h`` with a known label: the
threshold is the cut keeping the most period-1 survivors at a sudden-rug rate <= 3 %
(:func:`pick_threshold`); on period 2, unchanged, the survivors scored at or under it must have a
rate <= 3 %, be >= 30 % of the survivors, and beat the plain rule "LP burned >= 90 %"
(:func:`beats_rule`). Also reported on every token's first anchor of the period (any age).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import horizon_seconds, load_config
from .features import (
    HOLDERS_SUPPLY_COLUMNS,
    METADATA_COLUMNS,
    REPUTATION_COLUMNS,
    VENUES,
    WALLET_COLUMNS,
)
from .rug_labels import RugLabelSettings, label_in_groups, read_label_frame

logger = logging.getLogger(__name__)

KINDS = ("sudden", "gradual")
SEGMENTS = ("train", "val", "p1", "p2")
DEFAULT_ANCHOR_SECONDS = 900
SEED_WARMUP = timedelta(hours=1)
SURVIVOR_AGE_SECONDS = 6 * 3600


def _epoch_ns(ts: Any) -> pd.Series:
    """Timestamps as int64 epoch nanoseconds, whatever their stored unit (the database reads µs)."""
    return pd.Series(pd.to_datetime(ts, utc=True)).dt.as_unit("ns").astype("int64")


def _utc(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime().astimezone(UTC)


# The features table's risk columns: liquidity / activity context, burn, wallet & creator history,
# bundling, metadata, holder concentration. Coverage columns (replay warm-up bookkeeping) and the
# market-wide aggregates (the same on every token of a tick) are left out on purpose.
INPUT_COLUMNS: tuple[str, ...] = (
    "liquidity",
    "market_cap",
    "volume",
    "holders",
    "buy_events",
    "sell_events",
    "buy_sell_ratio",
    "token_age_sec",
    "volatility",
    "return_short",
    "return_medium",
    "return_long",
    "is_migrated",
    "net_sol_flow",
    "buy_volume_ratio",
    "max_buy_sol",
    "large_buy_count",
    "unique_traders",
    "top_trader_share",
    "trader_hhi",
    "new_trader_ratio",
    "burned_liquidity",
    *(c for c in WALLET_COLUMNS if c != "wal_coverage_sec"),
    *REPUTATION_COLUMNS,
    *(c for c in METADATA_COLUMNS if c != "meta_coverage_sec"),
    *HOLDERS_SUPPLY_COLUMNS,
)
# Derived in model_inputs: the seed (round amounts), the pool's liquidity against it, a wash-trading
# proxy (trades per distinct trader), the venue (categorical).
DERIVED_COLUMNS: tuple[str, ...] = (
    "pool_seed_sol",
    "pool_seed_frac",
    "liquidity_vs_seed",
    "events_per_trader",
    "venue_code",
)
MODEL_COLUMNS: tuple[str, ...] = (*INPUT_COLUMNS, *DERIVED_COLUMNS)
_BOOLEAN_COLUMNS = frozenset({"is_migrated"})

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
DEFAULT_EARLY_STOPPING_ROUNDS = 100


# --------------------------------------------------------------------------- split


@dataclass(frozen=True)
class Split:
    """Anchor windows ``[start, end)`` of the four segments, in time order."""

    train: tuple[datetime, datetime]
    val: tuple[datetime, datetime]
    p1: tuple[datetime, datetime]
    p2: tuple[datetime, datetime]

    def segments(self) -> list[tuple[str, tuple[datetime, datetime]]]:
        return [(name, getattr(self, name)) for name in SEGMENTS]

    def check(self, embargo_seconds: float) -> None:
        """Raise ``ValueError`` unless each segment is non-empty and starts at least
        ``embargo_seconds`` after the previous one ends (its labels' future)."""
        prev: tuple[str, datetime] | None = None
        for name, (start, end) in self.segments():
            if not start < end:
                raise ValueError(f"segment {name} is empty: [{start}, {end})")
            if prev and start < prev[1] + timedelta(seconds=embargo_seconds):
                raise ValueError(
                    f"segment {name} starts at {start:%Y-%m-%d %H:%M}, less than "
                    f"{embargo_seconds / 3600:g} h after {prev[0]} ends (labels would overlap)"
                )
            prev = (name, end)

    def segment_of(self, ts: pd.Series) -> np.ndarray:
        """Each timestamp's segment name, ``""`` outside every segment."""
        ns = _epoch_ns(ts).to_numpy()
        out = np.full(len(ns), "", dtype=object)
        for name, (start, end) in self.segments():
            out[(ns >= pd.Timestamp(start).value) & (ns < pd.Timestamp(end).value)] = name
        return out

    def describe(self) -> dict[str, list[str]]:
        return {n: [a.isoformat(), b.isoformat()] for n, (a, b) in self.segments()}


# Fixed before the run (#260): the retained features start 2026-10-27 00:00 UTC (the first day only
# seeds the pool tracker and the wallet memories); 24 h embargoes; period 2 ends 24 h before the
# data did when the criterion was written (2026-11-09 18:00 UTC).
DATA_SINCE = _utc("2026-10-27")
DEFAULT_SPLIT = Split(
    train=(_utc("2026-10-28"), _utc("2026-11-02")),
    val=(_utc("2026-11-03"), _utc("2026-11-04")),
    p1=(_utc("2026-11-05"), _utc("2026-11-06 12:00")),
    p2=(_utc("2026-11-07 12:00"), _utc("2026-11-08 18:00")),
)


def chunks(split: Split, since: datetime) -> list[tuple[datetime, datetime, str | None]]:
    """``[since, end of p2)`` cut at every UTC midnight and segment edge: ``(start, end,
    segment)``, ``None`` between segments (read only to follow the pools)."""
    stop = split.p2[1]
    if since >= stop:
        raise ValueError(f"since ({since}) must be before the end of p2 ({stop})")
    edges = {since, stop}
    for _, (start, end) in split.segments():
        edges |= {t for t in (start, end) if since < t < stop}
    day = since.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    while day < stop:
        edges.add(day)
        day += timedelta(days=1)
    ordered = sorted(edges)
    out = []
    for start, end in zip(ordered, ordered[1:], strict=False):
        seg = next((n for n, (a, b) in split.segments() if a <= start and end <= b), None)
        out.append((start, end, seg))
    return out


# --------------------------------------------------------------------------- pool seeds


def pool_keys(pool_address: pd.Series, venue: pd.Series) -> np.ndarray:
    """A pool's identity: its address; a NULL address is its venue's curve (as rug_labels)."""
    by_venue = ("venue:" + venue.astype(object).fillna("").map(str)).to_numpy(dtype=object)
    address = pool_address.astype(object).to_numpy()
    return np.where(pd.isna(address), by_venue, address)


class SeedTracker:
    """Each pool's liquidity at its first observation (its seed), fed in time order. Pools first
    seen before ``known_after`` may predate the data: their seed is unknown (NaN)."""

    def __init__(self, known_after: datetime) -> None:
        self._known_after = pd.Timestamp(known_after).value
        self._seed = pd.Series(dtype=np.float64)

    def __len__(self) -> int:
        return len(self._seed)

    def update(self, firsts: pd.DataFrame) -> None:
        """``firsts``: rows ``key``, ``ts``, ``liquidity`` (non-null), later than every row fed
        before; each new key's earliest row is its seed."""
        if not len(firsts):
            return
        rows = firsts.assign(ns=_epoch_ns(firsts["ts"]))
        rows = rows.sort_values("ns", kind="stable").drop_duplicates("key")
        rows = rows[~rows["key"].isin(self._seed.index)]
        value = np.where(
            rows["ns"].to_numpy() >= self._known_after, rows["liquidity"].astype(float), np.nan
        )
        self._seed = pd.concat([self._seed, pd.Series(value, index=rows["key"].to_numpy())])

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        return self._seed.reindex(pd.Index(keys, dtype=object)).to_numpy(np.float64)


def read_pool_firsts(engine: Any, start: datetime, end: datetime) -> pd.DataFrame:
    """Per pool, its first row with a liquidity in ``[start, end)`` (1-hour slices, in order)."""
    from sqlalchemy import text

    query = text(
        "SELECT DISTINCT ON (mint, pool_address, venue) pool_address, venue, ts, "
        "liquidity::float8 AS liquidity FROM features WHERE ts >= :lo AND ts < :hi "
        "AND liquidity IS NOT NULL ORDER BY mint, pool_address, venue, ts"
    )
    parts, lo = [], start
    while lo < end:
        hi = min(lo + timedelta(hours=1), end)
        with engine.connect() as conn:
            part = pd.read_sql(query, conn, params={"lo": lo, "hi": hi})
        parts.append(
            pd.DataFrame(
                {
                    "key": pool_keys(part["pool_address"], part["venue"]),
                    "ts": part["ts"],
                    "liquidity": part["liquidity"],
                }
            )
        )
        lo = hi
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# --------------------------------------------------------------------------- dataset


def read_anchor_features(
    engine: Any, start: datetime, end: datetime, anchor_seconds: float
) -> pd.DataFrame:
    """Each token's first row of every ``anchor_seconds`` bucket of ``[start, end)``: the same
    anchors as :func:`label_in_groups`' ``anchor_every_seconds`` (epoch-aligned buckets), with the
    :data:`INPUT_COLUMNS`. Read in 1-hour slices (bucket-aligned when ``anchor_seconds`` divides
    an hour)."""
    from sqlalchemy import text

    cols = ", ".join(
        f"{c}::int::float8 AS {c}" if c in _BOOLEAN_COLUMNS else f"{c}::float8 AS {c}"
        for c in INPUT_COLUMNS
    )
    bucket = "floor(extract(epoch FROM ts) / :every)"
    query = text(
        f"SELECT DISTINCT ON (mint, {bucket}) mint, ts, venue, pool_address, {cols} "
        f"FROM features WHERE ts >= :lo AND ts < :hi ORDER BY mint, {bucket}, ts"
    )
    parts, lo = [], start
    while lo < end:
        hi = min(lo + timedelta(hours=1), end)
        with engine.connect() as conn:
            part = pd.read_sql(query, conn, params={"lo": lo, "hi": hi, "every": anchor_seconds})
        parts.append(part.astype(dict.fromkeys(INPUT_COLUMNS, np.float32)))
        lo = hi
    return pd.concat(parts, ignore_index=True)


def assemble(
    labels: pd.DataFrame, wide: pd.DataFrame, tracker: SeedTracker, segment: str
) -> pd.DataFrame:
    """The anchors' rows: ``wide`` (features) joined on ``(mint, ts)`` with ``labels``' label
    columns, plus the pool seed and the segment. Anchors missing on either side are dropped."""
    lab_cols = [c for c in labels.columns if c.split("_", 1)[0] in (*KINDS, "drain")]
    left = wide.assign(mint=wide["mint"].astype(str), ts=_epoch_ns(wide["ts"]))
    right = labels[["mint", "ts", *lab_cols]].assign(
        mint=labels["mint"].astype(str), ts=_epoch_ns(labels["ts"])
    )
    frame = left.merge(right, on=["mint", "ts"], how="inner", validate="many_to_one")
    frame = frame.drop_duplicates(["mint", "ts"]).reset_index(drop=True)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["pool_seed_sol"] = tracker.lookup(pool_keys(frame["pool_address"], frame["venue"]))
    frame["segment"] = segment
    return frame


def venue_codes(venue: pd.Series) -> np.ndarray:
    """Index of the venue in :data:`VENUES`; any other value or NULL is ``len(VENUES)``."""
    codes = pd.Index(list(VENUES)).get_indexer(venue.astype(object))
    return np.where(codes < 0, len(VENUES), codes).astype(np.float32)


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """The :data:`MODEL_COLUMNS` of each anchor (float32, NaN = unknown)."""
    x = frame[list(INPUT_COLUMNS)].astype(np.float32)
    seed = frame["pool_seed_sol"].astype(np.float64)
    liquidity = frame["liquidity"].astype(np.float64)
    events = frame["buy_events"].astype(np.float64) + frame["sell_events"].astype(np.float64)
    traders = frame["unique_traders"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x["pool_seed_sol"] = seed
        x["pool_seed_frac"] = (seed - np.round(seed)).abs()
        x["liquidity_vs_seed"] = np.where(seed > 0, liquidity / seed, np.nan)
        x["events_per_trader"] = np.where(traders > 0, events / traders, np.nan)
    x["venue_code"] = venue_codes(frame["venue"])
    return x[list(MODEL_COLUMNS)].astype(np.float32)


def build(
    engine: Any,
    out_dir: Path,
    split: Split,
    settings: RugLabelSettings,
    *,
    since: datetime = DATA_SINCE,
    anchor_seconds: float = DEFAULT_ANCHOR_SECONDS,
    rows_per_group: int = 3_000_000,
    resume: bool = False,
) -> list[Path]:
    """Write one parquet file of anchors per chunk of a segment (module doc)."""
    max_h = max(horizon_seconds(h) for h in settings.horizons)
    split.check(max_h)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = SeedTracker(known_after=since + SEED_WARMUP)
    written = []
    for start, end, segment in chunks(split, since):
        tracker.update(read_pool_firsts(engine, start, end))
        logger.info("pools %s .. %s: %d known", start.isoformat(), end.isoformat(), len(tracker))
        if segment is None:
            continue
        path = out_dir / f"{segment}-{start:%Y%m%dT%H%M}.parquet"
        if resume and path.exists():
            written.append(path)
            continue
        frame = read_label_frame(engine, start, end, max_h)
        labels = label_in_groups(
            frame,
            settings,
            anchor_from=start,
            anchor_to=end,
            rows_per_group=rows_per_group,
            anchor_every_seconds=anchor_seconds,
        )
        del frame
        part = assemble(labels, read_anchor_features(engine, start, end, anchor_seconds), tracker,
                        segment)  # fmt: skip
        part.to_parquet(path, index=False)
        logger.info("wrote %s: %d anchors", path.name, len(part))
        written.append(path)
    return written


def load_dataset(path: Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("*.parquet"))
    if not files:
        raise ValueError(f"no parquet files in {path}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------- models & reasons


def require_lightgbm() -> Any:
    from .benchmark_gbdt import MissingDependencyError, require_lightgbm

    try:
        return require_lightgbm()
    except MissingDependencyError as exc:
        raise SystemExit(str(exc)) from None


def fit_target(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    *,
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    seed: int = 0,
    n_jobs: int = 0,
) -> Any:
    """One LightGBM binary classifier (``venue_code`` categorical), early-stopped on the
    validation rows (binary log-loss)."""
    lgb = require_lightgbm()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("training and validation rows must hold both classes")
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=n_jobs or None,
        verbose=-1,
        **dict(DEFAULT_PARAMS if params is None else params),
    )
    categorical = ["venue_code"] if "venue_code" in x_train.columns else "auto"
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="binary_logloss",
        categorical_feature=categorical,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def reasons(model: Any, x: pd.DataFrame, top: int = 3) -> list[list[dict[str, Any]]]:
    """Per row, its ``top`` risk-raising features: ``{feature, value, contribution}`` by
    decreasing TreeSHAP contribution to the log-odds (positive ones only)."""
    if top < 1:
        raise ValueError(f"top must be >= 1, got {top}")
    contrib = np.asarray(model.booster_.predict(x, pred_contrib=True))[:, :-1]  # last: bias
    names = list(x.columns)
    values = x.to_numpy(np.float64)
    out = []
    for r in range(len(x)):
        order = np.argsort(-contrib[r], kind="stable")[:top]
        out.append(
            [
                {"feature": names[c], "value": _finite(values[r, c]), "contribution":
                 float(contrib[r, c])}
                for c in order if contrib[r, c] > 0
            ]  # fmt: skip
        )
    return out


def format_reasons(items: Sequence[Mapping[str, Any]]) -> str:
    def value(v: float | None) -> str:
        return "NULL" if v is None else f"{v:.4g}"

    return ", ".join(
        f"{i['feature']}={value(i['value'])} (+{i['contribution']:.2f})" for i in items
    )


def _finite(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


# --------------------------------------------------------------------------- criterion


@dataclass(frozen=True)
class Criterion:
    """The pre-registered pass rule (module doc)."""

    target: str = "sudden_24h"
    min_age_seconds: float = SURVIVOR_AGE_SECONDS
    max_rate: float = 0.03
    min_coverage: float = 0.30
    rule_min_burned: float = 90.0


def first_anchors(frame: pd.DataFrame, target: str, *, min_age_seconds: float = 0) -> pd.DataFrame:
    """Each token's first anchor of ``frame`` at ``token_age_sec >= min_age_seconds``, kept only
    when its ``target`` label is known."""
    rows = frame
    if min_age_seconds > 0:
        rows = rows[rows["token_age_sec"] >= min_age_seconds]
    rows = rows.sort_values("ts", kind="stable").drop_duplicates("mint")
    return rows[rows[target].notna()]


def pick_threshold(score: np.ndarray, y: np.ndarray, max_rate: float) -> float | None:
    """The score cut keeping the most rows (``score <= cut``, ties together) whose positive
    rate is <= ``max_rate``; ``None`` when even the lowest-scored group is above it."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, np.float64)
    if not len(score):
        return None
    order = np.argsort(score, kind="stable")
    s, positives = score[order], np.cumsum(y[order])
    kept = np.arange(1, len(s) + 1)
    group_end = np.r_[s[1:] != s[:-1], True]  # a cut can't split tied scores
    ok = group_end & (positives / kept <= max_rate + 1e-12)
    if not ok.any():
        return None
    return float(s[np.flatnonzero(ok)[-1]])


def wilson_upper(positives: int, n: int, z: float = 1.96) -> float | None:
    """Upper bound of the Wilson score interval of a rate (95 % by default)."""
    if n == 0:
        return None
    p = positives / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre + margin) / (1 + z * z / n)


def kept_stats(kept: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Size, coverage and positive rate of the rows ``kept`` (a boolean mask over ``y``)."""
    kept = np.asarray(kept, bool)
    y = np.asarray(y, np.float64)
    n_kept, positives = int(kept.sum()), int(y[kept].sum())
    return {
        "n": int(len(y)),
        "kept": n_kept,
        "coverage": n_kept / len(y) if len(y) else None,
        "positives": positives,
        "rate": positives / n_kept if n_kept else None,
        "rate_upper95": wilson_upper(positives, n_kept),
    }


def beats_rule(
    score: np.ndarray, y: np.ndarray, rule_kept: np.ndarray, model_kept: np.ndarray
) -> dict[str, Any]:
    """Whether the model beats the plain rule on the same rows: the rule keeps ``n_R`` rows; the
    model's ``n_R`` lowest-scored rows have a lower rate, or the model's own safe set
    (``model_kept``) keeps more rows at a rate no higher; or the rule keeps nothing."""
    score = np.asarray(score, np.float64)
    y = np.asarray(y, np.float64)
    rule = kept_stats(rule_kept, y)
    own = kept_stats(model_kept, y)
    n_r = rule["kept"]
    same = np.zeros(len(y), bool)
    same[np.argsort(score, kind="stable")[:n_r]] = True
    at_rule = kept_stats(same, y)
    if n_r == 0:
        beats, why = True, "the rule keeps no survivor"
    elif at_rule["rate"] < rule["rate"]:
        beats, why = True, "lower rate at the rule's coverage"
    elif own["kept"] > n_r and own["rate"] is not None and own["rate"] <= rule["rate"]:
        beats, why = True, "more coverage at a rate no higher"
    else:
        beats, why = False, "no lower rate at the rule's coverage, no more coverage at its rate"
    return {"beats": beats, "why": why, "rule": rule, "model_at_rule_coverage": at_rule}


def judge(
    p1: pd.DataFrame, p2: pd.DataFrame, criterion: Criterion, score_column: str = "score"
) -> dict[str, Any]:
    """Apply :class:`Criterion` to the scored survivors of period 1 (threshold) and period 2
    (verdict). ``p1`` / ``p2``: :func:`first_anchors` rows with ``score_column``."""
    c = criterion
    threshold = pick_threshold(p1[score_column], p1[c.target], c.max_rate)
    result: dict[str, Any] = {"criterion": asdict(c), "threshold": threshold}
    if threshold is None:
        return {**result, "passed": False, "why": "no period-1 cut reaches the rate"}
    result["p1"] = kept_stats(p1[score_column].to_numpy() <= threshold, p1[c.target])
    score2, y2 = p2[score_column].to_numpy(), p2[c.target].to_numpy()
    model_kept = score2 <= threshold
    safe = kept_stats(model_kept, y2)
    burned = p2["burned_liquidity"].to_numpy(np.float64)
    rule = beats_rule(score2, y2, np.nan_to_num(burned, nan=-1.0) >= c.rule_min_burned, model_kept)
    checks = {
        "rate": safe["rate"] is not None and safe["rate"] <= c.max_rate,
        "coverage": safe["coverage"] is not None and safe["coverage"] >= c.min_coverage,
        "beats_rule": rule["beats"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    return {
        **result,
        "p2": safe,
        "rule_comparison": rule,
        "checks": checks,
        "passed": not failed,
        "why": "all checks pass" if not failed else "failed: " + ", ".join(failed),
    }


# --------------------------------------------------------------------------- evaluation


def targets(horizons: Sequence[str]) -> list[str]:
    return [f"{kind}_{h}" for kind in KINDS for h in horizons]


@dataclass
class Evaluation:
    report: dict[str, Any]
    models: dict[str, Any] = field(default_factory=dict)


def evaluate(
    frame: pd.DataFrame,
    horizons: Sequence[str],
    criterion: Criterion | None = None,
    *,
    params: Mapping[str, Any] | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    n_jobs: int = 0,
    top_reasons: int = 3,
    examples: int = 10,
) -> Evaluation:
    """Train one model per target on ``segment == "train"`` (early stopping on ``"val"``), score
    periods 1 and 2, judge the criterion (default: the pre-registered :class:`Criterion`).
    ``frame``: :func:`build`'s anchors."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    criterion = criterion or Criterion()
    if criterion.target not in targets(horizons):
        raise ValueError(f"criterion target {criterion.target!r} is not one of the targets")
    seg = {name: frame[frame["segment"] == name] for name in SEGMENTS}
    for name, rows in seg.items():
        if not len(rows):
            raise ValueError(f"segment {name} has no anchors")
    x = {name: model_inputs(rows) for name, rows in seg.items()}
    report: dict[str, Any] = {
        "anchors": {name: len(rows) for name, rows in seg.items()},
        "tokens": {name: int(rows["mint"].nunique()) for name, rows in seg.items()},
        "targets": {},
    }
    models: dict[str, Any] = {}
    scored = {p: {"all": None, "survivors": None} for p in ("p1", "p2")}
    for target in targets(horizons):
        tr = seg["train"][target].notna().to_numpy()
        va = seg["val"][target].notna().to_numpy()
        model = fit_target(
            x["train"][tr], seg["train"][target][tr].to_numpy(),
            x["val"][va], seg["val"][target][va].to_numpy(),
            params=params, early_stopping_rounds=early_stopping_rounds, n_jobs=n_jobs,
        )  # fmt: skip
        models[target] = model
        gains = model.booster_.feature_importance(importance_type="gain")
        share = gains / gains.sum() if gains.sum() > 0 else gains
        entry: dict[str, Any] = {
            "train_rows": int(tr.sum()),
            "train_rate": float(seg["train"][target][tr].mean()),
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "importance": [
                {"feature": MODEL_COLUMNS[i], "share": float(share[i])}
                for i in np.argsort(-share)[:15]
            ],
        }
        for period in ("p1", "p2"):
            for pop, age in (("all", 0), ("survivors", criterion.min_age_seconds)):
                rows = first_anchors(seg[period], target, min_age_seconds=age)
                prob = model.predict_proba(model_inputs(rows))[:, 1]
                y = rows[target].to_numpy(np.float64)
                both = len(np.unique(y)) == 2
                entry[f"{period}/{pop}"] = {
                    "n": int(len(y)),
                    "rate": float(y.mean()) if len(y) else None,
                    "roc_auc": float(roc_auc_score(y, prob)) if both else None,
                    "pr_auc": float(average_precision_score(y, prob)) if both else None,
                }
                if target == criterion.target:
                    scored[period][pop] = rows.assign(score=prob)
        report["targets"][target] = entry

    report["criterion"] = judge(scored["p1"]["survivors"], scored["p2"]["survivors"], criterion)
    threshold = report["criterion"]["threshold"]
    if threshold is not None:
        every = scored["p2"]["all"]
        report["all_tokens_p2_at_threshold"] = kept_stats(
            every["score"].to_numpy() <= threshold, every[criterion.target]
        )
    seen = set(seg["train"]["mint"].astype(str))
    surv2 = scored["p2"]["survivors"]
    report["p2_survivors_seen_in_train"] = float(surv2["mint"].astype(str).isin(seen).mean())
    report["examples"] = _examples(models[criterion.target], surv2, criterion, top_reasons,
                                   examples)  # fmt: skip
    return Evaluation(report=report, models=models)


def _examples(
    model: Any, rows: pd.DataFrame, criterion: Criterion, top: int, n: int
) -> list[dict[str, Any]]:
    """Period-2 survivors at both ends of the score, with their reasons."""
    picked = pd.concat([rows.nlargest(n, "score"), rows.nsmallest(min(n, 5), "score")])
    explained = reasons(model, model_inputs(picked), top=top)
    return [
        {
            "mint": str(r.mint),
            "ts": pd.Timestamp(r.ts).isoformat(),
            "venue": None if pd.isna(r.venue) else str(r.venue),
            "score": float(r.score),
            criterion.target: float(getattr(r, criterion.target)),
            "reasons": why,
        }
        for r, why in zip(picked.itertuples(), explained, strict=True)
    ]


def render(report: Mapping[str, Any]) -> str:
    """The text summary of an :func:`evaluate` report."""
    lines = [
        "Rug guard (#260): LightGBM per target, first anchor per token per period",
        f"anchors {report['anchors']}; tokens {report['tokens']}",
        "",
        f"{'target':<13} {'iter':>5}  {'period/pop':<14} {'n':>6} {'rate':>7} {'ROC':>6} "
        f"{'PR':>6}",
    ]
    for target, entry in report["targets"].items():
        for key in ("p1/all", "p1/survivors", "p2/all", "p2/survivors"):
            m = entry[key]
            lines.append(
                f"{target:<13} {entry['best_iteration']:>5}  {key:<14} {m['n']:>6} "
                f"{_pct(m['rate']):>7} {_num(m['roc_auc']):>6} {_num(m['pr_auc']):>6}"
            )
    crit = report["criterion"]
    c = crit["criterion"]
    lines += [
        "",
        f"Criterion on {c['target']}, survivors (age >= {c['min_age_seconds'] / 3600:g} h): "
        f"rate <= {_pct(c['max_rate'])}, coverage >= {_pct(c['min_coverage'])}, beats "
        f"'LP burned >= {c['rule_min_burned']:g} %'",
        f"threshold (period 1): {crit['threshold']}",
    ]
    for key in ("p1", "p2"):
        if key in crit:
            lines.append(f"  {key} safe set: {_kept(crit[key])}")
    if "rule_comparison" in crit:
        rc = crit["rule_comparison"]
        lines += [
            f"  p2 rule safe set:  {_kept(rc['rule'])}",
            f"  p2 model at the rule's coverage: {_kept(rc['model_at_rule_coverage'])}",
            f"  beats the rule: {rc['beats']} ({rc['why']})",
        ]
    lines.append(f"VERDICT: {'PASS' if crit['passed'] else 'FAIL'} ({crit['why']})")
    if "all_tokens_p2_at_threshold" in report:
        lines.append(
            f"all tokens, p2, same threshold: {_kept(report['all_tokens_p2_at_threshold'])}"
        )
    lines.append(
        f"p2 survivors also seen in training: {_pct(report['p2_survivors_seen_in_train'])}"
    )
    target = c["target"]
    lines += ["", f"Top features ({target}, gain share):"]
    for item in report["targets"][target]["importance"]:
        lines.append(f"  {item['feature']:<32} {item['share']:.1%}")
    lines += ["", "Examples (p2 survivors, highest then lowest scores):"]
    for ex in report["examples"]:
        lines.append(
            f"  {ex['mint'][:8]} {ex['venue'] or 'NULL':<10} score {ex['score']:.3f} "
            f"{target}={ex[target]:g}: {format_reasons(ex['reasons'])}"
        )
    return "\n".join(lines) + "\n"


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{v:.1%}"


def _num(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _kept(s: Mapping[str, Any]) -> str:
    return (
        f"{s['kept']}/{s['n']} kept ({_pct(s['coverage'])}), rate {_pct(s['rate'])} "
        f"(95 % upper {_pct(s['rate_upper95'])})"
    )


# --------------------------------------------------------------------------- CLI


def _window(value: str) -> tuple[datetime, datetime]:
    try:
        start, end = value.split("/")
        return _utc(start), _utc(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected START/END (UTC), got {value!r}") from exc


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the anchors dataset (parquet, one file per chunk)")
    b.add_argument("--out-dir", required=True, type=Path)
    b.add_argument("--since", type=_utc, default=DATA_SINCE, help="first row read (UTC)")
    b.add_argument("--anchor-seconds", type=float, default=DEFAULT_ANCHOR_SECONDS)
    b.add_argument("--rows-per-group", type=int, default=3_000_000)
    b.add_argument("--resume", action="store_true", help="skip chunks already written")
    e = sub.add_parser("evaluate", help="train, score and judge the criterion")
    e.add_argument("--dataset", required=True, type=Path)
    e.add_argument("--report", type=Path, help="write the JSON report (and a .txt summary)")
    e.add_argument("--jobs", type=int, default=0, help="LightGBM threads (0: all cores)")
    for cmd in (b, e):
        for name in SEGMENTS:
            cmd.add_argument(f"--{name}", type=_window, default=getattr(DEFAULT_SPLIT, name),
                             help=f"{name} anchors START/END (UTC; default: the pre-registered "
                             "split)")  # fmt: skip
    return p


def split_from(args: argparse.Namespace) -> Split:
    return Split(**{name: getattr(args, name) for name in SEGMENTS})


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings = RugLabelSettings.from_config(load_config())
    split = split_from(args)
    try:
        split.check(max(horizon_seconds(h) for h in settings.horizons))
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.command == "build":
        from .db import make_engine

        build(
            make_engine(), args.out_dir, split, settings, since=args.since,
            anchor_seconds=args.anchor_seconds, rows_per_group=args.rows_per_group,
            resume=args.resume,
        )  # fmt: skip
        return 0
    require_lightgbm()
    frame = load_dataset(args.dataset)
    frame["segment"] = split.segment_of(frame["ts"])  # the split in force, not the build's
    result = evaluate(frame, settings.horizons, n_jobs=args.jobs)
    result.report["split"] = split.describe()
    text = render(result.report)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result.report, indent=2, default=str), encoding="utf-8")
        args.report.with_suffix(".txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
