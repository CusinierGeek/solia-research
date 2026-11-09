"""Rug-guard labels: the sudden rug and the gradual dump, each measured on its own (#259).

The 15m model's ``rug`` label (:mod:`solia_ai.dataset`) fires on any >= threshold fall within the
horizon, mixing a slow decline with the event that ruins a trade: the pool emptied or dumped in
**one transaction**, with no warning. These labels separate the two, for the rug guard and the
survivor study. They change nothing in the model's labels.

Per token, rows in time order, a **step** is a row and the one before it. A step is a *drop* when,
from a tradeable row (at or above its venue's ``labeling.liquidity_floor``,
:mod:`solia_ai.liquidity` — dust pools jump by multiples on a micro-trade), the price or the
liquidity falls by at least ``sudden_drop``, or the pool falls under its venue's floor (a
*drained* pool; curve venues have a floor of 0, so they are never drained, only dumped), **and
stays there** for the next :data:`MAIN_POOL_WINDOW` seconds (or the token stops trading). A token
traded on several pools carries, in each snapshot, the liquidity (and venue) of the pool traded
last, so switching pools must not read as a drop: a liquidity fall by ``sudden_drop`` counts only
between two rows of the **same pool** (``pool_address``; NULL = the venue's curve), staying there on
that pool; a fall under the floor counts across pools when the token stays under it for
:data:`CROSS_POOL_WINDOW` seconds (its main pool emptied, trading left on a dust pool). Either
must start from the token's main pool (:data:`MAIN_POOL_SHARE` of its largest liquidity over the
previous :data:`MAIN_POOL_WINDOW` seconds), not from a side pool. A drop is:

* **sudden** when the two snapshots are at most ``max_step_seconds`` apart. Snapshots are ~10 s
  apart while a token trades, so this approximates one transaction (a drain or a dump inside one
  snapshot interval);
* **ambiguous** when they are further apart (the pipeline wrote nothing in between: a restart, a
  replay boundary, an evicted idle token): it may have been one transaction or many, so it decides
  nothing.

For an anchor row ``i`` (price ``p``) and a horizon ``H``, over the rows ``(t, t + H]``:

* ``sudden_<H>`` = 1 if a sudden drop comes before any ambiguous one; 0 if there is no drop at all
  and the data covers the horizon (same rule as :mod:`solia_ai.dataset`: rows reach ``t + H`` and
  the last row in it is within ``max(max_gap_fraction * H, min_gap_seconds)`` of its end); otherwise
  unknown (NaN). A drained token stops being snapshotted, so a sudden event already seen is labeled
  without coverage.
* ``gradual_<H>`` = 1 if a tradeable price ``<= p * (1 - gradual_drop)`` is reached before the first
  drop (sudden or ambiguous) — the price got there without one big step; 0 if not and either a
  sudden drop came first (the fall, if any, was sudden: decided) or there is no drop and the horizon
  is covered; otherwise NaN. Both can be 1 (a gradual dump, then a drain).
* ``drain_<H>`` = 1 when ``sudden_<H>`` is 1 and its first sudden drop left the pool under its
  floor (emptied), 0 when it left it above (a crash); NaN when ``sudden_<H>`` isn't 1.

Anchors without a usable price or below their floor are not labeled (all NaN).

**Who caused it** is not in ``features``: the pipeline folds remove-liquidity events into the pool
state and keeps no per-row event kind or signer. The price read after an emptied pool is arbitrary
(x0, x1 or x10^7 on the live data), so it can't tell a liquidity removal from a sale of a huge
position either. ``drain`` (pool emptied) vs crash is the most the snapshots support; the creator /
LP-holder attribution needs a pipeline column (remove-liquidity events and their signer) and a
re-replay. Documented in docs/ML-MODEL.md "Rug-guard labels".

Run ``python -m solia_ai.rug_labels --start ... --end ...`` for the quality report (rates by venue
and token age, agreement with the bot's session-5 trades).
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import RUG_LABELS_DEFAULTS, horizon_seconds, load_config, rug_labels_error
from .dataset import DEFAULT_MAX_GAP_FRACTION, DEFAULT_MIN_GAP_SECONDS, _range_extrema
from .liquidity import LiquidityFloor

logger = logging.getLogger(__name__)

# A token traded on several pools gets, in each snapshot, the liquidity of the pool traded last: a
# drop must last MAIN_POOL_WINDOW seconds (a switch to a smaller pool doesn't), and a liquidity
# drop must start from a pool holding at least MAIN_POOL_SHARE of the token's largest liquidity
# over the previous MAIN_POOL_WINDOW seconds (the main pool, not a side pool).
MAIN_POOL_SHARE = 0.5
MAIN_POOL_WINDOW = 60
# A fall under the floor seen on another pool than the row before counts only if the token stays
# under the floor this long: measured 2026-11-06, 109 of 114 such orca cases came back above it
# within 30 min (a dust pool trading while the main pool rested), vs 6 of 89 same-pool pumpswap
# ones.
CROSS_POOL_WINDOW = 1800

# Token-age buckets of the quality report (token_age_sec upper edges, seconds).
AGE_BUCKETS: tuple[tuple[str, float], ...] = (
    ("<5m", 300),
    ("5-30m", 1800),
    ("30m-1h", 3600),
    ("1-6h", 6 * 3600),
    ("6-24h", 24 * 3600),
    (">=24h", math.inf),
)


@dataclass(frozen=True)
class RugLabelSettings:
    """The labels' knobs (module doc); defaults from ``config.yaml`` ``rug_labels`` and the
    ``labeling`` section's gap rule and liquidity floor."""

    horizons: tuple[str, ...] = tuple(RUG_LABELS_DEFAULTS["horizons"])
    sudden_drop: float = RUG_LABELS_DEFAULTS["sudden_drop"]
    gradual_drop: float = RUG_LABELS_DEFAULTS["gradual_drop"]
    max_step_seconds: float = RUG_LABELS_DEFAULTS["max_step_seconds"]
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS
    floor: LiquidityFloor = LiquidityFloor()

    def __post_init__(self) -> None:
        section = {
            "horizons": list(self.horizons),
            "sudden_drop": self.sudden_drop,
            "gradual_drop": self.gradual_drop,
            "max_step_seconds": self.max_step_seconds,
        }
        if problem := rug_labels_error(section):
            raise ValueError(problem)
        if not 0 < self.max_gap_fraction <= 1:
            raise ValueError(f"max_gap_fraction must be in (0, 1], got {self.max_gap_fraction}")
        if self.min_gap_seconds < 0:
            raise ValueError(f"min_gap_seconds must be >= 0, got {self.min_gap_seconds}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> RugLabelSettings:
        section = {**RUG_LABELS_DEFAULTS, **(config.get("rug_labels") or {})}
        if problem := rug_labels_error(section):
            raise ValueError(problem)
        labeling = config.get("labeling", {})
        return cls(
            horizons=tuple(section["horizons"]),
            sudden_drop=float(section["sudden_drop"]),
            gradual_drop=float(section["gradual_drop"]),
            max_step_seconds=float(section["max_step_seconds"]),
            max_gap_fraction=labeling.get("max_gap_fraction", DEFAULT_MAX_GAP_FRACTION),
            min_gap_seconds=labeling.get("min_gap_seconds", DEFAULT_MIN_GAP_SECONDS),
            floor=LiquidityFloor.from_config(config),
        )


def label_columns(horizons: Sequence[str]) -> list[str]:
    return [f"{kind}_{h}" for h in horizons for kind in ("sudden", "gradual", "drain")]


def label_rug_events(
    df: pd.DataFrame,
    settings: RugLabelSettings | None = None,
    *,
    anchor_from: Any = None,
    anchor_to: Any = None,
) -> pd.DataFrame:
    """Label every row of ``df`` (``mint``, ``ts``, ``price``, ``liquidity``; ``venue`` and
    ``token_age_sec`` if present) as an anchor (module doc). Rows only serve as future unless their
    ``ts`` is in ``[anchor_from, anchor_to)`` (each bound optional, UTC).

    Returns one row per anchor, in ``(mint, ts)`` order: ``mint``, ``ts``, ``venue``,
    ``token_age_sec``, ``liquidity``, ``next_sudden_sec`` (seconds to the token's next sudden drop,
    NaN if none in the data), ``next_sudden_drain`` and the :func:`label_columns` (float, NaN =
    unknown)."""
    s = RugLabelSettings() if settings is None else settings
    for col in ("mint", "ts", "price", "liquidity"):
        if col not in df.columns:
            raise ValueError(f"the feature frame needs a {col!r} column")

    codes, _ = pd.factorize(df["mint"], sort=True)
    has_mint = codes >= 0
    ts_ns = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).to_numpy("datetime64[ns]")
    rows = np.flatnonzero(has_mint)
    order = rows[np.lexsort((ts_ns[rows].astype(np.int64), codes[rows]))]

    token = codes[order].astype(np.int64)
    ts_sec = ts_ns[order].astype("datetime64[s]").astype(np.int64)
    price = pd.to_numeric(df["price"].iloc[order], errors="coerce").to_numpy(np.float64)
    liq = pd.to_numeric(df["liquidity"].iloc[order], errors="coerce").to_numpy(np.float64)
    venue = df["venue"].iloc[order].to_numpy(dtype=object) if "venue" in df.columns else None
    n = len(order)

    floor = s.floor.floors(venue, n)
    tradeable = s.floor.tradeable(liq, venue)
    same = np.r_[False, token[1:] == token[:-1]]  # row j continues row j-1's token
    prev = np.r_[0, np.arange(n - 1)]
    rel0 = ts_sec - (ts_sec.min() if n else 0)
    key0 = (token << 32) | rel0
    first_row = np.flatnonzero(np.r_[True, token[1:] != token[:-1]]) if n else np.array([], int)
    token_start = first_row[np.searchsorted(first_row, np.arange(n), side="right") - 1]
    rows = np.arange(n)
    liq_hi = np.where(np.isfinite(liq), liq, -np.inf)
    price_hi = np.where(np.isfinite(price) & tradeable, price, -np.inf)
    # The token's largest pool liquidity over the MAIN_POOL_WINDOW before each row (main pool)...
    back = np.maximum(np.searchsorted(key0, key0 - MAIN_POOL_WINDOW), token_start)
    recent_max = _window_max(back, rows, liq_hi)
    # ... and the most it recovers to over the MAIN_POOL_WINDOW after it (rows (j, j + W]), on any
    # pool and on the row's own pool.
    ahead = np.searchsorted(key0, key0 + MAIN_POOL_WINDOW, side="right")
    price_after = _window_max(rows + 1, ahead, price_hi)
    far = np.searchsorted(key0, key0 + CROSS_POOL_WINDOW, side="right")
    liq_after_far = _window_max(rows + 1, far, liq_hi)
    pool = _pool_codes(df, order, venue)
    same_pool = same & (pool == pool[prev])
    pool_after = _pool_window_max_ahead(token, pool, rel0, liq_hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = ts_sec - ts_sec[prev]
        keep = 1.0 - s.sudden_drop
        price_level, liq_level = price[prev] * keep, liq[prev] * keep
        price_drop = (price[prev] > 0) & (price <= price_level) & (price_after <= price_level)
        drained = (floor > 0) & (liq < floor)  # NaN < x is False: an unknown pool isn't drained
        # The same pool fell by the drop (or under the floor) and stayed there for a minute; or the
        # token fell under the floor on another pool and stayed under it for CROSS_POOL_WINDOW (its
        # main pool emptied, trading left on dust — not a dust pool trading while the main rests).
        fell = same_pool & (liq <= liq_level) & (pool_after <= liq_level)
        emptied = drained & (
            (same_pool & (pool_after < floor)) | (~same_pool & (liq_after_far < floor))
        )
        liq_drop = (liq[prev] > 0) & (fell | emptied)
        main_pool = liq[prev] >= MAIN_POOL_SHARE * recent_max
    pool_drop = main_pool & liq_drop
    drop = same & tradeable[prev] & (price_drop | pool_drop)
    sudden = drop & (gap <= s.max_step_seconds)
    ambiguous = drop & ~sudden

    nxt_sudden = _next_index(sudden)
    nxt_amb = _next_index(ambiguous)

    # Anchors: rows in the anchor range, with a usable price, at or above their floor.
    t_ns = ts_ns[order].astype(np.int64)
    in_range = np.ones(n, bool)
    if anchor_from is not None:
        in_range &= t_ns >= _ns(anchor_from)
    if anchor_to is not None:
        in_range &= t_ns < _ns(anchor_to)
    i = np.flatnonzero(in_range)
    p_i = price[i]
    anchored = np.isfinite(p_i) & (p_i > 0) & tradeable[i]

    # (token, ts) as one sortable key: a horizon's search never leaves its token.
    rel = ts_sec - (ts_sec.min() if n else 0)
    max_h = max(horizon_seconds(h) for h in s.horizons)
    if n and int(rel.max()) + max_h >= 2**32:
        raise ValueError("the features span too long a time range to label")
    key = (token << 32) | rel
    lo = np.searchsorted(key, key[i], side="right")
    token_last = np.r_[np.flatnonzero(token[1:] != token[:-1]), n - 1] if n else np.array([], int)
    last_of_token = token_last[np.searchsorted(token_last, i)] if n else i
    last_sec = ts_sec[last_of_token]

    counted_price = np.where(np.isfinite(price) & tradeable, price, np.inf)
    s_i, a_i = nxt_sudden[i], nxt_amb[i]
    out: dict[str, Any] = {
        "mint": df["mint"].to_numpy(dtype=object)[order[i]],
        "ts": pd.to_datetime(t_ns[i], utc=True),
        "venue": None if venue is None else venue[i],
        "token_age_sec": (
            pd.to_numeric(df["token_age_sec"].iloc[order[i]], errors="coerce").to_numpy(np.float64)
            if "token_age_sec" in df.columns
            else np.full(len(i), np.nan)
        ),
        "liquidity": liq[i],
    }
    s_safe = np.minimum(s_i, max(n - 1, 0))
    has_next = (s_i < n) & (token[s_safe] == token[i])  # the next sudden drop of the same token
    out["next_sudden_sec"] = np.where(has_next, ts_sec[s_safe] - ts_sec[i], np.nan)
    drain_at = pool_drop & drained
    out["next_sudden_drain"] = np.where(has_next, drain_at[s_safe].astype(np.float64), np.nan)

    for h in s.horizons:
        h_s = horizon_seconds(h)
        hi = np.searchsorted(key, key[i] + h_s, side="right")  # rows (t, t + H]
        horizon_end = ts_sec[i] + h_s
        max_gap = max(h_s * s.max_gap_fraction, s.min_gap_seconds)
        future = anchored & (hi > lo)
        covered = future & (last_sec >= horizon_end)
        covered &= ts_sec[np.maximum(hi - 1, 0)] >= horizon_end - max_gap
        s_in, a_in = s_i < hi, a_i < hi
        sudden_first = s_in & (~a_in | (s_i < a_i))  # else an ambiguous drop first: unknown
        quiet = ~s_in & ~a_in  # no drop at all in the horizon

        stop = np.minimum(np.where(s_in, s_i, hi), np.where(a_in, a_i, hi))
        has_rows = future & (stop > lo)
        p_min = np.full(len(i), np.inf)
        if has_rows.any():
            (p_min[has_rows],) = _range_extrema(
                lo[has_rows], stop[has_rows], (counted_price, np.minimum)
            )
        with np.errstate(invalid="ignore"):
            reached = has_rows & (p_min <= p_i * (1.0 - s.gradual_drop))

        sudden_lab = np.full(len(i), np.nan)
        sudden_lab[future & sudden_first] = 1.0
        sudden_lab[covered & quiet] = 0.0
        gradual_lab = np.full(len(i), np.nan)
        gradual_lab[future & ~reached & (sudden_first | (covered & quiet))] = 0.0
        gradual_lab[reached] = 1.0
        drain_lab = np.where(sudden_lab == 1.0, drain_at[s_safe].astype(np.float64), np.nan)
        out[f"sudden_{h}"] = sudden_lab
        out[f"gradual_{h}"] = gradual_lab
        out[f"drain_{h}"] = drain_lab
    return pd.DataFrame(out)


def label_in_groups(
    df: pd.DataFrame,
    settings: RugLabelSettings,
    *,
    anchor_from: Any = None,
    anchor_to: Any = None,
    rows_per_group: int = 3_000_000,
    anchor_every_seconds: float = 0,
) -> pd.DataFrame:
    """:func:`label_rug_events` over groups of whole tokens of about ``rows_per_group`` rows each,
    so the per-row work arrays never span the whole frame; labels stored as float32. With
    ``anchor_every_seconds`` > 0, keeps each token's first anchor of every such bucket of time."""
    if rows_per_group < 1:
        raise ValueError(f"rows_per_group must be >= 1, got {rows_per_group}")
    if anchor_every_seconds < 0:
        raise ValueError(f"anchor_every_seconds must be >= 0, got {anchor_every_seconds}")
    codes, _ = pd.factorize(df["mint"])
    sizes = np.bincount(codes[codes >= 0])
    # Tokens are dealt into groups by cumulative row count (whole tokens, never split).
    group_of_token = (np.cumsum(sizes) - 1) // rows_per_group
    group_of_row = np.where(codes >= 0, group_of_token[np.maximum(codes, 0)], -1)
    parts = []
    for g in np.unique(group_of_row[group_of_row >= 0]):
        labels = label_rug_events(
            df[group_of_row == g], settings, anchor_from=anchor_from, anchor_to=anchor_to
        )
        if anchor_every_seconds > 0 and len(labels):
            bucket = labels["ts"].astype("int64") // int(anchor_every_seconds * 1e9)
            labels = labels[~pd.DataFrame({"m": labels["mint"], "b": bucket}).duplicated()]
        floats = [c for c in labels.columns if labels[c].dtype == np.float64]
        parts.append(labels.astype(dict.fromkeys(floats, np.float32)))
        logger.info("labeled group %d: %d anchors", g, len(labels))
    if not parts:
        return label_rug_events(df.iloc[:0], settings)
    return pd.concat(parts, ignore_index=True)


def _pool_codes(df: pd.DataFrame, order: np.ndarray, venue: np.ndarray | None) -> np.ndarray:
    """``[R]`` int64 pool identity of each row (in ``order``): its ``pool_address``; a NULL
    address is its venue's (the bonding curve); without the column, the venue; else one pool."""
    venues = pd.Series(np.full(len(order), "", object) if venue is None else venue, dtype=object)
    by_venue = ("venue:" + venues.fillna("").map(str)).to_numpy(dtype=object)
    if "pool_address" in df.columns:
        address = df["pool_address"].iloc[order].to_numpy(dtype=object)
        by_venue = np.where(pd.isna(address), by_venue, address)
    return pd.factorize(pd.Series(by_venue, dtype=object))[0].astype(np.int64)


def _pool_window_max_ahead(
    token: np.ndarray, pool: np.ndarray, rel: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Per row, ``max(values)`` over the later rows of the same token and pool within
    :data:`MAIN_POOL_WINDOW` seconds (``-inf``: none)."""
    n = len(token)
    if n == 0:
        return np.empty(0)
    group = pd.factorize(pd.Series(token * (int(pool.max()) + 1) + pool))[0].astype(np.int64)
    perm = np.lexsort((rel, group))  # rows by (token, pool), then time
    key = (group[perm] << 32) | rel[perm]
    ahead = np.searchsorted(key, key + MAIN_POOL_WINDOW, side="right")
    out = np.empty(n)
    out[perm] = _window_max(np.arange(n) + 1, ahead, values[perm])
    return out


def _window_max(lo: np.ndarray, hi: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``max(values[lo:hi])`` per row, ``-inf`` for an empty range."""
    out = np.full(len(lo), -np.inf)
    some = np.flatnonzero(hi > lo)
    if len(some):
        (out[some],) = _range_extrema(lo[some], hi[some], (values, np.maximum))
    return out


def _next_index(flag: np.ndarray) -> np.ndarray:
    """For each row, the index of the first flagged row strictly after it (``len(flag)``: none)."""
    n = len(flag)
    idx = np.where(flag, np.arange(n), n)
    after = np.minimum.accumulate(idx[::-1])[::-1]  # first flagged at or after
    return np.r_[after[1:], n].astype(np.int64)


def _ns(value: Any) -> int:
    """``value`` as epoch nanoseconds (a naive time is UTC)."""
    stamp = pd.Timestamp(value)
    return int((stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp).value)


# --------------------------------------------------------------------------- quality report


def age_bucket(token_age_sec: np.ndarray) -> np.ndarray:
    """Each row's :data:`AGE_BUCKETS` name (``unknown`` for a NULL age)."""
    age = np.asarray(token_age_sec, np.float64)
    out = np.full(len(age), "unknown", dtype=object)
    lower = -math.inf
    for name, upper in AGE_BUCKETS:
        out[(age >= lower) & (age < upper)] = name
        lower = upper
    return out


def rates(labels: pd.DataFrame, by: str, horizons: Sequence[str]) -> pd.DataFrame:
    """Per ``by`` group and horizon: anchors, the share labeled, and among labeled anchors the
    sudden-rug rate, the drained share of the sudden ones, the gradual-dump rate."""
    frames = []
    for h in horizons:
        g = labels.groupby(by, dropna=False)
        frame = pd.DataFrame(
            {
                "horizon": h,
                "anchors": g.size(),
                "labeled": g[f"sudden_{h}"].apply(lambda x: x.notna().mean()),
                "sudden": g[f"sudden_{h}"].mean(),
                "drained_of_sudden": g[f"drain_{h}"].mean(),
                "gradual": g[f"gradual_{h}"].mean(),
            }
        )
        frames.append(frame)
    return pd.concat(frames).reset_index().rename(columns={"index": by})


def read_label_frame(engine: Any, start: datetime, end: datetime, horizon_s: int) -> pd.DataFrame:
    """The rows needed to label anchors in ``[start, end)``: every ``features`` row of
    ``[start, end + horizon]`` (the futures), read in 1-hour ``ts`` slices through the ts index."""
    from sqlalchemy import text

    query = text(
        "SELECT mint, ts, venue, price::float8 AS price, liquidity::float8 AS liquidity, "
        "token_age_sec, pool_address FROM features WHERE ts >= :lo AND ts < :hi"
    )
    stop = end + timedelta(seconds=horizon_s + 1)
    chunks, lo = [], start
    while lo < stop:
        hi = min(lo + timedelta(hours=1), stop)
        with engine.connect() as conn:
            chunk = pd.read_sql(query, conn, params={"lo": lo, "hi": hi})
        for col in ("mint", "venue", "pool_address"):
            chunk[col] = chunk[col].astype("category")
        chunks.append(chunk)
        logger.info("read %s .. %s: %d rows", lo.isoformat(), hi.isoformat(), len(chunk))
        lo = hi
    return concat_categorical(chunks, ("mint", "venue", "pool_address"))


def concat_categorical(chunks: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    """``chunks`` concatenated, ``columns`` kept categorical over one shared set of categories
    (each chunk has its own, of object dtype, or of another dtype when a chunk's values are all
    NULL)."""
    from pandas.api.types import union_categoricals

    frame = pd.concat(chunks, ignore_index=True)
    for col in columns:
        parts = [pd.Categorical(c[col].astype(object)) for c in chunks]
        parts = [p.set_categories(p.categories.astype(object)) for p in parts]
        frame[col] = union_categoricals(parts)
    return frame


def session_agreement(
    labels: pd.DataFrame, trades: pd.DataFrame, horizons: Sequence[str]
) -> pd.DataFrame:
    """For each trade (``mint``, ``opened_at``): its token's labels at the entry (the last anchor at
    or before ``opened_at``) — whether the drain that followed is labeled."""
    labels = labels.assign(
        ts=labels["ts"].astype("datetime64[ns, UTC]"), mint=labels["mint"].astype(object)
    ).sort_values("ts")
    trades = trades.assign(
        opened_at=trades["opened_at"].astype("datetime64[ns, UTC]"),
        mint=trades["mint"].astype(object),
    )
    trades = trades.sort_values("opened_at")
    merged = pd.merge_asof(
        trades, labels, left_on="opened_at", right_on="ts", by="mint", direction="backward"
    )
    keep = ["mint", "opened_at", "ts", "venue", "next_sudden_sec", "next_sudden_drain"]
    keep += [c for h in horizons for c in (f"sudden_{h}", f"drain_{h}", f"gradual_{h}")]
    return merged[keep]


def main(argv: Sequence[str] | None = None) -> int:
    from .db import make_engine

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, type=_utc_arg, help="first anchor (UTC)")
    parser.add_argument("--end", required=True, type=_utc_arg, help="anchors before this (UTC)")
    parser.add_argument("--session", type=int, default=5, help="bot session to check (0: none)")
    parser.add_argument("--out", help="write the labels to this parquet file")
    parser.add_argument(
        "--rows-per-group",
        type=int,
        default=3_000_000,
        help="label the tokens in groups of about this many rows (bounds the peak memory)",
    )
    parser.add_argument(
        "--anchor-every-seconds",
        type=float,
        default=0,
        help="keep at most one anchor per token per this many seconds (0: every row)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    settings = RugLabelSettings.from_config(load_config())
    max_h = max(horizon_seconds(h) for h in settings.horizons)
    engine = make_engine()
    frame = read_label_frame(engine, args.start, args.end, max_h)
    labels = label_in_groups(
        frame,
        settings,
        anchor_from=args.start,
        anchor_to=args.end,
        rows_per_group=args.rows_per_group,
        anchor_every_seconds=args.anchor_every_seconds,
    )
    del frame
    if args.out:
        labels.to_parquet(args.out, index=False)

    labels["age_bucket"] = age_bucket(labels["token_age_sec"].to_numpy())
    labels["venue"] = labels["venue"].fillna("NULL")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    window = f"[{args.start:%Y-%m-%d %H:%M}, {args.end:%Y-%m-%d %H:%M}) UTC"
    print(f"anchors {len(labels):,} in {window}")
    for by in ("venue", "age_bucket"):
        print(f"\n== by {by} ==")
        print(rates(labels, by, settings.horizons).round(4).to_string(index=False))

    if args.session:
        from sqlalchemy import text

        with engine.connect() as conn:
            trades = pd.read_sql(
                text(
                    "SELECT DISTINCT ON (mint) mint, opened_at FROM trades WHERE session_id = :s "
                    "AND opened_at >= :a AND opened_at < :b ORDER BY mint, opened_at"
                ),
                conn,
                params={"s": args.session, "a": args.start, "b": args.end},
            )
        trades["opened_at"] = pd.to_datetime(trades["opened_at"], utc=True)
        agree = session_agreement(labels, trades, settings.horizons)
        print(f"\n== session {args.session}: {len(agree)} traded tokens, labels at entry ==")
        for h in settings.horizons:
            col = agree[f"sudden_{h}"]
            print(
                f"{h}: sudden {int((col == 1).sum())}, none {int((col == 0).sum())}, "
                f"unknown {int(col.isna().sum())}; drained {int((agree[f'drain_{h}'] == 1).sum())}"
            )
        print(
            "minutes from entry to the first sudden drop: median "
            f"{agree['next_sudden_sec'].median() / 60:.1f}, "
            f"p90 {agree['next_sudden_sec'].quantile(0.9) / 60:.1f}"
        )
        missed = agree[agree[f"sudden_{settings.horizons[-1]}"] != 1]
        if len(missed):
            print("\nnot labeled sudden at the longest horizon:")
            print(missed.to_string(index=False))
    return 0


def _utc_arg(value: str) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime().astimezone(UTC)


if __name__ == "__main__":
    raise SystemExit(main())
