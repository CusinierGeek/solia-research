"""Survivor study: what happens to established tokens (#261). Descriptive only: no model, no rule
tuning — any rule is pre-registered and judged later, in the trader task.

* **Population**: per token, its first tradeable row (price > 0, liquidity at or above its venue's
  floor) at ``token_age_sec >= A``, for A in 1 h / 6 h / 24 h (:data:`AGES`) — the *entry*. Split by
  the plain guard "LP burned >= 90 %" (#260 failed its criterion, so the guard is this rule),
  burned < 90 %, burn unknown, and migrated or not, at the entry.
* **Forward return** at 4 h / 24 h / 72 h (:data:`HORIZONS`), held to the horizon and sold at the
  last row at or before it (a token that stopped trading is sold at its last mark and pool), **net
  of the bot's costs** (:class:`Costs`, the exit simulation's execution model: a 1 SOL position,
  constant-product impact on the pool's reserve, venue fee each side, network fees and account
  rent). An emptied pool pays (almost) nothing, whatever price it reads. Only entries whose horizon
  ends before the data does are counted (no censoring bias toward short-lived tokens).
* **Tail**: the share whose net return (sold at that row) reaches +50 % / +100 % **before** it
  reaches -30 % within the horizon, on the snapshots (10 s).
* **CIs**: bootstrap by token (one entry per token and age).
* **Losses**: entries losing >= 30 % at the horizon, by cause from the rug-guard labels at the entry
  (:mod:`solia_ai.rug_labels`, same horizons): a gradual dump first, a sudden rug, neither (a fall
  of 30-50 % or a crash without the 50 % step), unknown.
* **Warning signs** (6 h entries, 24 h, one period, descriptive): for gradual dumps, the hour
  before the dump got under way (the first row 30 % under the price's running peak since the entry)
  against tokens with no drop at all, looked at the same time after their entry: top-10 holders'
  net flow, holder change, buy/sell ratio, volume decay, liquidity trend (:data:`SIGNS`).

Run ``python -m solia_ai.survivor_study --out docs/reports/survivor-study-261.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import horizon_seconds, load_config
from .exit_sim import ACCOUNT_RENT_SOL, BOT_EXECUTION, FEE_PCT, NETWORK_FEE_SOL, VIRTUAL_SOL
from .liquidity import LiquidityFloor, normalize_venue
from .rug_labels import RugLabelSettings, label_rug_events

logger = logging.getLogger(__name__)

AGES: dict[str, int] = {"1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}
HORIZONS: tuple[str, ...] = ("4h", "24h", "72h")
TARGETS: tuple[float, ...] = (0.5, 1.0)
STOP = -0.3
LOSS = -0.3
GUARD_MIN_BURNED = 90.0
DUMP_DRAWDOWN = 0.3
SIGN_WINDOW = 3600
SIGN_AGE, SIGN_HORIZON = "6h", "24h"
SIGNS: tuple[str, ...] = (
    "top10_net_flow",
    "holder_change",
    "buy_sell_ratio",
    "volume_decay",
    "liquidity_trend",
)
DEFAULT_BOOTSTRAP = 1000
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


# --------------------------------------------------------------------------- costs


@dataclass(frozen=True)
class Costs:
    """The bot's round trip (exit_sim's realistic execution): ``size`` SOL in, constant-product
    impact on each side's reserve (liquidity + the venue's virtual SOL), the venue's fee each
    side, two network fees and the token account's rent."""

    size: float = BOT_EXECUTION.position_size
    network_fee: float = NETWORK_FEE_SOL
    account_rent: float = ACCOUNT_RENT_SOL
    fee_pct: Mapping[str, float] = field(default_factory=lambda: dict(FEE_PCT))
    virtual_sol: Mapping[str, float] = field(default_factory=lambda: dict(VIRTUAL_SOL))

    def fee(self, venue: np.ndarray) -> np.ndarray:
        default = self.fee_pct.get("default", 0.0)
        return _per_venue(venue, lambda v: self.fee_pct.get(v, default))

    def reserve(self, liquidity: np.ndarray, venue: np.ndarray) -> np.ndarray:
        virtual = _per_venue(venue, lambda v: self.virtual_sol.get(v, 0.0))
        liquidity = np.asarray(liquidity, np.float64)
        with np.errstate(invalid="ignore"):
            return np.where(liquidity >= 0, liquidity + virtual, np.nan)


FREE = Costs(size=1e-9, network_fee=0.0, account_rent=0.0, fee_pct={"default": 0.0}, virtual_sol={})


def _per_venue(venue: np.ndarray, value: Callable[[str], float]) -> np.ndarray:
    codes, uniques = pd.factorize(pd.Series(np.asarray(venue, dtype=object)), use_na_sentinel=True)
    table = np.array([value(normalize_venue(u)) for u in uniques] + [value("")], np.float64)
    return table[np.where(codes < 0, len(uniques), codes)]


def net_return(
    entry_mark: np.ndarray,
    entry_reserve: np.ndarray,
    entry_fee: np.ndarray,
    exit_mark: np.ndarray,
    exit_reserve: np.ndarray,
    exit_fee: np.ndarray,
    costs: Costs,
) -> np.ndarray:
    """Net return of buying ``costs.size`` SOL at the entry and selling it all at the exit, with
    exit_sim's fill formulas (buy and sell along the pool's constant product, fee each side) and
    the fixed costs. An empty (or unknown) exit reserve pays nothing: -100 % minus the costs."""
    s = costs.size
    keep = 1.0 - np.asarray(entry_fee, np.float64)
    entry_reserve = np.asarray(entry_reserve, np.float64)
    exit_reserve = np.asarray(exit_reserve, np.float64)
    exit_mark = np.asarray(exit_mark, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        entry_fill = entry_mark * (entry_reserve + s * keep) / (entry_reserve * keep)
        value = s * exit_mark / entry_fill  # the position at the exit mark
        sold = exit_mark * exit_reserve * (1.0 - exit_fee) / (exit_reserve + value)
        exit_fill = np.where((exit_reserve > 0) & np.isfinite(sold), sold, 0.0)
        proceeds = s / entry_fill * exit_fill
    fixed = 2 * costs.network_fee + costs.account_rent
    return (proceeds - s - fixed) / s


# --------------------------------------------------------------------------- frame & entries


@dataclass
class Paths:
    """Every row of the population, by token then time: arrays for the vectorized outcomes."""

    mint: np.ndarray  # object
    code: np.ndarray  # int64 token code
    ts: np.ndarray  # int64 epoch seconds
    price: np.ndarray  # float64, carried forward over unusable rows (NaN before the first)
    liquidity: np.ndarray  # float64, carried forward
    venue: np.ndarray  # object, carried forward
    age: np.ndarray  # float64
    frame: pd.DataFrame  # the sorted frame (sign columns)

    @classmethod
    def of(cls, frame: pd.DataFrame) -> Paths:
        df = frame.assign(
            mint=frame["mint"].astype(object),
            ts=pd.to_datetime(frame["ts"], utc=True).dt.as_unit("ns"),
        ).sort_values(["mint", "ts"], kind="stable")
        df = df.reset_index(drop=True)
        code = pd.factorize(df["mint"])[0].astype(np.int64)
        price = pd.to_numeric(df["price"], errors="coerce").astype(np.float64)
        price = price.where(price > 0)
        grouped = pd.DataFrame(
            {
                "price": price,
                "liquidity": pd.to_numeric(df["liquidity"], errors="coerce").astype(np.float64),
                "venue": df["venue"].astype(object),
            }
        ).groupby(code, sort=False)
        filled = grouped.ffill()
        return cls(
            mint=df["mint"].to_numpy(object),
            code=code,
            ts=df["ts"].astype("int64").to_numpy() // 1_000_000_000,
            price=filled["price"].to_numpy(np.float64),
            liquidity=filled["liquidity"].to_numpy(np.float64),
            venue=filled["venue"].to_numpy(object),
            age=pd.to_numeric(df["token_age_sec"], errors="coerce").to_numpy(np.float64),
            frame=df,
        )

    def __len__(self) -> int:
        return len(self.code)

    def last_row_at_or_before(self, code: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Per (token, time), the index of its last row at or before ``ts`` (its first row's index
        minus one when none)."""
        base = int(self.ts.min()) if len(self) else 0
        key = (self.code << 32) + (self.ts - base)  # time offsets stay well within +-2^31 s
        query = (np.asarray(code, np.int64) << 32) + (np.asarray(ts, np.int64) - base)
        return np.searchsorted(key, query, side="right") - 1


def entries(paths: Paths, min_age: float, floor: LiquidityFloor) -> pd.DataFrame:
    """Per token, its first row at ``token_age_sec >= min_age`` with a price and a tradeable pool:
    ``row``, ``mint``, ``code``, ``ts`` (epoch s), ``price``, ``liquidity``, ``venue``,
    ``burned_liquidity``, ``is_migrated``."""
    ok = (paths.age >= min_age) & np.isfinite(paths.price)
    ok &= floor.tradeable(paths.liquidity, paths.venue)
    rows = np.flatnonzero(ok)
    rows = rows[np.r_[True, paths.code[rows][1:] != paths.code[rows][:-1]]] if len(rows) else rows
    df = paths.frame
    return pd.DataFrame(
        {
            "row": rows,
            "mint": paths.mint[rows],
            "code": paths.code[rows],
            "ts": paths.ts[rows],
            "price": paths.price[rows],
            "liquidity": paths.liquidity[rows],
            "venue": paths.venue[rows],
            "burned_liquidity": _column(df, "burned_liquidity", rows),
            "is_migrated": _column(df, "is_migrated", rows),
        }
    )


def _column(df: pd.DataFrame, name: str, rows: np.ndarray) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(rows), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)[rows]


def guard_group(burned: np.ndarray) -> np.ndarray:
    """``guard-safe`` (LP burned >= 90 %), ``burned<90`` or ``burn-unknown``."""
    burned = np.asarray(burned, np.float64)
    return np.where(
        np.isnan(burned),
        "burn-unknown",
        np.where(burned >= GUARD_MIN_BURNED, "guard-safe", "burned<90"),
    )


# --------------------------------------------------------------------------- outcomes


def outcomes(
    paths: Paths,
    entry: pd.DataFrame,
    horizons: Sequence[str],
    costs: Costs,
    data_end: int,
    *,
    targets: Sequence[float] = TARGETS,
    stop: float = STOP,
) -> pd.DataFrame:
    """Per entry and horizon (columns ``<metric>_<H>``): ``covered`` (the horizon ends before
    ``data_end``, epoch s), the ``net`` and ``gross`` return held to it, and ``up<k>``: the net
    return (sold at that row) reached ``+k`` before ``stop`` within it (k in percent)."""
    out = pd.DataFrame(index=entry.index)
    if not len(entry):
        return out
    row = entry["row"].to_numpy(np.int64)
    code = entry["code"].to_numpy(np.int64)
    t0 = entry["ts"].to_numpy(np.int64)
    fee0 = costs.fee(entry["venue"].to_numpy(object))
    res0 = costs.reserve(entry["liquidity"].to_numpy(np.float64), entry["venue"].to_numpy(object))
    p0 = entry["price"].to_numpy(np.float64)

    # Every row's net return if sold there, against its token's entry (rows before it: NaN).
    of_row = np.full(int(paths.code.max()) + 1 if len(paths) else 0, -1, np.int64)
    of_row[code] = np.arange(len(entry))
    e = of_row[paths.code]
    has = e >= 0
    after = has & (np.arange(len(paths)) > np.where(has, row[np.maximum(e, 0)], 0))
    net_row = np.full(len(paths), np.nan)
    idx = np.flatnonzero(after)
    ee = e[idx]
    net_row[idx] = net_return(
        p0[ee], res0[ee], fee0[ee], paths.price[idx],
        costs.reserve(paths.liquidity[idx], paths.venue[idx]), costs.fee(paths.venue[idx]), costs,
    )  # fmt: skip
    nxt_stop = _next_flag(net_row <= stop)
    nxt_up = {k: _next_flag(net_row >= k) for k in targets}

    for h in horizons:
        hs = horizon_seconds(h)
        last = paths.last_row_at_or_before(code, t0 + hs)  # >= row: the entry itself is <= t0
        covered = t0 + hs <= data_end
        exit_fee = costs.fee(paths.venue[last])
        exit_res = costs.reserve(paths.liquidity[last], paths.venue[last])
        net = net_return(p0, res0, fee0, paths.price[last], exit_res, exit_fee, costs)
        out[f"covered_{h}"] = covered
        out[f"net_{h}"] = np.where(covered, net, np.nan)
        out[f"gross_{h}"] = np.where(covered, paths.price[last] / p0 - 1.0, np.nan)
        first_stop = nxt_stop[np.minimum(row + 1, len(paths))]
        for k in targets:
            first_up = nxt_up[k][np.minimum(row + 1, len(paths))]
            hit = (first_up <= last) & ((first_up < first_stop) | (first_stop > last))
            out[f"up{int(round(k * 100))}_{h}"] = np.where(covered, hit.astype(float), np.nan)
    return out


def _next_flag(flag: np.ndarray) -> np.ndarray:
    """``[n + 1]``: for each index (and ``n``), the first flagged index at or after it (``n``:
    none)."""
    n = len(flag)
    idx = np.where(flag, np.arange(n), n)
    return np.r_[np.minimum.accumulate(idx[::-1])[::-1], n].astype(np.int64)


def loss_causes(net: np.ndarray, sudden: np.ndarray, gradual: np.ndarray) -> pd.Series:
    """For entries losing at least :data:`LOSS`, the cause by the labels at the entry: ``gradual``
    (the price got 50 % down before any drop), ``sudden``, ``neither`` (both 0), ``unknown``."""
    net = np.asarray(net, np.float64)
    sudden = np.asarray(sudden, np.float64)
    gradual = np.asarray(gradual, np.float64)
    lost = net <= LOSS
    cause = np.where(
        gradual == 1,
        "gradual",
        np.where(
            sudden == 1,
            "sudden",
            np.where((gradual == 0) & (sudden == 0), "neither", "unknown"),
        ),
    )
    return (
        pd.Series(cause[lost])
        .value_counts()
        .reindex(["gradual", "sudden", "neither", "unknown"], fill_value=0)
    )


# --------------------------------------------------------------------------- statistics


def bootstrap_ci(
    values: np.ndarray,
    stat: Callable[[np.ndarray], np.ndarray],
    *,
    samples: int = DEFAULT_BOOTSTRAP,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Percentile bootstrap CI of ``stat`` (applied along axis 1 of the resampled matrix), one
    value per token; ``None`` with fewer than 2 values."""
    values = np.asarray(values, np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    draws = stat(values[rng.integers(0, len(values), size=(samples, len(values)))])
    alpha = (1.0 - ci) / 2
    lo, hi = np.quantile(draws, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _mean(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1)


def _median(x: np.ndarray) -> np.ndarray:
    return np.median(x, axis=1)


def summarize(
    rows: pd.DataFrame, horizon: str, *, samples: int = DEFAULT_BOOTSTRAP, seed: int = 0
) -> dict[str, Any]:
    """The net-return distribution and tail shares of ``rows`` (covered entries) at ``horizon``."""
    rows = rows[rows[f"covered_{horizon}"]]
    net = rows[f"net_{horizon}"].to_numpy(np.float64)
    out: dict[str, Any] = {"n": int(len(net))}
    if not len(net):
        return out
    out["mean"] = float(net.mean())
    out["mean_ci"] = bootstrap_ci(net, _mean, samples=samples, seed=seed)
    out["median"] = float(np.median(net))
    out["median_ci"] = bootstrap_ci(net, _median, samples=samples, seed=seed)
    out["quantiles"] = {f"q{int(q * 100)}": float(np.quantile(net, q)) for q in QUANTILES}
    out["positive"] = float((net > 0).mean())
    out["gross_median"] = float(np.median(rows[f"gross_{horizon}"]))
    for k in TARGETS:
        col = f"up{int(round(k * 100))}_{horizon}"
        hits = rows[col].to_numpy(np.float64)
        out[col.split("_")[0]] = float(hits.mean())
        out[col.split("_")[0] + "_ci"] = bootstrap_ci(hits, _mean, samples=samples, seed=seed)
    return out


# --------------------------------------------------------------------------- warning signs


def dump_onsets(paths: Paths, entry: pd.DataFrame, horizon: str) -> np.ndarray:
    """Per entry, the epoch second of its first row within the horizon whose price is
    :data:`DUMP_DRAWDOWN` under its running peak since the entry (``-1``: none)."""
    out = np.full(len(entry), -1, np.int64)
    hs = horizon_seconds(horizon)
    for i, (row, code, t0) in enumerate(entry[["row", "code", "ts"]].to_numpy(np.int64)):
        end = paths.last_row_at_or_before(np.array([code]), np.array([t0 + hs]))[0]
        price = paths.price[row : end + 1]
        if len(price) < 2:
            continue
        peak = np.fmax.accumulate(price)
        hit = np.flatnonzero(price <= peak * (1.0 - DUMP_DRAWDOWN))
        if len(hit):
            out[i] = paths.ts[row + hit[0]]
    return out


def signs_at(paths: Paths, code: int, at: int, window: int = SIGN_WINDOW) -> dict[str, float]:
    """The :data:`SIGNS` over the token's rows in ``[at - window, at]`` (NaN when not computable):
    mean top-10 holders' net flow, holder change, mean buy/sell ratio, volume decay (mean of the
    last quarter over the first quarter) and liquidity trend (last / first - 1)."""
    lo, hi = paths.last_row_at_or_before(np.array([code, code]), np.array([at - window - 1, at]))
    rows = np.arange(lo + 1, hi + 1)
    rows = rows[paths.code[rows] == code] if len(rows) else rows
    nan = dict.fromkeys(SIGNS, math.nan)
    if len(rows) < 2:
        return nan
    df = paths.frame

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(rows), np.nan)
        return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)[rows]

    ts = paths.ts[rows]
    holders = col("holders")
    volume = col("volume")
    liq = paths.liquidity[rows]
    early = ts <= at - window * 3 // 4
    late = ts >= at - window // 4
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "top10_net_flow": _nanmean(col("hs_top10_net_flow")),
            "holder_change": _ratio(holders[-1], holders[0]) - 1.0,
            "buy_sell_ratio": _nanmean(col("buy_sell_ratio")),
            "volume_decay": _ratio(_nanmean(volume[late]), _nanmean(volume[early])),
            "liquidity_trend": _ratio(liq[-1], liq[0]) - 1.0,
        }


def _nanmean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else math.nan


def _ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else math.nan


def dump_and_quiet(entry: pd.DataFrame, horizon: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Covered entries with a gradual dump (``gradual_<H>`` = 1), and with no drop at all
    (``sudden_<H>`` = ``gradual_<H>`` = 0)."""
    covered = entry[f"covered_{horizon}"].to_numpy(bool)
    gradual = entry[f"gradual_{horizon}"].to_numpy(np.float64)
    sudden = entry[f"sudden_{horizon}"].to_numpy(np.float64)
    return entry[covered & (gradual == 1)], entry[covered & (gradual == 0) & (sudden == 0)]


def dump_signs(paths: Paths, entry: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Per gradual dump with an onset: ``mint``, ``offset`` (onset - entry, s) and the
    :data:`SIGNS` over the hour before the onset."""
    dumps, _ = dump_and_quiet(entry, horizon)
    onset = dump_onsets(paths, dumps, horizon)
    keep = onset >= 0
    dumps, onset = dumps[keep], onset[keep]
    signs = [signs_at(paths, int(c), int(t)) for c, t in zip(dumps["code"], onset, strict=True)]
    out = pd.DataFrame(signs, columns=list(SIGNS))
    out.insert(0, "offset", onset - dumps["ts"].to_numpy(np.int64))
    out.insert(0, "mint", dumps["mint"].to_numpy(object))
    return out


def control_offsets(controls: pd.Series, offsets: np.ndarray) -> pd.Series:
    """Each no-drop token (``controls``: mints) gets a dump's offset: the sorted offsets dealt in
    mint order, so the controls are looked at the same times after their entry as the dumps."""
    offsets = np.sort(np.asarray(offsets, np.int64))
    order = np.argsort(controls.astype(str).to_numpy(), kind="stable")
    dealt = np.empty(len(controls), np.int64)
    dealt[order] = offsets[np.arange(len(controls)) % len(offsets)]
    return pd.Series(dealt, index=controls.to_numpy(object))


def control_signs(
    paths: Paths, entry: pd.DataFrame, horizon: str, offset_of: pd.Series
) -> pd.DataFrame:
    """The :data:`SIGNS` of this partition's no-drop tokens at their dealt offset."""
    _, quiet = dump_and_quiet(entry, horizon)
    quiet = quiet[quiet["mint"].isin(offset_of.index)]
    at = quiet["ts"].to_numpy(np.int64) + offset_of.reindex(quiet["mint"]).to_numpy(np.int64)
    signs = [signs_at(paths, int(c), int(t)) for c, t in zip(quiet["code"], at, strict=True)]
    out = pd.DataFrame(signs, columns=list(SIGNS))
    out.insert(0, "mint", quiet["mint"].to_numpy(object))
    return out


def compare_signs(dumps: pd.DataFrame, controls: pd.DataFrame) -> dict[str, Any]:
    """Per sign, the medians before the dumps and on the controls, and the AUC of the sign
    separating them (0.5 = no signal; < 0.5: lower before dumps)."""
    from sklearn.metrics import roc_auc_score

    result: dict[str, Any] = {"dumps": int(len(dumps)), "controls": int(len(controls))}
    if not len(dumps) or not len(controls):
        return result
    result["onset_minutes_median"] = float(dumps["offset"].median() / 60)
    result["signs"] = {}
    for sign in SIGNS:
        a, b = dumps[sign].dropna().to_numpy(), controls[sign].dropna().to_numpy()
        auc = None
        if len(a) and len(b):
            auc = float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
        result["signs"][sign] = {
            "n_dumps": int(len(a)),
            "n_controls": int(len(b)),
            "median_dumps": float(np.median(a)) if len(a) else None,
            "median_controls": float(np.median(b)) if len(b) else None,
            "auc": auc,
        }
    return result


def warning_signs(paths: Paths, entry: pd.DataFrame, horizon: str) -> dict[str, Any]:
    """:func:`compare_signs` of one partition holding every token."""
    dumps = dump_signs(paths, entry, horizon)
    _, quiet = dump_and_quiet(entry, horizon)
    if not len(dumps) or not len(quiet):
        return {"dumps": int(len(dumps)), "controls": int(len(quiet))}
    offset_of = control_offsets(quiet["mint"], dumps["offset"].to_numpy())
    return compare_signs(dumps, control_signs(paths, entry, horizon, offset_of))


# --------------------------------------------------------------------------- the study


def entry_labels(
    paths: Paths, keys: pd.DataFrame, settings: RugLabelSettings, rows_per_group: int = 3_000_000
) -> pd.DataFrame:
    """The rug-guard labels at the rows ``keys`` (``mint``, ``ts``), token group by token group."""
    df = paths.frame
    sizes = np.bincount(paths.code)
    group_of_token = (np.cumsum(sizes) - 1) // rows_per_group
    group = group_of_token[paths.code]
    wanted = keys[["mint", "ts"]].assign(mint=keys["mint"].astype(object)).drop_duplicates()
    cols = [f"{k}_{h}" for h in settings.horizons for k in ("sudden", "gradual", "drain")]
    parts = []
    for g in np.unique(group):
        lab = label_rug_events(df[group == g], settings)
        lab = lab.assign(
            mint=lab["mint"].astype(object),
            ts=pd.to_datetime(lab["ts"], utc=True).dt.as_unit("ns").astype("int64") // 10**9,
        )
        parts.append(lab[["mint", "ts", *cols]].merge(wanted, on=["mint", "ts"]))
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["mint", "ts"])


def partition_entries(
    paths: Paths,
    config: Mapping[str, Any],
    costs: Costs,
    horizons: Sequence[str],
    data_end: int,
) -> dict[str, pd.DataFrame]:
    """Per age, this partition's entries with their outcomes and rug-guard labels."""
    floor = LiquidityFloor.from_config(config)
    settings = replace(RugLabelSettings.from_config(config), horizons=tuple(horizons))
    entry_sets = {}
    for age, min_age in AGES.items():
        entry = entries(paths, min_age, floor)
        entry_sets[age] = entry.join(outcomes(paths, entry, horizons, costs, data_end))
    keys = pd.concat([e[["mint", "ts"]] for e in entry_sets.values()]).drop_duplicates()
    if not len(keys):
        return entry_sets
    labels = entry_labels(paths, keys, settings)
    return {
        age: e.merge(labels, on=["mint", "ts"], how="left").set_index(e.index)
        for age, e in entry_sets.items()
    }


def build_report(
    entry_sets: Mapping[str, pd.DataFrame],
    signs: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    horizons: Sequence[str],
    samples: int,
) -> dict[str, Any]:
    """The report from every partition's entries (one per token and age) and the signs."""
    report: dict[str, Any] = {**meta, "ages": {}}
    for age, entry in entry_sets.items():
        groups = {
            "all": np.ones(len(entry), bool),
            **{g: guard_group(entry["burned_liquidity"]) == g
               for g in ("guard-safe", "burned<90", "burn-unknown")},
            "migrated": entry["is_migrated"].to_numpy() == 1,
            "not-migrated": entry["is_migrated"].to_numpy() != 1,
        }  # fmt: skip
        block: dict[str, Any] = {"entries": int(len(entry)), "groups": {}, "loss_causes": {}}
        for name, mask in groups.items():
            rows = entry[mask]
            block["groups"][name] = {h: summarize(rows, h, samples=samples) for h in horizons}
        for h in horizons:
            cov = entry[entry[f"covered_{h}"].to_numpy(bool)]
            causes = loss_causes(cov[f"net_{h}"], cov[f"sudden_{h}"], cov[f"gradual_{h}"])
            block["loss_causes"][h] = {k: int(v) for k, v in causes.items()}
        report["ages"][age] = block
    report["warning_signs"] = {"age": SIGN_AGE, "horizon": SIGN_HORIZON, **signs}
    return report


def _meta(costs: Costs, rows: int, tokens: int, span: Sequence[int] | None) -> dict[str, Any]:
    return {
        "rows": int(rows),
        "tokens": int(tokens),
        "data": None if span is None else [int(span[0]), int(span[1])],
        "costs": {"size_sol": costs.size, "fee_pct": dict(costs.fee_pct),
                  "network_fee": costs.network_fee, "account_rent": costs.account_rent},
    }  # fmt: skip


def study(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    costs: Costs | None = None,
    horizons: Sequence[str] = HORIZONS,
    samples: int = DEFAULT_BOOTSTRAP,
    data_end: int | None = None,
) -> dict[str, Any]:
    """The whole study on ``frame`` held at once (every row of tokens reaching 1 h: ``mint``,
    ``ts``, ``price``, ``liquidity``, ``venue``, ``pool_address``, ``token_age_sec``, the entry
    splits and the sign columns); :func:`study_partitions` for data that doesn't fit."""
    costs = costs or Costs()
    paths = Paths.of(frame)
    end = int(paths.ts.max()) if data_end is None else int(data_end)
    entry_sets = partition_entries(paths, config, costs, horizons, end)
    signs = warning_signs(paths, entry_sets[SIGN_AGE], SIGN_HORIZON)
    tokens = int(paths.code.max()) + 1 if len(paths) else 0
    span = [int(paths.ts.min()), end] if len(paths) else None
    return build_report(entry_sets, signs, meta=_meta(costs, len(paths), tokens, span),
                        horizons=horizons, samples=samples)  # fmt: skip


def study_partitions(
    files: Sequence[Path],
    config: Mapping[str, Any],
    data_end: int,
    *,
    costs: Costs | None = None,
    horizons: Sequence[str] = HORIZONS,
    samples: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """The study over token partitions (each file holds whole tokens): entries, outcomes, labels
    and the dumps' signs per partition; then the controls' signs at offsets dealt from every
    partition's dumps (a second pass); then the report over all entries."""
    costs = costs or Costs()
    sets: dict[str, list[pd.DataFrame]] = {age: [] for age in AGES}
    dumps, rows, tokens, lo = [], 0, 0, None
    for path in files:
        paths = Paths.of(pd.read_parquet(path))
        if not len(paths):
            continue
        part = partition_entries(paths, config, costs, horizons, data_end)
        for age, entry in part.items():
            sets[age].append(entry.drop(columns=["row", "code"]))
        dumps.append(dump_signs(paths, part[SIGN_AGE], SIGN_HORIZON))
        rows, tokens = rows + len(paths), tokens + int(paths.code.max()) + 1
        lo = int(paths.ts.min()) if lo is None else min(lo, int(paths.ts.min()))
        logger.info("partition %s: %d rows, %d tokens", path.name, len(paths), tokens)
    entry_sets = {age: pd.concat(parts, ignore_index=True) for age, parts in sets.items()}
    dump_rows = pd.concat(dumps, ignore_index=True)
    _, quiet = dump_and_quiet(entry_sets[SIGN_AGE], SIGN_HORIZON)
    controls = []
    if len(dump_rows) and len(quiet):
        offset_of = control_offsets(quiet["mint"], dump_rows["offset"].to_numpy())
        for path in files:
            paths = Paths.of(pd.read_parquet(path))
            if not len(paths):
                continue
            mine = entries(paths, AGES[SIGN_AGE], LiquidityFloor.from_config(config))
            extra = list(mine.columns.difference(["mint", "ts"]))
            known = entry_sets[SIGN_AGE].drop(columns=extra, errors="ignore")
            mine = mine.merge(known, on=["mint", "ts"])
            controls.append(control_signs(paths, mine, SIGN_HORIZON, offset_of))
    ctrl_rows = pd.concat(controls, ignore_index=True) if controls else dump_rows.iloc[:0]
    signs = compare_signs(dump_rows, ctrl_rows)
    span = None if lo is None else [lo, data_end]
    return build_report(entry_sets, signs, meta=_meta(costs, rows, tokens, span),
                        horizons=horizons, samples=samples)  # fmt: skip


# --------------------------------------------------------------------------- report


def render(report: Mapping[str, Any]) -> str:
    c = report["costs"]
    lines = [
        "Survivor study (#261): net returns of established tokens, held to the horizon",
        f"rows {report['rows']:,}, tokens {report['tokens']:,}; costs: {c['size_sol']:g} SOL "
        "position, pool impact, venue fee each side, network fees + rent",
        "",
    ]
    for age, block in report["ages"].items():
        lines.append(f"== entry at age >= {age}: {block['entries']} tokens ==")
        lines.append(
            f"{'group':<13} {'H':>4} {'n':>6} {'mean [95% CI]':>24} {'median [95% CI]':>24} "
            f"{'q10':>7} {'q90':>7} {'>0':>6} {'+50%':>6} {'+100%':>6}"
        )
        for group, by_h in block["groups"].items():
            for h, s in by_h.items():
                if not s.get("n"):
                    lines.append(f"{group:<13} {h:>4} {0:>6}")
                    continue
                lines.append(
                    f"{group:<13} {h:>4} {s['n']:>6} {_ci(s['mean'], s['mean_ci']):>24} "
                    f"{_ci(s['median'], s['median_ci']):>24} {_p(s['quantiles']['q10']):>7} "
                    f"{_p(s['quantiles']['q90']):>7} {_p(s['positive']):>6} {_p(s['up50']):>6} "
                    f"{_p(s['up100']):>6}"
                )
        lines.append(
            "losses >= 30 % at the horizon by cause (gradual / sudden / neither / unknown):"
        )
        for h, causes in block["loss_causes"].items():
            total = sum(causes.values())
            share = " / ".join(f"{causes[k]} ({_p(causes[k] / total if total else None)})"
                               for k in ("gradual", "sudden", "neither", "unknown"))  # fmt: skip
            lines.append(f"  {h}: {total} losses: {share}")
        lines.append("")
    w = report["warning_signs"]
    lines.append(
        f"== warning signs: {w['age']} entries, gradual dumps within {w['horizon']} "
        f"({w['dumps']}) vs no-drop tokens ({w['controls']}), the hour before the onset =="
    )
    if "signs" in w:
        lines.append(f"median onset {w['onset_minutes_median']:.0f} min after the entry")
        lines.append(f"{'sign':<18} {'dumps':>10} {'controls':>10} {'AUC':>6}  (n dumps/controls)")
        for sign, s in w["signs"].items():
            lines.append(
                f"{sign:<18} {_n(s['median_dumps']):>10} {_n(s['median_controls']):>10} "
                f"{_n(s['auc']):>6}  ({s['n_dumps']}/{s['n_controls']})"
            )
    return "\n".join(lines) + "\n"


def _p(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def _ci(v: float, ci: Sequence[float] | None) -> str:
    if ci is None:
        return f"{v:+.1%}"
    return f"{v:+.1%} [{ci[0]:+.1%}, {ci[1]:+.1%}]"


def _n(v: Any) -> str:
    return "n/a" if v is None else f"{v:.3g}"


# --------------------------------------------------------------------------- data & CLI


SIGN_COLUMNS = ("holders", "volume", "buy_sell_ratio", "hs_top10_net_flow")


def partition_of(mint: Any, parts: int) -> np.ndarray:
    """A token's partition, stable across reads (a hash of the mint)."""
    text_mint = pd.Series(np.asarray(mint, dtype=object).astype(str), dtype=object)
    hashed = pd.util.hash_pandas_object(text_mint, index=False).to_numpy()
    return (hashed % np.uint64(parts)).astype(np.int64)


def _schema() -> Any:
    import pyarrow as pa

    return pa.schema(
        [("mint", pa.string()), ("ts", pa.timestamp("us", tz="UTC")), ("venue", pa.string()),
         ("pool_address", pa.string()), ("price", pa.float64()), ("liquidity", pa.float64()),
         ("token_age_sec", pa.float64()), ("burned_liquidity", pa.float32()),
         ("is_migrated", pa.float32()), *[(c, pa.float32()) for c in SIGN_COLUMNS]]
    )  # fmt: skip


def read_partitions(
    engine: Any, start: datetime, end: datetime, min_age: int, out_dir: Path, parts: int
) -> list[Path]:
    """Every ``features`` row at ``token_age_sec >= min_age`` in ``[start, end)``, read in 1-hour
    slices and appended to ``parts`` parquet files by token (:func:`partition_of`), so no step
    holds the whole population."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sqlalchemy import text

    signs = ", ".join(f"{c}::float4 AS {c}" for c in SIGN_COLUMNS)
    query = text(
        "SELECT mint, ts, venue, pool_address, price::float8 AS price, liquidity::float8 AS "
        "liquidity, token_age_sec::float8 AS token_age_sec, burned_liquidity::float4 AS "
        f"burned_liquidity, is_migrated::int::float4 AS is_migrated, {signs} FROM features "
        "WHERE ts >= :lo AND ts < :hi AND token_age_sec >= :age"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = _schema()
    files = [out_dir / f"part-{k:02d}.parquet" for k in range(parts)]
    writers = [pq.ParquetWriter(f, schema) for f in files]
    try:
        lo = start
        while lo < end:
            hi = min(lo + timedelta(hours=1), end)
            with engine.connect() as conn:
                chunk = pd.read_sql(query, conn, params={"lo": lo, "hi": hi, "age": min_age})
            part = partition_of(chunk["mint"], parts)
            for k, writer in enumerate(writers):
                rows = chunk[part == k]
                if len(rows):
                    writer.write_table(pa.Table.from_pandas(rows, schema, preserve_index=False))
            logger.info("read %s: %d rows", lo.isoformat(), len(chunk))
            lo = hi
    finally:
        for writer in writers:
            writer.close()
    return files


def data_end_of(files: Sequence[Path]) -> int:
    """The latest row's epoch second over the partition files."""
    import pyarrow.parquet as pq

    latest = [pq.read_table(f, columns=["ts"]).column("ts").to_pandas().max() for f in files]
    return int(max(pd.Timestamp(t) for t in latest if pd.notna(t)).value // 10**9)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", type=_utc, default=None, help="first row (UTC; default: all)")
    p.add_argument("--end", type=_utc, default=None, help="rows before this (UTC; default: now)")
    p.add_argument("--samples", type=int, default=DEFAULT_BOOTSTRAP, help="bootstrap resamples")
    p.add_argument("--out", type=Path, help="write the JSON report (and a .txt summary)")
    p.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="token partitions (parquet): reused when present, else read and written",
    )
    p.add_argument("--parts", type=int, default=8, help="token partitions (bounds the memory)")
    return p  # fmt: skip


def _utc(value: str) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime()


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.samples < 1:
        raise SystemExit(f"--samples must be >= 1, got {args.samples}")
    if args.parts < 1:
        raise SystemExit(f"--parts must be >= 1, got {args.parts}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    files = sorted(args.cache_dir.glob("part-*.parquet")) if args.cache_dir.exists() else []
    if not files:
        from sqlalchemy import text

        from .db import make_engine

        engine = make_engine()
        with engine.connect() as conn:
            lo, hi = conn.execute(text("SELECT min(ts), max(ts) FROM features")).one()
        start = args.start or lo
        end = args.end or hi + timedelta(seconds=1)
        files = read_partitions(engine, start, end, min(AGES.values()), args.cache_dir, args.parts)
    report = study_partitions(files, load_config(), data_end_of(files), samples=args.samples)
    text_report = render(report)
    print(text_report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        args.out.with_suffix(".txt").write_text(text_report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
