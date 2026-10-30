"""Build labeled training sequences from the pipeline's ``features`` rows.

The Transformer consumes *sequences*, so we slide a ``seq_len`` window over each token's
time-ordered features. Labels come only from the realized future trajectory (objective — no
policy, no paper trades needed). For a window ending at ``t`` with anchor price ``p`` and anchor
liquidity ``l``, over each horizon ``(t, t + H]``:

* ``pump`` — the *highest* price in the horizon reached ``p * (1 + pump_threshold)``;
* ``rug`` — the *lowest* price fell to ``p * (1 - rug_threshold)`` **or** the lowest liquidity
  fell to ``l * (1 - rug_threshold)`` (a liquidity pull is a rug even if the last price barely
  moves);
* ``expected_return`` — the price at the end of the horizon over ``p``, minus 1, capped at
  ``max_return``: a price read off a pool drained to dust (a rug's aftermath) can be 10^8 times the
  anchor, and one such label would dominate the regression loss. The floor is -1 (prices are > 0).

``pump`` and ``rug`` are independent heads: a violent swing can be both, and that is kept.

A horizon is labeled (``mask = 1``) when the data covers it: rows reach ``t + H`` and the last
row inside the horizon is recent enough (within ``max(max_gap_fraction * H, min_gap_seconds)``
of its end — the floor keeps the normal snapshot spacing from counting as a gap), so a gap in
the data never passes a stale price off as the realized outcome. Otherwise it is masked — except
when a rug has *already happened* inside the observed part of the horizon: that event is certain
whatever comes next, so it is labeled. This matters because a token that collapses stops trading
and the pipeline stops snapshotting it; without this rule the windows leading into a rug would
never be labeled. In that case ``pump`` and ``expected_return`` come from the observed part (the
token is treated as dead).

The split is temporal with an embargo so a training window's look-ahead can't leak into
validation. Windows are not copied: samples keep each token's feature rows once and slice
windows on access, so memory stays proportional to the data, not ``seq_len`` times it.

Only windows whose anchor (last) row is *eligible* (:mod:`solia_ai.eligibility`, ``config.yaml``
``eligibility``) become samples — the population the bot trades. The rule is causal, so it never
peeks at the future; labels are unchanged, and the rows before the token became eligible may still
sit in a window's history.

Prices read off near-empty pools are artifacts (a micro-trade moves them by multiples). Under a
liquidity floor (:mod:`solia_ai.liquidity`, ``config.yaml`` ``labeling.liquidity_floor``,
venue-aware) a window anchored on a pool below its floor is not a sample, and a price in the horizon
read off one counts neither towards ``pump`` nor ``expected_return`` (which falls back to the last
counted price, the anchor's at worst). ``rug`` is unchanged: a liquidity collapse is still a rug.

Consecutive snapshots of a token are ~10 s apart and nearly identical, so a window per snapshot
shows the model the same situation many times (it memorizes). ``window_stride_seconds`` keeps at
most **one window per stride of a token's life**: the token's time since its first row is cut into
``stride``-long buckets and the first eligible window ending in each is kept. The thinning picks
window *ends* before labeling, from the token's own clock — deterministic, causal, the same rule for
train / validation / held-out (all built here) — and changes no window's history or labels.

Young tokens matter most, so windows may be shorter than ``seq_len`` (down to ``min_seq_len``
rows): a window holds a token's most recent ``<= seq_len`` rows, and :func:`pad_window` left-pads
it to ``seq_len`` with a padding mask the model uses to ignore the padding.
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from .config import DATA_DEFAULTS, MAX_POS_WEIGHT, horizon_seconds, load_config, window_stride_error
from .eligibility import INPUT_COLUMNS as ELIGIBILITY_INPUTS
from .eligibility import EligibilityRule
from .features import (
    BASELINE_GROUPS,
    FEATURE_COLUMNS,
    build_feature_matrix,
    check_group_coverage,
)
from .liquidity import LiquidityFloor

DEFAULT_MAX_GAP_FRACTION = 0.25
DEFAULT_MIN_GAP_SECONDS = 20  # two pipeline snapshot cadences: normal spacing is never a "gap"
DEFAULT_MAX_RETURN = 10.0  # +1000 %: above every pump threshold, far below dust-pool artifacts


@dataclass(frozen=True)
class Samples:
    """Labeled windows over shared feature rows. First axis of every per-sample array is the
    sample index; :meth:`window` slices a sample's ``[<= seq_len, F]`` input on demand."""

    features: np.ndarray  # [R, F] float32 — every token's rows, tokens contiguous
    start_row: np.ndarray  # [N] int64 — row index (into features) of each window's first row
    end_row: np.ndarray  # [N] int64 — row index (into features) of each window's last row
    seq_len: int
    pump: np.ndarray  # [N, H] float32 (0/1)
    rug: np.ndarray  # [N, H] float32 (0/1)
    expected_return: np.ndarray  # [N, H] float32
    mask: np.ndarray  # [N, H] float32 (1 = labeled)
    end_ts: np.ndarray  # [N] int64 epoch seconds of the window's last row
    mint: np.ndarray  # [N] object
    horizons: tuple[str, ...]
    # [N, 3] float32 — the anchor row's eligibility inputs (solia_ai.eligibility.running_inputs),
    # so another rule can be applied without rebuilding (:meth:`EligibilityRule.accepts`).
    eligibility_inputs: np.ndarray | None = None
    # [N] float32 / object — the anchor row's pool liquidity (SOL, NaN = unknown) and venue, so the
    # windows can be split by liquidity (solia_ai.liquidity.liquidity_buckets) without a rebuild.
    anchor_liquidity: np.ndarray | None = None
    anchor_venue: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.end_row)

    def window(self, i: int) -> np.ndarray:
        """The unpadded ``[L, F]`` input of sample ``i``, ``L <= seq_len`` (a view, no copy)."""
        return self.features[int(self.start_row[i]) : int(self.end_row[i]) + 1]

    def padded(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``i`` left-padded to ``[seq_len, F]``, with its ``[seq_len]`` padding mask."""
        return pad_window(self.window(i), self.seq_len)

    def padded_batch(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """:meth:`padded` of every sample in ``indices`` at once: ``(x[B, seq_len, F],
        padding_mask[B, seq_len])``, gathered in one indexing pass instead of one per sample."""
        indices = np.asarray(indices, np.int64)
        # Position j of a window holds row end - (seq_len - 1 - j); rows before its start are pad.
        rows = self.end_row[indices, None] - np.arange(self.seq_len - 1, -1, -1)
        padding_mask = rows < self.start_row[indices, None]
        x = self.features[np.where(padding_mask, 0, rows)]
        x[padding_mask] = 0.0
        return x, padding_mask

    @property
    def x(self) -> np.ndarray:
        """All windows, left-padded, as one dense ``[N, seq_len, F]`` array. Materializes
        ``seq_len`` copies of every row — for small sets and inspection only; training goes through
        :meth:`padded`."""
        if len(self) == 0:
            return np.empty((0, self.seq_len, self.features.shape[1]), np.float32)
        return np.stack([self.padded(i)[0] for i in range(len(self))])


def pad_window(window: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad (or keep the most recent ``seq_len`` rows of) a ``[L, F]`` window.

    Returns ``(x[seq_len, F] float32, padding_mask[seq_len] bool)`` where ``True`` marks padding.
    The most recent row is always last, so positions are relative to "now" whatever ``L`` is.
    """
    if window.ndim != 2 or len(window) == 0:
        raise ValueError(f"expected a non-empty [L, F] window, got shape {window.shape}")
    rows = window[-seq_len:]
    pad = seq_len - len(rows)
    x = np.zeros((seq_len, window.shape[1]), np.float32)
    x[pad:] = rows
    padding_mask = np.zeros(seq_len, bool)
    padding_mask[:pad] = True
    return x, padding_mask


def build_samples(
    df: pd.DataFrame,
    *,
    seq_len: int,
    horizons: Sequence[str],
    pump_thresholds: Mapping[str, float],
    rug_thresholds: Mapping[str, float],
    max_gap_fraction: float = DEFAULT_MAX_GAP_FRACTION,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    min_seq_len: int | None = None,
    max_return: float = DEFAULT_MAX_RETURN,
    eligibility: EligibilityRule | None = None,
    window_stride_seconds: float = 0,
    liquidity_floor: LiquidityFloor | None = None,
) -> Samples:
    """Slide windows per mint and label each by the realized future trajectory (module doc).

    A window ends at every row from the token's ``min_seq_len``-th on (default ``seq_len``: full
    windows only) that ``eligibility`` accepts (default: every row) and whose pool is at or above
    ``liquidity_floor`` (default: disabled; :mod:`solia_ai.liquidity`) — at most one per
    ``window_stride_seconds`` of the token's life (default 0: every row) — and holds its most
    recent ``<= seq_len`` rows. Built in vectorized passes over every token at once (one sort, one
    feature projection, the labels by searches and range extrema), never a loop per token or
    window.
    """
    rule = EligibilityRule() if eligibility is None else eligibility
    rule.check_columns(df)
    floor = LiquidityFloor() if liquidity_floor is None else liquidity_floor
    if floor.enabled and "liquidity" not in df.columns:
        raise ValueError("the liquidity floor needs the 'liquidity' column")
    horizons = tuple(horizons)
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    min_len = seq_len if min_seq_len is None else min_seq_len
    if not 1 <= min_len <= seq_len:
        raise ValueError(f"min_seq_len must be in [1, seq_len ({seq_len})], got {min_len}")
    if not horizons:
        raise ValueError("at least one horizon is required")
    if not 0 < max_gap_fraction <= 1:
        raise ValueError(f"max_gap_fraction must be in (0, 1], got {max_gap_fraction}")
    if min_gap_seconds < 0:
        raise ValueError(f"min_gap_seconds must be >= 0, got {min_gap_seconds}")
    if not max_return > 0:
        raise ValueError(f"max_return must be > 0, got {max_return}")
    if (problem := window_stride_error(window_stride_seconds)) is not None:
        raise ValueError(problem)
    for name, thresholds in (("pump", pump_thresholds), ("rug", rug_thresholds)):
        missing = [h for h in horizons if h not in thresholds]
        if missing:
            raise ValueError(f"no {name} threshold for horizon(s): {', '.join(missing)}")
    if "mint" not in df.columns or "ts" not in df.columns:
        raise ValueError("feature frame needs 'mint' and 'ts' columns")

    horizon_s = np.array([horizon_seconds(h) for h in horizons], np.int64)
    pump_thr = np.array([pump_thresholds[h] for h in horizons], np.float64)
    rug_thr = np.array([rug_thresholds[h] for h in horizons], np.float64)
    max_gap = np.maximum(horizon_s * max_gap_fraction, min_gap_seconds)

    # The frame is never copied (a copy costs as much as the read): rows are picked and ordered
    # through an index, and each column is gathered once, already ordered.
    codes, _ = pd.factorize(df["mint"], sort=True)  # mint order; -1 = no mint
    sizes = np.bincount(codes[codes >= 0], minlength=1)
    # groupby semantics: rows without a mint belong to no token; tokens too short hold no window
    rows = np.flatnonzero((codes >= 0) & (sizes[np.maximum(codes, 0)] >= min_len))
    n_feat = len(FEATURE_COLUMNS)
    if len(rows) == 0:
        return _empty_samples(seq_len, horizons, n_feat)
    # Tokens in mint order, each token's rows in time order; ties keep a deterministic order (by
    # id when present, else input order: lexsort is stable).
    ts = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).to_numpy()
    keys = [ts[rows].astype("datetime64[ns]").astype(np.int64), codes[rows]]
    if "id" in df.columns:
        keys.insert(0, df["id"].to_numpy()[rows])
    order = rows[np.lexsort(keys)]

    token_codes = codes[order]
    token = np.r_[0, np.cumsum(token_codes[1:] != token_codes[:-1])].astype(np.int64)
    first = np.flatnonzero(np.r_[True, token[1:] != token[:-1]])
    token_start = first[token]  # each row's token's first row
    token_last = np.r_[first[1:], len(token)][token] - 1  # ... and last row
    pos = np.arange(len(token)) - token_start  # row position inside its token

    price = _numeric(df, "price", order)
    liquidity = _numeric(df, "liquidity", order)
    venue = df["venue"].iloc[order].to_numpy(dtype=object) if "venue" in df.columns else None
    ts_sec = ts[order].astype("datetime64[s]").astype(np.int64)
    # Rows whose price is a tradeable outcome (solia_ai.liquidity); None = every row (no floor).
    tradeable = floor.tradeable(liquidity, venue) if floor.enabled else None

    window_ends = np.flatnonzero(pos >= min_len - 1)
    anchor_inputs = _running_inputs(df, order, token)[window_ends]
    eligible = rule.accepts(anchor_inputs)  # anchored on an eligible row
    if tradeable is not None:
        eligible &= tradeable[window_ends]  # ... of a pool at or above the liquidity floor
    window_ends, anchor_inputs = window_ends[eligible], anchor_inputs[eligible]
    if window_stride_seconds > 0:  # thinned before labeling: only kept windows are labeled
        age = ts_sec - ts_sec[token_start]
        strided = stride_ends(window_ends, token, age, window_stride_seconds)
        window_ends, anchor_inputs = window_ends[strided], anchor_inputs[strided]
    pump, rug, ret, mask = _label(
        window_ends, token, token_last, ts_sec, price, liquidity,
        horizon_s, pump_thr, rug_thr, max_gap, max_return, tradeable,
    )  # fmt: skip
    keep = mask.any(axis=1)  # drop windows with nothing labelable yet
    if not keep.any():
        return _empty_samples(seq_len, horizons, n_feat)
    ends = window_ends[keep]

    # Only the rows of tokens holding a sample are stored; renumber rows over them.
    has_sample = np.zeros(int(token[-1]) + 1, bool)
    has_sample[token[ends]] = True
    stored = has_sample[token]
    new_row = np.cumsum(stored) - 1
    end_row = new_row[ends]
    start_row = end_row - np.minimum(pos[ends], seq_len - 1)

    return Samples(
        build_feature_matrix(df, rows=order[stored]),  # [R, F] float32
        start_row.astype(np.int64),
        end_row.astype(np.int64),
        seq_len,
        pump[keep],
        rug[keep],
        ret[keep],
        mask[keep],
        ts_sec[ends].astype(np.int64),
        df["mint"].to_numpy(dtype=object)[order[ends]],
        horizons,
        anchor_inputs[keep].astype(np.float32),
        liquidity[ends].astype(np.float32),
        np.full(len(ends), None, object) if venue is None else venue[ends],
    )


def _numeric(df: pd.DataFrame, col: str, rows: np.ndarray) -> np.ndarray:
    """``df[col]`` at ``rows`` as float64 (non-numeric values -> NaN)."""
    return pd.to_numeric(df[col].iloc[rows], errors="coerce").to_numpy(dtype=np.float64)


def stride_ends(
    ends: np.ndarray, token: np.ndarray, age_sec: np.ndarray, stride: float
) -> np.ndarray:
    """Which of the candidate window ``ends`` (row indices, ascending: tokens contiguous, time
    ordered) to keep: the first one in each ``stride``-second bucket of its token's age
    (``age_sec``: seconds since the token's first row). A boolean mask over ``ends``."""
    if len(ends) == 0:
        return np.zeros(0, bool)
    bucket = np.floor(age_sec[ends] / stride).astype(np.int64)
    tok = token[ends]
    return np.r_[True, (tok[1:] != tok[:-1]) | (bucket[1:] != bucket[:-1])]


def _empty_samples(seq_len: int, horizons: tuple[str, ...], n_feat: int) -> Samples:
    empty_h = np.empty((0, len(horizons)), np.float32)
    return Samples(
        np.empty((0, n_feat), np.float32),
        np.empty((0,), np.int64),
        np.empty((0,), np.int64),
        seq_len,
        empty_h,
        empty_h.copy(),
        empty_h.copy(),
        empty_h.copy(),
        np.empty((0,), np.int64),
        np.empty((0,), object),
        horizons,
        np.empty((0, len(ELIGIBILITY_INPUTS)), np.float32),
        np.empty((0,), np.float32),
        np.empty((0,), object),
    )


def _running_inputs(df: pd.DataFrame, rows: np.ndarray, token: np.ndarray) -> np.ndarray:
    """:func:`solia_ai.eligibility.frame_inputs` of every token at once: ``[R, 3]`` running maxima
    of the eligibility inputs per token over ``df``'s ``rows`` (in that order; NaN until a token's
    first known value)."""
    columns = []
    for col in ELIGIBILITY_INPUTS:
        if col not in df:
            columns.append(np.full(len(rows), np.nan))
            continue
        values = pd.Series(_numeric(df, col, rows))
        # cummax skips NaN but leaves it in place; carrying the max over it = np.fmax.accumulate.
        running = values.groupby(token, sort=False).cummax().groupby(token, sort=False).ffill()
        columns.append(running.to_numpy(np.float64))
    return np.stack(columns, axis=1)


def _label(
    window_ends: np.ndarray,
    token: np.ndarray,
    token_last: np.ndarray,
    ts_sec: np.ndarray,
    price: np.ndarray,
    liquidity: np.ndarray,
    horizon_s: np.ndarray,
    pump_thr: np.ndarray,
    rug_thr: np.ndarray,
    max_gap: np.ndarray,
    max_return: float,
    tradeable: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Label every window (module doc) at once, over every token's rows (time-ordered, tokens
    contiguous). Each horizon looks at the rows ``(t, t + H]`` of the window's own token: a
    search over ``(token, ts)`` keys, then range minima / maxima over those rows.

    ``tradeable`` (``[R]`` bool, None = every row) marks the rows at or above the liquidity floor:
    only their prices count towards ``pump`` and ``expected_return``, and only they anchor."""
    n, n_h = len(window_ends), len(horizon_s)
    pump = np.zeros((n, n_h), np.float32)
    rug = np.zeros((n, n_h), np.float32)
    ret = np.zeros((n, n_h), np.float32)
    mask = np.zeros((n, n_h), np.float32)
    if n == 0:
        return pump, rug, ret, mask

    # (token, ts) as one sortable int64 key: a horizon's search never leaves its token.
    rel = ts_sec - ts_sec.min()
    if int(rel.max()) + int(horizon_s.max()) >= 2**32:
        raise ValueError("the features span too long a time range to label")
    key = (token << 32) | rel

    i = window_ends
    p_i, l_i, t_i = price[i], liquidity[i], ts_sec[i]
    anchored = np.isfinite(p_i) & (p_i > 0)  # no usable anchor price: unlabeled
    if tradeable is not None:
        anchored &= tradeable[i]
    lo = np.searchsorted(key, key[i], side="right")  # first row strictly after t_i
    last_sec = ts_sec[token_last[i]]

    price_finite = np.isfinite(price)
    n_price = np.r_[0, np.cumsum(price_finite)]  # finite values in [lo, hi) = n[hi] - n[lo]
    n_liq = np.r_[0, np.cumsum(np.isfinite(liquidity))]
    # Prices that count towards pump / return: finite, read off a pool at or above the floor.
    counted = price_finite if tradeable is None else price_finite & tradeable
    # Index of the last counted price at or before each row (-1: none).
    last_finite = np.maximum.accumulate(np.where(counted, np.arange(len(price)), -1))
    price_lo = np.where(np.isnan(price), np.inf, price)  # nanmin (rug: dust prices included)
    price_hi = np.where(counted, price, -np.inf)  # max over the counted values
    liq_lo = np.where(np.isnan(liquidity), np.inf, liquidity)

    for h in range(n_h):
        horizon_end = t_i + horizon_s[h]
        hi = np.searchsorted(key, key[i] + horizon_s[h], side="right")  # rows (t_i, t_i + H]
        future = anchored & (hi > lo)
        future &= n_price[hi] - n_price[np.minimum(lo, hi)] > 0  # some finite future price
        lo_f, hi_f = np.where(future, lo, 0), np.where(future, hi, 1)
        p_min, p_max, l_min = _range_extrema(
            lo_f, hi_f, (price_lo, np.minimum), (price_hi, np.maximum), (liq_lo, np.minimum)
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            crashed = (p_i - p_min) / p_i >= rug_thr[h]
            liq_known = np.isfinite(l_i) & (l_i > 0) & (n_liq[hi_f] - n_liq[lo_f] > 0)
            crashed |= liq_known & ((l_i - l_min) / l_i >= rug_thr[h])
            covered = (last_sec >= horizon_end) & (ts_sec[hi_f - 1] >= horizon_end - max_gap[h])
            labeled = future & (covered | crashed)
            pump[:, h] = np.where(labeled, p_max / p_i - 1.0 >= pump_thr[h], 0.0)
            rug[:, h] = np.where(labeled, crashed, 0.0)
            # Counted price at (or nearest before) the horizon end, capped (module doc). The
            # anchor counts, so it is the fallback when the horizon only read dust pools.
            last_price = price[last_finite[hi_f - 1]]
            ret[:, h] = np.where(labeled, np.minimum(last_price / p_i - 1.0, max_return), 0.0)
        mask[:, h] = labeled

    return pump, rug, ret, mask


def _range_extrema(
    lo: np.ndarray, hi: np.ndarray, *series: tuple[np.ndarray, Any]
) -> list[np.ndarray]:
    """``reduce(values[lo:hi])`` for every ``(values, reduce)`` in ``series`` and every range
    (``hi > lo``), vectorized: a sparse table built one level at a time, so only one level per
    series is held at once. A range of length ``L`` is answered at level ``k = floor(log2 L)``
    by its two overlapping ``2**k``-long halves."""
    level = np.floor(np.log2(np.maximum(hi - lo, 1))).astype(np.int64)
    out = [np.empty(len(lo), np.float64) for _ in series]
    tables = [values for values, _ in series]
    for k in range(int(level.max()) + 1):
        if k:
            step = 1 << (k - 1)
            tables = [
                reduce(table[:-step], table[step:])
                for table, (_, reduce) in zip(tables, series, strict=True)
            ]
        at = np.flatnonzero(level == k)
        if len(at) == 0:
            continue
        left, right = lo[at], hi[at] - (1 << k)
        for o, table, (_, reduce) in zip(out, tables, series, strict=True):
            o[at] = reduce(table[left], table[right])
    return out


def temporal_split(
    end_ts: np.ndarray, *, val_fraction: float = 0.2, embargo_seconds: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices by time. Validation is strictly later than the cutoff; an embargo gap
    drops training samples whose label look-ahead could cross into validation (no leakage)."""
    if not 0 < val_fraction < 1:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if embargo_seconds < 0:
        raise ValueError(f"embargo_seconds must be >= 0, got {embargo_seconds}")
    if len(end_ts) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    cutoff = np.quantile(end_ts.astype(np.float64), 1.0 - val_fraction)
    train_idx = np.where(end_ts <= cutoff - embargo_seconds)[0]
    val_idx = np.where(end_ts > cutoff)[0]
    return train_idx, val_idx


class SequenceDataset(Dataset):
    """Torch view over :class:`Samples`. Yields ``(x[seq_len, F], padding_mask[seq_len], targets)``:
    ``x`` is left-padded, ``padding_mask`` is ``True`` on padding, and targets holds ``pump`` /
    ``rug`` / ``expected_return`` / ``mask`` as ``[H]`` float32 tensors."""

    def __init__(self, samples: Samples, indices: np.ndarray | None = None) -> None:
        self.samples = samples
        self.indices = np.arange(len(samples)) if indices is None else np.asarray(indices, np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        i = int(self.indices[k])
        s = self.samples
        targets = {
            "pump": torch.from_numpy(s.pump[i]),
            "rug": torch.from_numpy(s.rug[i]),
            "expected_return": torch.from_numpy(s.expected_return[i]),
            "mask": torch.from_numpy(s.mask[i]),
        }
        x, padding_mask = s.padded(i)
        return torch.from_numpy(x), torch.from_numpy(padding_mask), targets

    def batch(
        self, ks: Sequence[int]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Items ``ks`` already collated — exactly what the default collation of
        ``[self[k] for k in ks]`` gives (``[B, ...]`` tensors), in a few vectorized gathers."""
        i = self.indices[np.asarray(ks, np.int64)]
        s = self.samples
        x, padding_mask = s.padded_batch(i)
        targets = {
            name: torch.from_numpy(getattr(s, name)[i])
            for name in ("pump", "rug", "expected_return", "mask")
        }
        return torch.from_numpy(x), torch.from_numpy(padding_mask), targets


class _Batches(Dataset):
    """A loader's view of a :class:`SequenceDataset` indexed by whole batches (lists of item
    indices from a :class:`~torch.utils.data.BatchSampler`). Only small index lists and finished
    batches cross to a worker: the rows are read in place (shared copy-on-write under fork) and no
    per-sample Python object is created or ref-counted."""

    def __init__(self, dataset: SequenceDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, ks: Sequence[int]):
        return self.dataset.batch(ks)


# --------------------------------------------------------------------------- class imbalance


def positive_weights(dataset: SequenceDataset) -> dict[str, torch.Tensor]:
    """Per-horizon ``pos_weight`` (negatives / positives over labeled samples) for
    ``BCEWithLogitsLoss``, for ``pump`` and ``rug``. Pumps and rugs are rare, so this up-weights
    them. 1.0 when a horizon has no positives (or no labels); capped at :data:`MAX_POS_WEIGHT`."""
    s, idx = dataset.samples, dataset.indices
    mask = s.mask[idx].astype(bool)
    weights = {}
    for name in ("pump", "rug"):
        labels = getattr(s, name)[idx].astype(bool)
        pos = (labels & mask).sum(axis=0).astype(np.float64)
        neg = (~labels & mask).sum(axis=0).astype(np.float64)
        w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
        weights[name] = torch.tensor(np.clip(w, 1.0, MAX_POS_WEIGHT), dtype=torch.float32)
    return weights


def balanced_sampler(dataset: SequenceDataset, *, seed: int = 0) -> WeightedRandomSampler:
    """Sampler that draws event windows (any labeled pump or rug) and quiet windows equally
    often, so batches aren't almost all negatives. Reproducible for a given ``seed``."""
    s, idx = dataset.samples, dataset.indices
    mask = s.mask[idx].astype(bool)
    event = ((s.pump[idx].astype(bool) | s.rug[idx].astype(bool)) & mask).any(axis=1)
    n, n_event = len(idx), int(event.sum())
    if n == 0:
        raise ValueError("cannot build a sampler over an empty dataset")
    if n_event in (0, n):
        weights = np.ones(n)  # a single class: plain uniform sampling
    else:
        weights = np.where(event, n / (2 * n_event), n / (2 * (n - n_event)))
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), n, replacement=True, generator=generator
    )


MAX_AUTO_WORKERS = 4


def auto_num_workers() -> int:
    """``training.num_workers: auto``: half the usable CPUs (at most :data:`MAX_AUTO_WORKERS`)
    where workers fork — and so share the dataset's rows copy-on-write — else ``0``: a spawned
    worker (Windows, macOS) would receive its own copy of every row."""
    if multiprocessing.get_start_method() != "fork":
        return 0
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return min(MAX_AUTO_WORKERS, max(1, (cpus or 1) // 2))


@dataclass(frozen=True)
class LoaderSettings:
    """How batches are fed to the model (``config.yaml`` ``training``): ``num_workers`` processes
    gather them (``0`` = the training process), ``pin_memory`` for faster host-to-GPU copies,
    ``prefetch_factor`` batches queued per worker. Workers persist across epochs. Batches don't
    depend on these: the sampling happens in the training process."""

    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise ValueError(f"num_workers must be an integer, got {self.num_workers!r}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if isinstance(self.prefetch_factor, bool) or not isinstance(self.prefetch_factor, int):
            raise ValueError(f"prefetch_factor must be an integer, got {self.prefetch_factor!r}")
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], device: str | torch.device | None = None
    ) -> LoaderSettings:
        """The settings for training on ``device``: ``num_workers: auto`` is
        :func:`auto_num_workers`, ``pin_memory: auto`` is on for CUDA."""
        training = config["training"]
        workers = training.get("num_workers", 0)
        pin = training.get("pin_memory", "auto")
        if pin == "auto":
            pin = device is not None and torch.device(device).type == "cuda"
        if not isinstance(pin, bool):
            raise ValueError(f"pin_memory must be a boolean or 'auto', got {pin!r}")
        return cls(
            num_workers=auto_num_workers() if workers == "auto" else workers,
            pin_memory=pin,
            prefetch_factor=training.get("prefetch_factor", 2),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def batch_loader(
    dataset: SequenceDataset,
    sampler: Sampler[int],
    *,
    batch_size: int,
    seed: int = 0,
    loader: LoaderSettings | None = None,
) -> DataLoader:
    """A loader yielding :meth:`SequenceDataset.batch` batches of ``batch_size`` items drawn by
    ``sampler``.

    The loader gets its own seeded generator: the base seed it draws for workers at each iteration
    never touches the global RNG, so dropout — hence the trained weights — is identical whatever
    ``num_workers`` (with persistent workers that seed is drawn once, not once per epoch).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    settings = loader or LoaderSettings()
    workers = settings.num_workers
    return DataLoader(
        _Batches(dataset),
        sampler=BatchSampler(sampler, batch_size, drop_last=False),
        batch_size=None,  # the sampler yields whole batches, collated by SequenceDataset.batch
        num_workers=workers,
        pin_memory=settings.pin_memory,
        persistent_workers=workers > 0,
        prefetch_factor=settings.prefetch_factor if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


def make_loaders(
    train: SequenceDataset,
    val: SequenceDataset,
    *,
    batch_size: int,
    balanced: bool = True,
    seed: int = 0,
    loader: LoaderSettings | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (class-balanced sampling by default) and an ordered, unshuffled val loader,
    both fed per ``loader`` (default: in the training process)."""
    if balanced:
        sampler: Sampler[int] = balanced_sampler(train, seed=seed)
    else:
        sampler = RandomSampler(train, generator=torch.Generator().manual_seed(seed))
    return (
        batch_loader(train, sampler, batch_size=batch_size, seed=seed, loader=loader),
        batch_loader(val, SequentialSampler(val), batch_size=batch_size, seed=seed, loader=loader),
    )


def make_datasets(
    df: pd.DataFrame, config: Mapping[str, Any] | None = None, *, val_fraction: float = 0.2
) -> tuple[SequenceDataset, SequenceDataset]:
    """Build train/val datasets from a `features` frame using config knobs.

    The embargo is the longest horizon. Raises if either split ends up empty (e.g. a data span
    shorter than the embargo), rather than failing later in training.
    """
    cfg = load_config() if config is None else config
    samples = config_samples(df, cfg)
    embargo = embargo_seconds(samples.horizons)
    train_idx, val_idx = temporal_split(
        samples.end_ts, val_fraction=val_fraction, embargo_seconds=embargo
    )
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError(
            f"empty split ({len(samples)} labeled windows -> train={len(train_idx)}, "
            f"val={len(val_idx)}); the data must span well beyond the {embargo}s embargo "
            "(longest horizon) — use a longer time range"
        )
    return SequenceDataset(samples, train_idx), SequenceDataset(samples, val_idx)


def embargo_seconds(horizons: Sequence[str]) -> int:
    """The train/validation gap: the longest horizon's label look-ahead."""
    return max(horizon_seconds(h) for h in horizons)


def config_samples(df: pd.DataFrame, cfg: Mapping[str, Any]) -> Samples:
    """The windows of ``df`` per ``cfg`` (anchored on rows its ``eligibility`` rule accepts,
    labeled under its ``labeling.liquidity_floor``);
    refuses rows predating an enabled feature group."""
    check_group_coverage(df, cfg.get("features", {}).get("groups", BASELINE_GROUPS))
    labeling = cfg["labeling"]
    return build_samples(
        df,
        seq_len=cfg["model"]["seq_len"],
        min_seq_len=cfg["model"].get("min_seq_len"),
        horizons=tuple(cfg["horizons"]),
        pump_thresholds=labeling["pump_threshold"],
        rug_thresholds=labeling["rug_threshold"],
        max_gap_fraction=labeling.get("max_gap_fraction", DEFAULT_MAX_GAP_FRACTION),
        min_gap_seconds=labeling.get("min_gap_seconds", DEFAULT_MIN_GAP_SECONDS),
        max_return=labeling.get("max_return", DEFAULT_MAX_RETURN),
        eligibility=EligibilityRule.from_config(cfg),
        window_stride_seconds=window_stride(cfg),
        liquidity_floor=LiquidityFloor.from_config(cfg),
    )


def window_stride(cfg: Mapping[str, Any]) -> float:
    """``config.yaml`` ``data.window_stride_seconds`` (0 = a window per snapshot)."""
    return cfg.get("data", {}).get("window_stride_seconds", DATA_DEFAULTS["window_stride_seconds"])


def make_holdout_dataset(
    df: pd.DataFrame, config: Mapping[str, Any] | None = None, *, train_end_ts: int
) -> SequenceDataset:
    """Every window of ``df`` that ends after a model's training data, embargo included.

    ``train_end_ts`` is the checkpoint's last training window end; keeping only windows ending
    strictly after ``train_end_ts + embargo`` is the same rule :func:`temporal_split` applies, so
    no window the model was fit on (nor one whose label overlaps its training labels) is scored.
    Raises when nothing is left — evaluating on an empty window would report nothing.
    """
    cfg = load_config() if config is None else config
    samples = config_samples(df, cfg)
    after = int(train_end_ts) + embargo_seconds(samples.horizons)
    indices = np.where(samples.end_ts > after)[0]
    if len(indices) == 0:
        raise ValueError(
            f"no window ends after the model's training data (train_end_ts={train_end_ts}, "
            f"+{after - int(train_end_ts)}s embargo); read a later time range to evaluate"
        )
    return SequenceDataset(samples, indices)
