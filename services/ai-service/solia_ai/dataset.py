"""Build labeled training sequences from the pipeline's ``features`` rows.

The Transformer consumes *sequences*, so we slide a ``seq_len`` window over each token's
time-ordered features. Labels come only from the realized future trajectory (objective — no
policy, no paper trades needed). For a window ending at ``t`` with anchor price ``p`` and anchor
liquidity ``l``, over each horizon ``(t, t + H]``:

* ``pump`` — the *highest* price in the horizon reached ``p * (1 + pump_threshold)``;
* ``rug`` — the *lowest* price fell to ``p * (1 - rug_threshold)`` **or** the lowest liquidity
  fell to ``l * (1 - rug_threshold)`` (a liquidity pull is a rug even if the last price barely
  moves);
* ``expected_return`` — the price at the end of the horizon over ``p``, minus 1.

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

Young tokens matter most, so windows may be shorter than ``seq_len`` (down to ``min_seq_len``
rows): a window holds a token's most recent ``<= seq_len`` rows, and :func:`pad_window` left-pads
it to ``seq_len`` with a padding mask the model uses to ignore the padding.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import MAX_POS_WEIGHT, load_config
from .features import (
    BASELINE_GROUPS,
    FEATURE_COLUMNS,
    build_feature_matrix,
    check_group_coverage,
)

_HORIZON_UNITS = {"s": 1, "m": 60, "h": 3600}

DEFAULT_MAX_GAP_FRACTION = 0.25
DEFAULT_MIN_GAP_SECONDS = 20  # two pipeline snapshot cadences: normal spacing is never a "gap"


def horizon_seconds(label: str) -> int:
    """Parse a horizon like ``'30s'``, ``'5m'``, ``'1h'`` into seconds."""
    text = label.strip().lower()
    if len(text) < 2 or text[-1] not in _HORIZON_UNITS or not text[:-1].isdigit():
        raise ValueError(f"invalid horizon {label!r} (expected e.g. '30s', '5m', '1h')")
    seconds = int(text[:-1]) * _HORIZON_UNITS[text[-1]]
    if seconds <= 0:
        raise ValueError(f"invalid horizon {label!r} (must be positive)")
    return seconds


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

    def __len__(self) -> int:
        return len(self.end_row)

    def window(self, i: int) -> np.ndarray:
        """The unpadded ``[L, F]`` input of sample ``i``, ``L <= seq_len`` (a view, no copy)."""
        return self.features[int(self.start_row[i]) : int(self.end_row[i]) + 1]

    def padded(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample ``i`` left-padded to ``[seq_len, F]``, with its ``[seq_len]`` padding mask."""
        return pad_window(self.window(i), self.seq_len)

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
) -> Samples:
    """Slide windows per mint and label each by the realized future trajectory (module doc).

    A window ends at every row from the token's ``min_seq_len``-th on (default ``seq_len``: full
    windows only) and holds its most recent ``<= seq_len`` rows.
    """
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

    blocks: list[np.ndarray] = []
    start_rows: list[np.ndarray] = []
    end_rows: list[np.ndarray] = []
    labels: list[tuple[np.ndarray, ...]] = []
    end_ts: list[np.ndarray] = []
    mints: list[np.ndarray] = []
    offset = 0

    for mint, sub in df.groupby("mint", sort=True):
        if len(sub) < min_len:
            continue
        # Stable sort (ties keep a deterministic order: by id when present, else input order).
        sort_keys = ["ts", "id"] if "id" in sub.columns else ["ts"]
        sub = sub.sort_values(sort_keys, kind="stable")

        matrix = build_feature_matrix(sub)  # [T, F] float32
        price = pd.to_numeric(sub["price"], errors="coerce").to_numpy(dtype=np.float64)
        liquidity = pd.to_numeric(sub["liquidity"], errors="coerce").to_numpy(dtype=np.float64)
        ts_utc = pd.to_datetime(sub["ts"], utc=True).dt.tz_localize(None)
        ts_sec = ts_utc.to_numpy().astype("datetime64[s]").astype(np.int64)

        window_ends = np.arange(min_len - 1, len(sub))
        pump, rug, ret, mask = _label(
            window_ends, ts_sec, price, liquidity, horizon_s, pump_thr, rug_thr, max_gap
        )
        keep = mask.any(axis=1)  # drop windows with nothing labelable yet
        if not keep.any():
            continue

        blocks.append(matrix)
        start_rows.append(np.maximum(window_ends[keep] - seq_len + 1, 0) + offset)
        end_rows.append(window_ends[keep] + offset)
        labels.append((pump[keep], rug[keep], ret[keep], mask[keep]))
        end_ts.append(ts_sec[window_ends[keep]])
        mints.append(np.full(int(keep.sum()), mint, dtype=object))
        offset += len(matrix)

    n_feat = len(FEATURE_COLUMNS)
    n_h = len(horizons)
    if not blocks:
        empty_h = np.empty((0, n_h), np.float32)
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
        )

    return Samples(
        np.concatenate(blocks),
        np.concatenate(start_rows).astype(np.int64),
        np.concatenate(end_rows).astype(np.int64),
        seq_len,
        *(np.concatenate([lab[k] for lab in labels]).astype(np.float32) for k in range(4)),
        np.concatenate(end_ts).astype(np.int64),
        np.concatenate(mints),
        horizons,
    )


def _label(
    window_ends: np.ndarray,
    ts_sec: np.ndarray,
    price: np.ndarray,
    liquidity: np.ndarray,
    horizon_s: np.ndarray,
    pump_thr: np.ndarray,
    rug_thr: np.ndarray,
    max_gap: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, n_h = len(window_ends), len(horizon_s)
    pump = np.zeros((n, n_h), np.float32)
    rug = np.zeros((n, n_h), np.float32)
    ret = np.zeros((n, n_h), np.float32)
    mask = np.zeros((n, n_h), np.float32)
    last_sec = ts_sec[-1]

    for k, i in enumerate(window_ends):
        p_i, l_i, t_i = price[i], liquidity[i], ts_sec[i]
        if not np.isfinite(p_i) or p_i <= 0:
            continue  # no usable anchor price
        lo = np.searchsorted(ts_sec, t_i, side="right")  # first row strictly after t_i

        for h in range(n_h):
            horizon_end = t_i + horizon_s[h]
            hi = np.searchsorted(ts_sec, horizon_end, side="right")  # rows (t_i, t_i + H]
            if hi <= lo:
                continue
            future_p = price[lo:hi]
            if not np.isfinite(future_p).any():
                continue

            p_min = np.nanmin(future_p)
            crashed = (p_i - p_min) / p_i >= rug_thr[h]
            if np.isfinite(l_i) and l_i > 0:
                future_l = liquidity[lo:hi]
                if np.isfinite(future_l).any():
                    crashed |= (l_i - np.nanmin(future_l)) / l_i >= rug_thr[h]

            covered = last_sec >= horizon_end and ts_sec[hi - 1] >= horizon_end - max_gap[h]
            if not covered and not crashed:
                continue  # outcome unknown: data ends early or has a gap

            finite = future_p[np.isfinite(future_p)]
            pump[k, h] = float(np.max(finite) / p_i - 1.0 >= pump_thr[h])
            rug[k, h] = float(crashed)
            ret[k, h] = finite[-1] / p_i - 1.0  # price at (or nearest before) the horizon end
            mask[k, h] = 1.0

    return pump, rug, ret, mask


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


def make_loaders(
    train: SequenceDataset,
    val: SequenceDataset,
    *,
    batch_size: int,
    balanced: bool = True,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train loader (class-balanced sampling by default) and an ordered, unshuffled val loader."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if balanced:
        train_loader = DataLoader(
            train, batch_size=batch_size, sampler=balanced_sampler(train, seed=seed)
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    return train_loader, DataLoader(val, batch_size=batch_size, shuffle=False)


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
    """The windows of ``df`` per ``cfg``; refuses rows predating an enabled feature group."""
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
    )


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
