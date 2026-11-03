"""Compact training rows for large research reads (#202 step 2).

The default training read (:func:`solia_ai.db.read_training_features`) returns a DataFrame of every
read column, and :func:`solia_ai.dataset.build_samples` copies the windows' rows out of it into
each :class:`~solia_ai.dataset.Samples` matrix. Measured with ``--prune-columns``: ~880 bytes per
row at peak (frame + chunks + concat + build temporaries + matrix). Every token of 3.5 days
(~60 M rows) would need ~50 GB.

:class:`TrainingRows` holds the same rows once, compactly:

* ``features``: the narrow ``float32`` matrix of the enabled groups' columns (``[R, C]``), built
  once and **shared** by every Samples built on it (one per stride / labeling of a search), so a
  rebuild costs only its windows' indices and labels;
* ``frame``: only what labels, eligibility and the token bookkeeping read (``mint`` as a category,
  ``ts``, ``venue``, ``price``, ``liquidity`` and the eligibility inputs, float64 as in the default
  read), sorted by ``(mint, ts)`` so every token's rows are contiguous and in time order.

Built from a frame (:meth:`TrainingRows.from_frame`, the reference) or straight from the database
without ever holding a frame of every row (:func:`solia_ai.db.read_training_rows`).
:func:`solia_ai.dataset.rows_samples` builds the windows token block by token block, so the build's
temporaries follow the block, not the read. Research CLIs only (``--prune-columns``): the windows,
labels and served inputs are the same as the default path's (``tests/test_training_rows.py``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .eligibility import INPUT_COLUMNS as ELIGIBILITY_INPUTS
from .features import ALWAYS_WRITTEN, build_feature_matrix, is_ordered_subset

# What the frame keeps besides mint / ts / venue: the labels' inputs and the eligibility rule's.
FRAME_FLOAT_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(("price", "liquidity", *ELIGIBILITY_INPUTS))
)


@dataclass(frozen=True)
class TrainingRows:
    """Training rows held once: a narrow feature matrix aligned with a slim frame (module doc)."""

    frame: pd.DataFrame  # [R] mint (category, sorted), ts, venue, FRAME_FLOAT_COLUMNS; (mint, ts)
    features: np.ndarray  # [R, C] float32, row i = frame row i
    columns: tuple[str, ...]  # the C matrix columns, an ordered subset of FEATURE_COLUMNS
    # Rows read with a NULL in each always-written column (before the matrix's NULL fill): a group
    # whose rows predate its migration is refused (:meth:`check_groups`), as in the default path.
    nulls: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.features.ndim != 2 or self.features.shape != (len(self.frame), len(self.columns)):
            raise ValueError(
                f"features must be [{len(self.frame)}, {len(self.columns)}], "
                f"got {self.features.shape}"
            )
        if self.features.dtype != np.float32:
            raise ValueError(f"features must be float32, got {self.features.dtype}")

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def n_tokens(self) -> int:
        return int(self.frame["mint"].nunique())

    @property
    def nbytes(self) -> int:
        """Bytes held: the matrix plus the frame (deep)."""
        return int(self.features.nbytes + self.frame.memory_usage(deep=False, index=False).sum())

    @classmethod
    def from_frame(cls, df: pd.DataFrame, columns: Sequence[str]) -> TrainingRows:
        """The rows of ``df`` (a training read: rows without a mint dropped, as the build drops
        them), sorted by ``(mint, ts)`` (ties in input order), with ``columns``' matrix."""
        columns = _check_matrix_columns(columns)
        if "mint" not in df.columns or "ts" not in df.columns:
            raise ValueError("feature frame needs 'mint' and 'ts' columns")
        codes, mints = pd.factorize(df["mint"], sort=True)
        known = np.flatnonzero(codes >= 0)
        ts = epoch_ns(df["ts"])
        order = known[np.lexsort((ts[known], codes[known]))]
        categories = pd.Index(mints, dtype=object)
        data = {
            "mint": pd.Categorical.from_codes(codes[order], categories=categories),
            "ts": pd.to_datetime(ts[order], utc=True),
            "venue": (
                df["venue"].to_numpy(dtype=object)[order]
                if "venue" in df.columns
                else np.full(len(order), None, object)
            ),
        }
        for col in FRAME_FLOAT_COLUMNS:
            if col in df.columns:
                values = pd.to_numeric(df[col].iloc[order], errors="coerce")
                data[col] = values.to_numpy(dtype=np.float64)
        nulls = {
            col: int(df[col].iloc[order].isna().sum()) for col in always_written_columns(df.columns)
        }
        return cls(
            pd.DataFrame(data),
            build_feature_matrix(df, rows=order, columns=columns),
            columns,
            nulls,
        )

    def check_groups(self, groups: Iterable[str]) -> None:
        """Refuse rows written before an enabled group's migration (the rule of
        :func:`solia_ai.features.check_group_coverage`).

        Raises:
            ValueError: naming the group and (at least) how many rows lack it.
        """
        for group in dict.fromkeys(groups):
            missing = [self.nulls.get(c, 0) for c in ALWAYS_WRITTEN.get(group, ())]
            if any(missing):
                raise ValueError(
                    f"at least {max(missing)} of {len(self)} feature rows predate the '{group}' "
                    "group: re-replay them before enabling it"
                )

    def token_blocks(self, max_rows: int) -> list[tuple[int, int]]:
        """``[lo, hi)`` row ranges of whole tokens, each holding at most ``max_rows`` rows unless a
        single token is larger (it is never split)."""
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows}")
        codes = self.frame["mint"].cat.codes.to_numpy()
        if len(codes) == 0:
            return []
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        bounds = np.r_[starts, len(codes)]
        blocks, lo = [], 0
        for i in range(1, len(bounds)):
            if bounds[i] - lo > max_rows and bounds[i - 1] > lo:
                blocks.append((lo, int(bounds[i - 1])))
                lo = int(bounds[i - 1])
        blocks.append((lo, len(codes)))
        return blocks


def always_written_columns(available: Iterable[str]) -> list[str]:
    """The always-written columns (every group's) among ``available``."""
    present = set(available)
    return list(dict.fromkeys(c for cols in ALWAYS_WRITTEN.values() for c in cols if c in present))


def _check_matrix_columns(columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(columns)
    if not columns or not is_ordered_subset(columns):
        raise ValueError(
            f"matrix columns must be a non-empty ordered subset of FEATURE_COLUMNS, got {columns}"
        )
    return columns


def epoch_ns(ts: pd.Series) -> np.ndarray:
    naive = pd.to_datetime(ts, utc=True).dt.tz_localize(None).to_numpy()
    return naive.astype("datetime64[ns]").astype(np.int64)
