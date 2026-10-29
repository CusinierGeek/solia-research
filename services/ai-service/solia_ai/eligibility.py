"""Token eligibility: which feature rows the model trains on and the bot may trade.

Most Pump.fun tokens are created and never really traded. Training on them spends the model's
capacity on noise, and scoring them live wastes ``/predict`` calls. The rule that drops them must be
**causal** — a row's eligibility depends only on that row and the token's earlier rows, never on
what happened next (dropping the tokens that *turned out* dead trains on an optimistic population) —
and **identical** in training and trading, so the model learns on exactly the population it trades.
The .NET bot implements the same rule (``Solia.Bot.Policy.EligibilityRule``); the shared fixture
``services/shared/testdata/eligibility-cases.json`` keeps the two in lockstep.

A row is eligible when every enabled criterion holds (a threshold of 0 disables its criterion):

* ``min_unique_traders`` — the most distinct traders any snapshot of the token has counted so far
  (``max(unique_traders)`` over the rows up to this one). The pipeline's ``unique_traders`` is per
  rolling window, so its running maximum is a causal lower bound of the traders seen so far;
* ``min_volume_sol`` — the highest window SOL ``volume`` any snapshot has reported so far (a sum of
  overlapping rolling windows would count each trade many times, a running maximum does not);
* ``min_token_age_sec`` — the oldest ``token_age_sec`` known so far (the row's own age, or the last
  known one when the row's is NULL).

NULL inputs carry no information: they don't raise a running maximum, so a token with no known
value yet fails that (enabled) criterion — an unknown age can't be proven old enough. Every
criterion is a running maximum, so once eligible a token stays eligible.

The rule is configured once (``config.yaml`` ``eligibility``), recorded with every trained model
(``hyperparams["eligibility"]``, in the checkpoint and the ``models`` row) and read back by the bot
from the champion's row, so the bot trades the population its champion was trained on. A model
recorded without a rule predates it and was trained on every token: :meth:`EligibilityRule.of`
reads it as the disabled rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ELIGIBILITY_DEFAULTS, eligibility_error

# The `features` columns the rule reads.
INPUT_COLUMNS = ("unique_traders", "volume", "token_age_sec")


@dataclass(frozen=True)
class EligibilityRule:
    """The thresholds of the eligibility rule (module doc); all zero = every row is eligible."""

    min_unique_traders: int = 0
    min_volume_sol: float = 0.0
    min_token_age_sec: float = 0.0

    def __post_init__(self) -> None:
        if problem := eligibility_error(asdict(self)):
            raise ValueError(problem)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> EligibilityRule:
        """The rule of a config ``eligibility`` section (omitted keys keep their 0 default).
        Raises :class:`ValueError` on unknown keys, non-numeric or negative thresholds."""
        values = {} if values is None else values
        if not isinstance(values, Mapping):
            raise ValueError(f"'eligibility' must be a mapping, got {values!r}")
        if problem := eligibility_error(dict(values)):
            raise ValueError(problem)
        merged = {**ELIGIBILITY_DEFAULTS, **values}
        return cls(
            min_unique_traders=int(merged["min_unique_traders"]),
            min_volume_sol=float(merged["min_volume_sol"]),
            min_token_age_sec=float(merged["min_token_age_sec"]),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> EligibilityRule:
        """The rule ``config`` trains under (its ``eligibility`` section; omitted = disabled)."""
        return cls.from_mapping(config.get("eligibility"))

    @classmethod
    def of(cls, hyperparams: Mapping[str, Any] | None) -> EligibilityRule:
        """The rule a model was trained under, from its recorded ``hyperparams`` (a checkpoint's or
        a ``models`` row's). None recorded = a model from before the rule: the disabled rule."""
        return cls.from_mapping((hyperparams or {}).get("eligibility"))

    @property
    def enabled(self) -> bool:
        return self != EligibilityRule()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if not self.enabled:
            return "disabled (every token)"
        return (
            f"traders>={self.min_unique_traders}, volume>={self.min_volume_sol:g} SOL, "
            f"age>={self.min_token_age_sec:g}s"
        )

    def accepts(self, inputs: np.ndarray) -> np.ndarray:
        """Which rows of ``inputs`` (``[N, 3]``, :func:`running_inputs`: the running maxima of
        :data:`INPUT_COLUMNS`, NaN = none known yet) the rule accepts."""
        inputs = np.asarray(inputs, np.float64).reshape(-1, len(INPUT_COLUMNS))
        ok = np.ones(len(inputs), bool)
        thresholds = (self.min_unique_traders, self.min_volume_sol, self.min_token_age_sec)
        for k, threshold in enumerate(thresholds):
            if threshold > 0:
                ok &= inputs[:, k] >= threshold  # NaN >= x is False: no known value yet fails
        return ok

    def eligible(
        self, unique_traders: np.ndarray, volume: np.ndarray, token_age_sec: np.ndarray
    ) -> np.ndarray:
        """Per-row eligibility of one token's rows, **in time order** (NaN = NULL): row ``i``
        depends on rows ``0..i`` only."""
        return self.accepts(running_inputs(unique_traders, volume, token_age_sec))

    def eligible_frame(self, rows: pd.DataFrame) -> np.ndarray:
        """:meth:`eligible` over one token's ``features`` rows, already sorted in time order."""
        if not self.enabled:
            return np.ones(len(rows), bool)
        self.check_columns(rows)
        return self.accepts(frame_inputs(rows))

    def check_columns(self, frame: pd.DataFrame) -> None:
        """Raise unless an enabled rule finds its input columns in ``frame``."""
        missing = [c for c in INPUT_COLUMNS if c not in frame.columns]
        if self.enabled and missing:
            raise ValueError(f"the eligibility rule needs column(s): {', '.join(missing)}")


def running_inputs(
    unique_traders: np.ndarray, volume: np.ndarray, token_age_sec: np.ndarray
) -> np.ndarray:
    """``[T, 3]`` running maxima of one token's :data:`INPUT_COLUMNS` in time order (NaN until
    the first known value): what the rule judges each row on."""
    columns = (np.asarray(c, np.float64) for c in (unique_traders, volume, token_age_sec))
    return np.stack([_running_max(c) for c in columns], axis=1)


def frame_inputs(rows: pd.DataFrame) -> np.ndarray:
    """:func:`running_inputs` of one token's ``features`` rows sorted in time order; a missing
    column is all NaN (only a disabled rule accepts it, see :meth:`EligibilityRule.check_columns`).
    """
    nan = np.full(len(rows), np.nan)
    return running_inputs(
        *(
            pd.to_numeric(rows[c], errors="coerce").to_numpy(np.float64) if c in rows else nan
            for c in INPUT_COLUMNS
        )
    )


def _running_max(values: np.ndarray) -> np.ndarray:
    """Running maximum ignoring NaN (NaN until the first known value)."""
    if len(values) == 0:
        return values
    return np.fmax.accumulate(values)
