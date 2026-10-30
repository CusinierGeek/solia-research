"""Liquidity floor: prices read off near-empty pools are not tradeable outcomes.

On an AMM pool a single micro-trade moves the price of a near-empty pool by multiples, so a price
read off it is an artifact, not a move anyone could trade: it fakes pump labels (a dust pool that
"jumps") and windows the model "predicts" by recognizing an empty pool. The floor (``config.yaml``
``labeling.liquidity_floor``, in SOL — ``features.liquidity`` is the SOL in the SOL-quoted pool) is
**venue-aware**: a row's floor is its venue's, else ``default``. A row is *tradeable* when its
floor is 0 or its liquidity is at least the floor; an unknown (NULL) liquidity can't be proven above
a positive floor, so it isn't.

Bonding curves price off **virtual** reserves (pumpfun: ~30 SOL virtual + the real SOL), so a fresh
curve token with almost no real SOL still has a valid, hard-to-move price: the curve venues get a
floor of 0 (decided from data, docs/ML-MODEL.md "Liquidity floor").

In the labels (:mod:`solia_ai.dataset`): a window whose anchor row isn't tradeable is not a sample,
and a price in the horizon read off an untradeable row counts neither towards ``pump`` (the highest
price) nor ``expected_return`` (the last tradeable price at or before the horizon end). ``rug`` is
unchanged: a liquidity collapse (into dust or not) is still a rug.

The floor is recorded with every trained model (``hyperparams["liquidity_floor"]``), like the
eligibility rule, so the bot's liquidity gate can trade the population its champion learned on. A
model recorded without one predates it: :meth:`LiquidityFloor.of` reads it as disabled.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import liquidity_floor_error

DEFAULT_KEY = "default"

# Bonding-curve venues (price off virtual reserves): the launchpads of the canonical venue list
# (solia_ai.features.VENUES). Reported as their own buckets by :func:`liquidity_buckets`.
CURVE_VENUES: tuple[str, ...] = ("pumpfun", "raydium-launchpad", "meteora-launchpad", "bonk")

# Bucket edges (SOL) of the per-liquidity diagnostic: dust / thin / realistic pools.
BUCKET_EDGES: tuple[float, ...] = (1.0, 10.0)


def normalize_venue(venue: Any) -> str:
    return str(venue).strip().lower()


@dataclass(frozen=True)
class LiquidityFloor:
    """The SOL floor per venue (module doc); ``default`` applies to every other venue (NULL
    included). All zero = disabled: every row is tradeable."""

    default: float = 0.0
    venues: tuple[tuple[str, float], ...] = field(default=())

    def __post_init__(self) -> None:
        if problem := liquidity_floor_error(self.as_dict()):
            raise ValueError(problem)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> LiquidityFloor:
        """The floor of a ``labeling.liquidity_floor`` mapping (``{default: 1.0, pumpfun: 0}``;
        omitted = disabled). Raises :class:`ValueError` on a non-numeric, negative or non-finite
        floor, or a venue listed twice once normalized."""
        values = {} if values is None else values
        if problem := liquidity_floor_error(values):
            raise ValueError(problem)
        venues = sorted(
            (normalize_venue(k), float(v)) for k, v in values.items() if k != DEFAULT_KEY
        )
        return cls(default=float(values.get(DEFAULT_KEY, 0.0)), venues=tuple(venues))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> LiquidityFloor:
        """The floor ``config`` labels under (``labeling.liquidity_floor``; omitted = disabled)."""
        return cls.from_mapping(config.get("labeling", {}).get("liquidity_floor"))

    @classmethod
    def of(cls, hyperparams: Mapping[str, Any] | None) -> LiquidityFloor:
        """The floor a model was trained under, from its recorded ``hyperparams``. None recorded =
        a model from before the floor: disabled."""
        return cls.from_mapping((hyperparams or {}).get("liquidity_floor"))

    @property
    def enabled(self) -> bool:
        return self.default > 0 or any(v > 0 for _, v in self.venues)

    def floor_of(self, venue: Any) -> float:
        """The floor of one venue (``default`` for an unlisted one, ``None`` / NaN included)."""
        return float(self.floors([venue], 1)[0])

    def as_dict(self) -> dict[str, float]:
        return {DEFAULT_KEY: self.default, **dict(self.venues)}

    def describe(self) -> str:
        if not self.enabled:
            return "disabled (every pool)"
        exceptions = ", ".join(f"{v} {f:g}" for v, f in self.venues)
        return f"{self.default:g} SOL" + (f" ({exceptions})" if exceptions else "")

    def floors(self, venue: Sequence[Any] | np.ndarray | pd.Series | None, n: int) -> np.ndarray:
        """``[n]`` float64 floor of each row from its ``venue`` (``None``: all ``default``)."""
        if venue is None:
            return np.full(n, self.default)
        floors = dict(self.venues)
        return _per_venue(venue, lambda v: floors.get(v, self.default), self.default, np.float64)

    def tradeable(
        self, liquidity: np.ndarray, venue: Sequence[Any] | np.ndarray | pd.Series | None
    ) -> np.ndarray:
        """Which rows (``liquidity`` in SOL, NaN = unknown; their ``venue``) are at or above their
        venue's floor. A floor of 0 accepts the row whatever its liquidity."""
        liquidity = np.asarray(liquidity, np.float64)
        floors = self.floors(venue, len(liquidity))
        with np.errstate(invalid="ignore"):
            return (floors <= 0) | (liquidity >= floors)  # NaN >= x is False


def liquidity_buckets(
    liquidity: np.ndarray, venue: Sequence[Any] | np.ndarray | pd.Series | None
) -> np.ndarray:
    """``[N]`` object array: each row's diagnostic bucket, ``<kind> <range>`` — kind ``curve``
    (:data:`CURVE_VENUES`) or ``amm`` (any other venue), range from :data:`BUCKET_EDGES`
    (``<1``, ``1-10``, ``>=10`` SOL) or ``unknown`` (NULL liquidity)."""
    liquidity = np.asarray(liquidity, np.float64)
    n = len(liquidity)
    if venue is None:
        curve = np.zeros(n, bool)
    else:
        curve = _per_venue(venue, lambda v: v in CURVE_VENUES, False, bool)
    low, high = BUCKET_EDGES
    ranges = np.full(n, "unknown", dtype=object)
    known = np.isfinite(liquidity)
    ranges[known & (liquidity < low)] = f"<{low:g}"
    ranges[known & (liquidity >= low) & (liquidity < high)] = f"{low:g}-{high:g}"
    ranges[known & (liquidity >= high)] = f">={high:g}"
    kinds = np.where(curve, "curve", "amm").astype(object)
    return kinds + " " + ranges


def _per_venue(venue: Any, value: Any, missing: Any, dtype: Any) -> np.ndarray:
    """``value(normalized venue)`` of every row (``missing`` for a NULL venue), computed once per
    distinct venue: a training read holds millions of rows but a handful of venues."""
    codes, uniques = pd.factorize(pd.Series(venue, dtype=object), use_na_sentinel=True)
    table = np.array([value(normalize_venue(v)) for v in uniques] + [missing], dtype=dtype)
    return table[codes]  # code -1 (NULL) -> the trailing `missing`
