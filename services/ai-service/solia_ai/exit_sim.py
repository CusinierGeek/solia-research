"""Exit-rule simulation: what an entry rule earns once the bot's exits apply (#221).

The entry sweep (:mod:`solia_ai.entry_sweep`) scores a rule by its label held to the horizon, on
**labeled** windows. The bot does not hold: it exits on take-profit, stop-loss or timeout, one
position per mint, with latency and costs. This read-only command scores a checkpoint once over its
held-out windows, then walks each selected entry along the token's real ``features`` rows, for a
grid of exit settings. Nothing is written but the report, and paper only: nothing is ever executed.

**Every selected window counts, labeled or not.** A token that goes quiet without crashing has no
15m label, so it is missing from the sweep's numbers, though nobody can know that at entry
(:mod:`solia_ai.dataset`, "covered"). :func:`selection_report` therefore breaks the gap between the
sweep and the simulation into steps: the labeled windows (the sweep), every selected window, one
position per mint, first signal per mint, and how many of those never move again.

The trade mechanics mirror ``Solia.Bot`` (``PositionManager``, ``ExitLevels``,
``PositionSizer.Gain``, ``Fills``; docs/ML-MODEL.md "Policy layer"). The bot's defaults live in the
constants below, and ``tests/test_exit_sim.py`` checks them against ``BotOptions.cs``. Marks are
the ~10 s ``features`` rows, so a fill waits for the next row. Not modelled: the rug exit (the
current rule has no rug gate), ``MaxConcurrentPositions``, a compounding balance (the position size
is the starting balance's fixed fraction), a silent feed (it is assumed live, so a stale mark fills
at the last mark), MEV.

**Partial sales and trailing stops** (:class:`ExitRule`) are research options: the bot has neither.
Each sale goes through the same latency, impact, slippage retries and fees as an exit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch

from .evaluate import apply_calibration, fmt, json_safe, score_logits
from .liquidity import LiquidityFloor, normalize_venue

# --- The bot's defaults (BotOptions.cs), in one place. A parity test reads them back.
TAKE_PROFIT_PCT = 0.5
MIN_TAKE_PROFIT_PCT = 0.05
STOP_LOSS_PCT = 0.3
STOP_LOSS_VOLATILITY_MULTIPLE = 3.0
MIN_STOP_LOSS_PCT = 0.05
POSITION_TIMEOUT_SECONDS = 900
STARTING_BALANCE = 10.0
FIXED_POSITION_FRACTION = 0.1
MIN_POSITION_SIZE = 0.01
MAX_POOL_FRACTION = 0.02
ENTRY_LATENCY_SECONDS = 3.0
EXIT_LATENCY_SECONDS = 2.0
MAX_MARK_AGE_SECONDS = 60.0
NETWORK_FEE_SOL = 0.0002
ACCOUNT_RENT_SOL = 0.00203928
MAX_SLIPPAGE_PCT = 0.15
MAX_EXIT_RETRIES = 3
FEE_PCT = {
    "default": 0.0025,
    "pumpfun": 0.0125,
    "pumpswap": 0.003,
    "raydium-launchpad": 0.01,
    "meteora-launchpad": 0.01,
    "bonk": 0.01,
}
VIRTUAL_SOL = {"pumpfun": 30.0, "raydium-launchpad": 30.0, "meteora-launchpad": 30.0, "bonk": 30.0}

# --- The grid (#221).
GRID_STOPS = (None, 0.1, 0.2, 0.3, 0.5, "vol3", "vol5", "vol8")
GRID_TAKE_PROFITS = ("cap", 0.5, 1.0, 2.0, None)
GRID_TIMEOUTS = (300, 900, 1800)

DEFAULT_PUMP_THRESHOLDS = (0.3, 0.4, 0.5)
DEFAULT_HORIZON = "15m"
DEFAULT_MIN_TRADES = 30
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CI = 0.95
DEFAULT_MAX_RETURN = 10.0  # the labels' cap (labeling.max_return), for the hold-to-horizon return

FILLED, FAILED, ABANDONED, DECLINED = "filled", "failed", "abandoned", "declined"
REASONS = ("tp", "sl", "timeout", "trail")
_PARTIAL = "partial"  # an internal trigger: the position stays open, so it is never an exit reason
STAKE = "stake"  # a partial sale of whatever recovers the position's size
REST_STOPS = ("keep", "entry", "none")


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class Execution:
    """How orders fill: the bot's defaults, or :data:`IDEAL` (at the mark, at once, for free)."""

    realistic_fills: bool = True
    entry_latency_s: float = ENTRY_LATENCY_SECONDS
    exit_latency_s: float = EXIT_LATENCY_SECONDS
    max_mark_age_s: float = MAX_MARK_AGE_SECONDS
    position_size: float = STARTING_BALANCE * FIXED_POSITION_FRACTION
    min_position_size: float = MIN_POSITION_SIZE
    max_pool_fraction: float = MAX_POOL_FRACTION
    network_fee: float = NETWORK_FEE_SOL
    account_rent: float = ACCOUNT_RENT_SOL
    max_slippage: float = MAX_SLIPPAGE_PCT
    max_exit_retries: int = MAX_EXIT_RETRIES
    fee_pct: Mapping[str, float] = field(default_factory=lambda: dict(FEE_PCT))
    virtual_sol: Mapping[str, float] = field(default_factory=lambda: dict(VIRTUAL_SOL))


BOT_EXECUTION = Execution()
IDEAL = Execution(
    realistic_fills=False,
    entry_latency_s=0.0,
    exit_latency_s=0.0,
    network_fee=0.0,
    account_rent=0.0,
    max_slippage=0.0,
)


@dataclass(frozen=True)
class ExitRule:
    """One cell of the grid.

    ``take_profit``: ``"cap"`` is the bot's ``PositionSizer.Gain``, ``min(expected_return,
    TakeProfitPct)``; a number is a fixed level; ``None`` means no take-profit.
    ``stop_loss``: a number is a fixed level; ``"volN"`` is the bot's volatility stop with multiple
    N; ``None`` means no stop-loss.

    ``partial_at``: at this gain from the fill mark, sell ``partial_share`` of the position, once:
    a fraction of it, or :data:`STAKE` for what recovers its size at that mark (everything if the
    pool can't pay that, which closes the position as a take-profit). The rest then follows
    ``take_profit``, ``rest_stop`` (``"keep"`` the stop, ``"entry"`` move it to the fill mark,
    ``"none"`` drop it) and ``rest_timeout_s`` from the fill (default: ``timeout_s``).

    ``trail``: a stop at this distance under the highest mark since the fill, from the fill or, with
    ``trail_after``, once a mark reached that gain. It applies to whatever is still held.
    """

    take_profit: float | str | None = "cap"
    stop_loss: float | str | None = f"vol{STOP_LOSS_VOLATILITY_MULTIPLE:g}"
    timeout_s: float = POSITION_TIMEOUT_SECONDS
    partial_at: float | None = None
    partial_share: float | str = STAKE
    rest_stop: str = "keep"
    rest_timeout_s: float | None = None
    trail: float | None = None
    trail_after: float | None = None

    def __post_init__(self) -> None:
        tp, sl = self.take_profit, self.stop_loss
        if not (tp is None or tp == "cap" or (_is_number(tp) and tp > 0)):
            raise ValueError(f"take_profit must be 'cap', None or > 0, got {tp!r}")
        if not (sl is None or _vol_multiple(sl) is not None or (_is_number(sl) and 0 < sl < 1)):
            raise ValueError(f"stop_loss must be 'volN', None or in (0, 1), got {sl!r}")
        if not (_is_number(self.timeout_s) and self.timeout_s > 0):
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s!r}")
        self._check_partial()
        self._check_trail()

    def _check_partial(self) -> None:
        at, share, rest = self.partial_at, self.partial_share, self.rest_timeout_s
        if at is None:
            if share != STAKE or self.rest_stop != "keep" or rest is not None:
                raise ValueError("partial_share, rest_stop and rest_timeout_s need partial_at")
            return
        if not (_is_number(at) and at > 0):
            raise ValueError(f"partial_at must be > 0, got {at!r}")
        if not (share == STAKE or (_is_number(share) and 0 < share < 1)):
            raise ValueError(f"partial_share must be {STAKE!r} or in (0, 1), got {share!r}")
        if self.rest_stop not in REST_STOPS:
            raise ValueError(f"rest_stop must be one of {REST_STOPS}, got {self.rest_stop!r}")
        if not (self.take_profit is None or _is_number(self.take_profit)):
            raise ValueError("a partial sale needs a fixed take_profit or none")
        if self.take_profit is not None and self.take_profit <= at:
            raise ValueError("take_profit must be above partial_at")
        if rest is not None and not (_is_number(rest) and rest >= self.timeout_s):
            raise ValueError(f"rest_timeout_s must be >= timeout_s, got {rest!r}")

    def _check_trail(self) -> None:
        if self.trail is None:
            if self.trail_after is not None:
                raise ValueError("trail_after needs trail")
            return
        if not (_is_number(self.trail) and 0 < self.trail < 1):
            raise ValueError(f"trail must be in (0, 1), got {self.trail!r}")
        if self.trail_after is not None and not (
            _is_number(self.trail_after) and self.trail_after > 0
        ):
            raise ValueError(f"trail_after must be > 0, got {self.trail_after!r}")

    @property
    def label(self) -> str:
        tp = "TP none" if self.take_profit is None else f"TP {self.take_profit}"
        sl = "SL none" if self.stop_loss is None else f"SL {self.stop_loss}"
        parts = [tp, sl, f"{self.timeout_s:g}s"]
        if self.partial_at is not None:
            share = self.partial_share if self.partial_share == STAKE else f"{self.partial_share:g}"
            rest = f"rest SL {self.rest_stop}"
            if self.rest_timeout_s is not None:
                rest += f" {self.rest_timeout_s:g}s"
            parts.append(f"sell {share} at +{self.partial_at:g}, {rest}")
        if self.trail is not None:
            after = "" if self.trail_after is None else f" after +{self.trail_after:g}"
            parts.append(f"trail {self.trail:g}{after}")
        return " / ".join(parts)

    def levels(self, expected_return: float, volatility: float) -> tuple[float, float]:
        """``(take-profit, stop-loss)`` distances; ``inf`` / ``1`` mean the level never fires."""
        return take_profit_level(self.take_profit, expected_return), stop_loss_level(
            self.stop_loss, volatility
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def _vol_multiple(stop: Any) -> float | None:
    if not (isinstance(stop, str) and stop.startswith("vol")):
        return None
    try:
        multiple = float(stop[3:])
    except ValueError:
        return None
    return multiple if multiple > 0 else None


def take_profit_level(take_profit: float | str | None, expected_return: float) -> float:
    if take_profit is None:
        return math.inf
    if take_profit == "cap":
        return min(expected_return, TAKE_PROFIT_PCT)
    return float(take_profit)


def stop_loss_level(stop_loss: float | str | None, volatility: float) -> float:
    """``ExitLevels.Compute``: multiple x volatility, floored then capped; the cap when the
    volatility is unknown. ``1`` (a price of 0, which is never a mark) means no stop."""
    if stop_loss is None:
        return 1.0
    multiple = _vol_multiple(stop_loss)
    if multiple is None:
        return float(stop_loss)
    if not (math.isfinite(volatility) and volatility > 0):
        return STOP_LOSS_PCT
    return min(max(multiple * volatility, MIN_STOP_LOSS_PCT), STOP_LOSS_PCT)


def has_upside(expected_return: float) -> bool:
    """``PositionSizer.Gain`` > 0: the entry gate on the predicted return, the same for every cell,
    so that the cells compare the same entries."""
    return math.isfinite(expected_return) and expected_return >= max(MIN_TAKE_PROFIT_PCT, 1e-12)


def default_grid() -> list[ExitRule]:
    return [
        ExitRule(take_profit=tp, stop_loss=sl, timeout_s=t)
        for sl in GRID_STOPS
        for tp in GRID_TAKE_PROFITS
        for t in GRID_TIMEOUTS
    ]


BOT_RULE = ExitRule()

# Today's frozen paper-test setting (docs/ML-MODEL.md, "Exit-rule simulation").
REFERENCE_RULE = ExitRule(take_profit=2.0, stop_loss="vol3", timeout_s=900)


def variant_grid() -> list[ExitRule]:
    """Partial sales and trailing stops against :data:`REFERENCE_RULE`. A short list fixed before
    any full run: the more cells, the more one of them wins by chance."""

    def partial(at: float, share: float | str, target: float | None, rest: str, t: float | None):
        return ExitRule(
            take_profit=target,
            stop_loss="vol3",
            timeout_s=900,
            partial_at=at,
            partial_share=share,
            rest_stop=rest,
            rest_timeout_s=t,
        )

    def trailing(target: float | None, distance: float, after: float | None):
        return ExitRule(
            take_profit=target, stop_loss="vol3", timeout_s=900, trail=distance, trail_after=after
        )

    return [
        REFERENCE_RULE,
        ExitRule(take_profit=3.0, stop_loss="vol3", timeout_s=900),
        partial(2.0, STAKE, 3.0, "none", None),
        partial(2.0, STAKE, 3.0, "none", 3600),
        partial(2.0, STAKE, 3.0, "entry", None),
        partial(2.0, STAKE, 3.0, "entry", 3600),
        partial(2.0, STAKE, 5.0, "none", None),
        partial(1.0, 0.5, 3.0, "none", None),
        partial(1.0, 0.5, 3.0, "entry", None),
        trailing(None, 0.3, 1.0),
        trailing(None, 0.5, 1.0),
        trailing(None, 0.3, 2.0),
        trailing(3.0, 0.3, 1.0),
    ]


# --------------------------------------------------------------------------- price paths


@dataclass(frozen=True)
class Paths:
    """Every token's valid marks (a non-positive price is a broken feed: the bot ignores it), time
    ordered, tokens contiguous. ``reserve`` is the fills' SOL reserve ``R`` (NaN = unknown)."""

    ts: np.ndarray  # [R] float64 epoch seconds
    price: np.ndarray  # [R] float64
    reserve: np.ndarray  # [R] float64
    fee: np.ndarray  # [R] float64, per side
    tradeable: np.ndarray  # [R] bool: at or above the liquidity floor
    volatility: np.ndarray  # [R] float64 (NaN = unknown)
    spans: Mapping[Any, tuple[int, int]]  # mint -> [start, stop)

    @property
    def data_end(self) -> float:
        return float(self.ts.max()) if len(self.ts) else math.nan

    @classmethod
    def from_frame(
        cls, df: pd.DataFrame, *, floor: LiquidityFloor, execution: Execution = BOT_EXECUTION
    ) -> Paths:
        missing = {"mint", "ts", "price", "liquidity", "venue", "volatility"} - set(df.columns)
        if missing:
            raise ValueError(f"the price paths need column(s): {', '.join(sorted(missing))}")
        order, price, ts, codes, mints = path_rows(df)

        liquidity = pd.to_numeric(df["liquidity"], errors="coerce").to_numpy(np.float64)[order]
        venue = df["venue"].to_numpy(dtype=object)[order]
        virtual = _per_venue(venue, execution.virtual_sol, 0.0)
        with np.errstate(invalid="ignore"):
            reserve = np.where(liquidity >= 0, liquidity + virtual, np.nan)
        volatility = pd.to_numeric(df["volatility"], errors="coerce").to_numpy(np.float64)[order]

        token_codes = codes[order]
        edges = (
            np.flatnonzero(np.r_[True, token_codes[1:] != token_codes[:-1]]) if len(order) else []
        )
        stops = np.r_[edges[1:], len(order)] if len(order) else []
        spans = {mints[token_codes[a]]: (int(a), int(b)) for a, b in zip(edges, stops, strict=True)}
        return cls(
            ts=ts[order],
            price=price[order],
            reserve=reserve,
            fee=_per_venue(venue, execution.fee_pct, execution.fee_pct.get("default", 0.0)),
            tradeable=floor.tradeable(liquidity, venue),
            volatility=volatility,
            spans=spans,
        )


def path_rows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """``order`` maps each mark of :class:`Paths` to its row of ``df``: only valid marks, tokens
    contiguous, time ordered. Also returns the frame's prices, epoch seconds and mint codes, and the
    mints. Another column of ``df`` lines up with the paths through ``values[order]``."""
    price = pd.to_numeric(df["price"], errors="coerce").to_numpy(np.float64)
    valid = np.flatnonzero(np.isfinite(price) & (price > 0) & df["mint"].notna().to_numpy())
    ts = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).to_numpy("datetime64[ms]")
    ts = ts.astype(np.int64).astype(np.float64) / 1000.0
    codes, mints = pd.factorize(df["mint"], sort=True)
    return valid[np.lexsort((ts[valid], codes[valid]))], price, ts, codes, mints


def _per_venue(venue: np.ndarray, table: Mapping[str, float], fallback: float) -> np.ndarray:
    lowered = {normalize_venue(k): float(v) for k, v in table.items()}
    codes, uniques = pd.factorize(pd.Series(venue, dtype=object), use_na_sentinel=True)
    values = [lowered.get(normalize_venue(v), fallback) for v in uniques] + [fallback]
    return np.array(values, np.float64)[codes]


# --------------------------------------------------------------------------- one position


class Outcome(NamedTuple):
    status: str  # FILLED / FAILED / ABANDONED / DECLINED
    free_at: float  # when the mint can take another entry
    reason: str | None = None
    net_return: float = 0.0  # pnl / size, costs included
    pnl_sol: float = 0.0
    hold_s: float = 0.0
    partial: bool = False  # a partial sale filled before the close


def _buy_price(mark: float, reserve: float, fee: float, size: float) -> float:
    keep = 1.0 - fee
    return mark * (reserve + size * keep) / (reserve * keep)


def _sell_price(
    mark: float,
    reserve: float,
    fee: float,
    size: float,
    entry_price: float,
    mark_is_realistic: bool,
) -> float:
    if not mark_is_realistic:
        return mark
    if not (reserve > 0):  # a NaN (unknown) reserve too: an empty or unknown pool pays nothing
        return 0.0
    return mark * reserve * (1.0 - fee) / (reserve + size * mark / entry_price)


def _stake_size(
    mark: float, reserve: float, fee: float, size: float, entry_price: float, realistic: bool
) -> float:
    """The part of a position of ``size`` whose sale at ``mark`` pays ``size`` back, in the same
    unit: at the mark, or through the pool's impact. ``size`` when the pool can't pay that."""
    gain = mark / entry_price
    if not realistic:
        return min(size / gain, size)
    room = reserve * (1.0 - fee) - size
    if not (reserve > 0 and room > 0):
        return size
    return min(size * reserve / room / gain, size)


class _Sale(NamedTuple):
    closed: float
    price: float  # per token
    attempts: int
    next_row: int  # where the walk resumes


def _fill_sale(
    paths: Paths,
    stop: int,
    k: int,
    trigger_at: float,
    trigger_row: int,
    sold: float,
    entry_price: float,
    execution: Execution,
) -> _Sale:
    """One sell order: at once, or at the first mark after the latency, with the slippage retries;
    with no mark in time, the stale-mark rule (the feed is assumed live), the last mark."""
    ts, price, reserve, fee = paths.ts, paths.price, paths.reserve, paths.fee

    def sell(i: int) -> float:
        return _sell_price(
            float(price[i]),
            float(reserve[i]),
            float(fee[i]),
            sold,
            entry_price,
            execution.realistic_fills,
        )

    quote = sell(trigger_row)
    if execution.exit_latency_s <= 0:
        return _Sale(trigger_at, quote, 1, k)
    due, failures = trigger_at + execution.exit_latency_s, 0
    while True:
        at = int(np.searchsorted(ts[k:stop], due, side="left")) + k
        if at >= stop or ts[at] > due + execution.max_mark_age_s:
            return _Sale(due + execution.max_mark_age_s, sell(at - 1), failures + 1, at)
        filled = sell(at)
        closed = float(ts[at])
        breached = (
            execution.max_slippage > 0
            and failures < execution.max_exit_retries
            and filled < quote * (1.0 - execution.max_slippage)
        )
        if not breached:
            return _Sale(closed, filled, failures + 1, at + 1)
        quote = filled
        due = closed + execution.exit_latency_s
        failures += 1
        k = at + 1


def simulate_position(
    paths: Paths,
    row: int,
    stop: int,
    expected_return: float,
    rule: ExitRule,
    execution: Execution = BOT_EXECUTION,
) -> Outcome:
    """One entry decided on mark ``row`` (of a token whose marks end before ``stop``), walked to its
    close the way ``PositionManager`` does: see the module doc for what isn't modelled."""
    ts, price, reserve, fee = paths.ts, paths.price, paths.reserve, paths.fee
    decided = float(ts[row])
    if not has_upside(expected_return):
        return Outcome(DECLINED, decided)

    realistic = execution.realistic_fills
    size = execution.position_size
    quote = float(price[row])
    if realistic:
        if not (reserve[row] > 0):  # Fills.IsTradeable (the windows are already at the floor)
            return Outcome(DECLINED, decided)
        size = min(size, execution.max_pool_fraction * float(reserve[row]))
        if size < execution.min_position_size:
            return Outcome(DECLINED, decided)
        quote = _buy_price(float(price[row]), float(reserve[row]), float(fee[row]), size)

    # Entry: the first mark at or after decision + latency, within the mark age; else abandoned.
    fill = row
    if execution.entry_latency_s > 0:
        due = decided + execution.entry_latency_s
        fill = int(np.searchsorted(ts[row + 1 : stop], due, side="left")) + row + 1
        if fill >= stop or ts[fill] > due + execution.max_mark_age_s:
            return Outcome(ABANDONED, due + execution.max_mark_age_s)

    opened = float(ts[fill])
    mark = float(price[fill])
    entry_price = mark
    if realistic:
        if not (reserve[fill] > 0):
            return Outcome(FAILED, opened)
        entry_price = _buy_price(mark, float(reserve[fill]), float(fee[fill]), size)
    if execution.max_slippage > 0 and entry_price > quote * (1.0 + execution.max_slippage):
        return Outcome(FAILED, opened)

    tp, sl = rule.levels(expected_return, float(paths.volatility[row]))
    tp_price, sl_price = mark * (1.0 + tp), mark * (1.0 - sl)
    deadline = opened + rule.timeout_s
    partial_price = math.inf if rule.partial_at is None else mark * (1.0 + rule.partial_at)
    trail_keep = None if rule.trail is None else 1.0 - rule.trail
    armed_at = mark if rule.trail_after is None else mark * (1.0 + rule.trail_after)

    held, proceeds, attempts, peak, partial_filled = size, 0.0, 0, mark, False
    k = fill + 1
    while True:
        # Triggers: the deadline is swept before a mark at or after it; then SL, trail, partial, TP.
        reason, trigger_at, trigger_row = "timeout", deadline, stop - 1
        while k < stop:
            if ts[k] >= deadline:
                trigger_row = k - 1
                break
            p = price[k]
            trail_price = peak * trail_keep if trail_keep and peak >= armed_at else 0.0
            if p <= sl_price:
                reason = "sl"
            elif p <= trail_price:
                reason = "trail"
            elif p >= partial_price:
                reason = _PARTIAL
            elif p >= tp_price:
                reason = "tp"
            else:
                peak = max(peak, float(p))
                k += 1
                continue
            trigger_at, trigger_row = float(ts[k]), k
            k += 1
            break

        sold = held
        if reason == _PARTIAL:
            sold = (
                held * rule.partial_share
                if rule.partial_share != STAKE
                else _stake_size(
                    float(price[trigger_row]),
                    float(reserve[trigger_row]),
                    float(fee[trigger_row]),
                    held,
                    entry_price,
                    realistic,
                )
            )
        sale = _fill_sale(paths, stop, k, trigger_at, trigger_row, sold, entry_price, execution)
        proceeds += sold * sale.price / entry_price
        attempts += sale.attempts
        k = sale.next_row
        if reason != _PARTIAL or sold >= held:
            break

        held -= sold
        partial_filled = True
        partial_price = math.inf
        if rule.rest_stop == "none":
            sl_price = 0.0
        elif rule.rest_stop == "entry":
            sl_price = mark
        if rule.rest_timeout_s is not None:
            deadline = opened + rule.rest_timeout_s
        deadline = max(deadline, sale.closed)

    if reason == _PARTIAL:  # the whole position went in the "partial" sale
        reason = "tp"
    entry_cost = execution.network_fee + execution.account_rent
    pnl = proceeds - size - entry_cost - execution.network_fee * attempts
    hold = sale.closed - opened
    return Outcome(FILLED, sale.closed, reason, pnl / size, pnl, hold, partial_filled)


# --------------------------------------------------------------------------- windows


@dataclass(frozen=True)
class Windows:
    """The held-out windows of one horizon, in the paths' terms."""

    mint: np.ndarray  # [N] object
    row: np.ndarray  # [N] int64: the decision mark in Paths
    pump_prob: np.ndarray  # [N] float64
    expected_return: np.ndarray  # [N] float64, the prediction
    labeled: np.ndarray  # [N] bool: the sweep's windows

    def __len__(self) -> int:
        return len(self.row)

    def take(self, keep: np.ndarray) -> Windows:
        return Windows(
            self.mint[keep],
            self.row[keep],
            self.pump_prob[keep],
            self.expected_return[keep],
            self.labeled[keep],
        )

    @classmethod
    def align(
        cls,
        paths: Paths,
        mint: np.ndarray,
        end_ts: np.ndarray,
        pump_prob: np.ndarray,
        expected_return: np.ndarray,
        labeled: np.ndarray,
    ) -> Windows:
        """Finds each window's decision mark (its token's last mark of ``end_ts``, epoch seconds).
        A window whose anchor is not a valid mark is dropped: the bot would have no market."""
        rows = np.full(len(mint), -1, np.int64)
        whole = np.floor(paths.ts)
        for i, (m, t) in enumerate(zip(mint, end_ts, strict=True)):
            span = paths.spans.get(m)
            if span is None:
                continue
            row = int(np.searchsorted(whole[span[0] : span[1]], t, side="right")) + span[0] - 1
            if row >= span[0] and whole[row] == t:
                rows[i] = row
        keep = rows >= 0
        return cls(
            np.asarray(mint, dtype=object)[keep],
            rows[keep],
            np.asarray(pump_prob, np.float64)[keep],
            np.asarray(expected_return, np.float64)[keep],
            np.asarray(labeled, bool)[keep],
        )


def uncensored(windows: Windows, paths: Paths, longest_s: float) -> Windows:
    """Drops the windows too close to the end of the read for the longest hold: their path is cut
    by the read, not by the token, so it would pass for a token going quiet."""
    return windows.take(paths.ts[windows.row] + longest_s <= paths.data_end)


def simulate_rule(
    paths: Paths, windows: Windows, rule: ExitRule, execution: Execution = BOT_EXECUTION
) -> list[tuple[Any, Outcome]]:
    """Every window, in time order per mint: an entry is decided only when the mint holds no
    position and no order in flight, as in the bot. Returns ``(mint, outcome)`` of the entries."""
    out: list[tuple[Any, Outcome]] = []
    order = np.lexsort((windows.row,))
    busy_until: dict[Any, float] = {}
    for i in order:
        mint, row = windows.mint[i], int(windows.row[i])
        if paths.ts[row] < busy_until.get(mint, -math.inf):
            continue
        stop = paths.spans[mint][1]
        outcome = simulate_position(
            paths, row, stop, float(windows.expected_return[i]), rule, execution
        )
        busy_until[mint] = outcome.free_at
        out.append((mint, outcome))
    return out


# --------------------------------------------------------------------------- metrics


def bootstrap_by_token(
    values: np.ndarray, tokens: np.ndarray, *, samples: int, ci: float, rng: np.random.Generator
) -> tuple[float | None, float | None]:
    """Percentile CI of the mean, drawing **tokens**: trades of one token are not independent.
    ``(None, None)`` under two tokens."""
    codes, uniques = pd.factorize(pd.Series(tokens, dtype=object))
    m = len(uniques)
    if m < 2:
        return None, None
    totals = np.bincount(codes, weights=values, minlength=m)
    counts = np.bincount(codes, minlength=m).astype(np.float64)
    draws = rng.integers(0, m, size=(samples, m))
    means = totals[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1.0 - ci) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def cell_metrics(
    rule: ExitRule,
    entries: Sequence[tuple[Any, Outcome]],
    *,
    execution: Execution,
    samples: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """One row of the grid. ``failed_entries`` / ``abandoned_entries`` / ``declined_entries`` are
    not trades; ``total_pnl_sol`` includes the network fee a failed entry still pays."""
    trades = [(m, o) for m, o in entries if o.status == FILLED]
    returns = np.array([o.net_return for _, o in trades], np.float64)
    tokens = np.array([m for m, _ in trades], dtype=object)
    failed = sum(o.status == FAILED for _, o in entries)
    row: dict[str, Any] = {
        "take_profit": rule.take_profit,
        "stop_loss": rule.stop_loss,
        "timeout_s": rule.timeout_s,
        "partial_at": rule.partial_at,
        "partial_share": rule.partial_share if rule.partial_at is not None else None,
        "rest_stop": rule.rest_stop if rule.partial_at is not None else None,
        "rest_timeout_s": rule.rest_timeout_s,
        "trail": rule.trail,
        "trail_after": rule.trail_after,
        "label": rule.label,
        "is_bot_default": rule == BOT_RULE,
        "n_trades": len(trades),
        "n_tokens": len(set(tokens.tolist())),
        "failed_entries": failed,
        "abandoned_entries": sum(o.status == ABANDONED for _, o in entries),
        "declined_entries": sum(o.status == DECLINED for _, o in entries),
    }
    if not trades:
        return row | {
            "win_rate": None,
            "mean_return": None,
            "median_return": None,
            "mean_return_ci": [None, None],
            "total_pnl_sol": -failed * execution.network_fee,
            "exit_share": dict.fromkeys(REASONS, None),
            "median_hold_s": None,
            "top_token_profit_share": None,
            "partial_fill_share": None,
        }
    low, high = bootstrap_by_token(returns, tokens, samples=samples, ci=ci, rng=rng)
    pnl = pd.Series([o.pnl_sol for _, o in trades]).groupby(tokens).sum()
    profit = pnl[pnl > 0]
    return row | {
        "win_rate": float((returns > 0).mean()),
        "mean_return": float(returns.mean()),
        "median_return": float(np.median(returns)),
        "mean_return_ci": [low, high],
        "total_pnl_sol": float(pnl.sum()) - failed * execution.network_fee,
        "exit_share": {r: float(np.mean([o.reason == r for _, o in trades])) for r in REASONS},
        "median_hold_s": float(np.median([o.hold_s for _, o in trades])),
        "top_token_profit_share": float(profit.max() / profit.sum()) if len(profit) else None,
        "partial_fill_share": float(np.mean([o.partial for _, o in trades])),
    }


def paired_difference(
    entries: Sequence[tuple[Any, Outcome]],
    reference: Sequence[tuple[Any, Outcome]],
    *,
    samples: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Two exit rules on the same tokens: per token traded by both, its mean net return minus the
    reference's, with a CI that draws tokens. Tokens are the unit, as an exit can change the number
    of trades on one token. ``mean_diff`` is ``None`` with no token in common."""

    def token_means(rows: Sequence[tuple[Any, Outcome]]) -> pd.Series:
        trades = [(m, o.net_return) for m, o in rows if o.status == FILLED]
        if not trades:
            return pd.Series(dtype=np.float64)
        frame = pd.DataFrame(trades, columns=["mint", "net"])
        return frame.groupby("mint")["net"].mean()

    joined = pd.concat([token_means(entries), token_means(reference)], axis=1, join="inner")
    if joined.empty:
        return {"n_tokens": 0, "mean_diff": None, "mean_diff_ci": [None, None]}
    diff = (joined.iloc[:, 0] - joined.iloc[:, 1]).to_numpy(np.float64)
    tokens = joined.index.to_numpy(dtype=object)
    low, high = bootstrap_by_token(diff, tokens, samples=samples, ci=ci, rng=rng)
    return {"n_tokens": len(diff), "mean_diff": float(diff.mean()), "mean_diff_ci": [low, high]}


def marginals(cells: Sequence[Mapping[str, Any]], *, min_trades: int) -> dict[str, Any]:
    """Mean of the cells' mean and median returns along each axis of the grid: a setting that only
    wins in one cell is noise (120 cells on the same trades), a stable zone shows up here."""

    def along(key: str) -> list[dict[str, Any]]:
        out = []
        for value in dict.fromkeys(c[key] for c in cells):
            kept = [
                c
                for c in cells
                if c[key] == value and c["n_trades"] >= min_trades and c["mean_return"] is not None
            ]
            out.append(
                {
                    "value": value,
                    "cells": len(kept),
                    "mean_return": (
                        float(np.mean([c["mean_return"] for c in kept])) if kept else None
                    ),
                    "median_return": (
                        float(np.mean([c["median_return"] for c in kept])) if kept else None
                    ),
                }
            )
        return out

    return {k: along(k) for k in ("stop_loss", "take_profit", "timeout_s")}


# --------------------------------------------------------------------------- selection


def window_outcomes(
    paths: Paths,
    windows: Windows,
    *,
    horizon_s: float,
    pump_threshold: float,
    max_return: float = DEFAULT_MAX_RETURN,
) -> dict[str, np.ndarray]:
    """Each window's outcome read off its path, labeled or not, with the labels' rules: ``touch``
    (a counted price reached the pump threshold), ``hold`` (last counted price of the horizon, the
    anchor's if none, capped) and ``flat`` (no row of the horizon moved from the anchor)."""
    n = len(windows)
    touch, flat = np.zeros(n, bool), np.zeros(n, bool)
    hold = np.zeros(n, np.float64)
    for i in range(n):
        row = int(windows.row[i])
        stop = paths.spans[windows.mint[i]][1]
        anchor = paths.price[row]
        end = int(np.searchsorted(paths.ts[row + 1 : stop], paths.ts[row] + horizon_s, "right"))
        prices = paths.price[row + 1 : row + 1 + end]
        counted = prices[paths.tradeable[row + 1 : row + 1 + end]]
        flat[i] = bool(np.all(prices == anchor))
        if len(counted):
            returns = counted / anchor - 1.0
            touch[i] = bool(returns.max() >= pump_threshold)
            hold[i] = min(float(returns[-1]), max_return)
    return {"touch": touch, "hold": hold, "flat": flat}


def one_per_mint(windows: Windows, paths: Paths, cooldown_s: float) -> np.ndarray:
    """Indices of the windows left when each mint is held for ``cooldown_s`` after an entry."""
    keep: list[int] = []
    busy_until: dict[Any, float] = {}
    for i in np.lexsort((windows.row,)):
        t = paths.ts[windows.row[i]]
        if t >= busy_until.get(windows.mint[i], -math.inf):
            busy_until[windows.mint[i]] = t + cooldown_s
            keep.append(int(i))
    return np.array(keep, np.int64)


def first_per_mint(windows: Windows) -> np.ndarray:
    seen: set[Any] = set()
    keep: list[int] = []
    for i in np.lexsort((windows.row,)):
        if windows.mint[i] not in seen:
            seen.add(windows.mint[i])
            keep.append(int(i))
    return np.array(keep, np.int64)


def selection_report(
    paths: Paths,
    windows: Windows,
    *,
    horizon_s: float,
    pump_threshold: float,
    samples: int,
    ci: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """The sweep's numbers, step by step down to the entries the bot could take."""
    outcomes = window_outcomes(paths, windows, horizon_s=horizon_s, pump_threshold=pump_threshold)

    def step(indices: np.ndarray) -> dict[str, Any]:
        n = int(len(indices))
        if n == 0:
            return {"n": 0, "n_tokens": 0}
        hold = outcomes["hold"][indices]
        tokens = windows.mint[indices]
        low, high = bootstrap_by_token(hold, tokens, samples=samples, ci=ci, rng=rng)
        return {
            "n": n,
            "n_tokens": len(set(tokens.tolist())),
            "precision": float(outcomes["touch"][indices].mean()),
            "win_rate": float((hold > 0).mean()),
            "mean_return": float(hold.mean()),
            "median_return": float(np.median(hold)),
            "mean_return_ci": [low, high],
            "labeled_share": float(windows.labeled[indices].mean()),
            "flat_share": float(outcomes["flat"][indices].mean()),
        }

    everything = np.arange(len(windows))
    return {
        "labeled_windows": step(everything[windows.labeled]),
        "all_windows": step(everything),
        "one_per_mint": step(one_per_mint(windows, paths, horizon_s)),
        "first_per_mint": step(first_per_mint(windows)),
    }


# --------------------------------------------------------------------------- run


def simulate_rules(
    paths: Paths,
    windows: Windows,
    *,
    pump_thresholds: Sequence[float],
    horizon_s: float,
    pump_threshold: float,
    grid: Sequence[ExitRule] | None = None,
    execution: Execution = BOT_EXECUTION,
    reference: ExitRule | None = None,
    min_trades: int = DEFAULT_MIN_TRADES,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci: float = DEFAULT_CI,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per entry rule ``pump_prob >= P``: the selection's steps and the grid of exits. With a
    ``reference``, every cell also gets its :func:`paired_difference` to that rule."""
    check_settings(pump_thresholds, min_trades, bootstrap_samples, ci)
    grid = default_grid() if grid is None else list(grid)
    longest = max([horizon_s, *(_longest_hold(rule) for rule in grid)])
    usable = uncensored(windows, paths, longest)
    rng = np.random.default_rng(seed)
    out = []
    for threshold in pump_thresholds:
        selected = usable.take(usable.pump_prob >= threshold)
        baseline = (
            None if reference is None else simulate_rule(paths, selected, reference, execution)
        )
        cells = []
        for rule in grid:
            entries = simulate_rule(paths, selected, rule, execution)
            cell = cell_metrics(
                rule, entries, execution=execution, samples=bootstrap_samples, ci=ci, rng=rng
            )
            if baseline is not None:
                cell["vs_reference"] = paired_difference(
                    entries, baseline, samples=bootstrap_samples, ci=ci, rng=rng
                )
            cells.append(cell)
        out.append(
            {
                "pump_threshold": threshold,
                "n_censored": len(windows.take(windows.pump_prob >= threshold)) - len(selected),
                "selection": selection_report(
                    paths,
                    selected,
                    horizon_s=horizon_s,
                    pump_threshold=pump_threshold,
                    samples=bootstrap_samples,
                    ci=ci,
                    rng=rng,
                ),
                "cells": cells,
                "marginals": marginals(cells, min_trades=min_trades),
            }
        )
    return out


def _longest_hold(rule: ExitRule) -> float:
    return rule.timeout_s if rule.rest_timeout_s is None else rule.rest_timeout_s


def check_settings(
    pump_thresholds: Sequence[float], min_trades: int, bootstrap_samples: int, ci: float
) -> None:
    if not pump_thresholds:
        raise ValueError("at least one pump threshold is required")
    for p in pump_thresholds:
        if not (_is_number(p) and 0.0 < p <= 1.0):
            raise ValueError(f"pump threshold must be in (0, 1], got {p!r}")
    if min_trades < 1:
        raise ValueError(f"min_trades must be at least 1, got {min_trades!r}")
    if bootstrap_samples < 1:
        raise ValueError(f"bootstrap_samples must be at least 1, got {bootstrap_samples!r}")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci!r}")


# --------------------------------------------------------------------------- report


def build_report(
    model_version: str,
    n_samples: int,
    horizon: str,
    rules: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return json_safe(
        {
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "n_samples": n_samples,
            "horizon": horizon,
            "settings": dict(settings),
            "rules": list(rules),
        }
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _render_step(name: str, s: Mapping[str, Any]) -> str:
    if not s["n"]:
        return f"  {name:<16} {0:>7} {0:>6}"
    low, high = s["mean_return_ci"]
    ci = "n/a" if low is None else f"[{low:+.3f},{high:+.3f}]"
    return (
        f"  {name:<16} {s['n']:>7} {s['n_tokens']:>6} {fmt(s['precision']):>7} "
        f"{fmt(s['win_rate']):>7} "
        f"{fmt(s['mean_return']):>8} {fmt(s['median_return']):>8} {ci:>17} "
        f"{_pct(s['labeled_share']):>7} {_pct(s['flat_share']):>5}"
    )


def _render_cell(c: Mapping[str, Any]) -> str:
    if not c["n_trades"]:
        return f"  {c['label']:<58} {0:>6}"
    low, high = c["mean_return_ci"]
    ci = "n/a" if low is None else f"[{low:+.3f},{high:+.3f}]"
    share = c["exit_share"]
    hold = "n/a" if c["median_hold_s"] is None else f"{c['median_hold_s']:.0f}"
    mark = " <- bot" if c["is_bot_default"] else ""
    versus = c.get("vs_reference")
    if versus and versus["mean_diff"] is not None:
        low_d, high_d = versus["mean_diff_ci"]
        band = "n/a" if low_d is None else f"[{low_d:+.3f},{high_d:+.3f}]"
        mark += f"  vs ref {versus['mean_diff']:+.3f} {band}"
    return (
        f"  {c['label']:<58} {c['n_trades']:>6} {c['n_tokens']:>6} {fmt(c['win_rate']):>7} "
        f"{fmt(c['mean_return']):>8} {fmt(c['median_return']):>8} {ci:>17} "
        f"{_pct(share['tp']):>4} {_pct(share['sl']):>4} {_pct(share['timeout']):>4} "
        f"{_pct(share['trail']):>5} {hold:>6} "
        f"{c['failed_entries']:>4}{mark}"
    )


def render(report: Mapping[str, Any]) -> str:
    settings = report["settings"]
    lines = [
        f"Exit-rule simulation — model {report['model_version']} "
        f"({report['n_samples']} held-out windows), horizon {report['horizon']}, "
        f"generated {report['generated_at']}",
        f"execution: {'bot defaults' if settings['execution']['realistic_fills'] else 'ideal'}, "
        f"{settings['ci'] * 100:g}% bootstrap CI by token. Many cells share the same trades: read "
        "the marginals and the stable zones, not the best cell.",
    ]
    for rule in report["rules"]:
        lines += [
            "",
            f"[pump >= {rule['pump_threshold']:g}] {rule['n_censored']} windows censored by the "
            "end of the read",
            "  from the sweep to the entries (held to the horizon, no exits, no costs)",
            f"  {'step':<16} {'n':>7} {'tokens':>6} {'prec':>7} {'win':>7} {'mean':>8} "
            f"{'median':>8} "
            f"{'mean CI':>17} {'labeled':>7} {'flat':>5}",
        ]
        lines += [_render_step(k, s) for k, s in rule["selection"].items()]
        lines += [
            "  exits (net of costs)",
            f"  {'cell':<58} {'trades':>6} {'tokens':>6} {'win':>7} {'mean':>8} {'median':>8} "
            f"{'mean CI':>17} {'tp':>4} {'sl':>4} {'time':>4} {'trail':>5} {'hold s':>6} "
            f"{'fail':>4}",
        ]
        lines += [_render_cell(c) for c in rule["cells"]]
        lines.append(f"  marginals (mean of the cells with >= {settings['min_trades']} trades)")
        for axis, values in rule["marginals"].items():
            parts = ", ".join(
                f"{v['value']}: {fmt(v['mean_return'])} / {fmt(v['median_return'])}" for v in values
            )
            lines.append(f"    {axis} (mean / median): {parts}")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], path: str | Path) -> tuple[Path, Path]:
    """The JSON report at ``path`` and its rendered table next to it (``.txt``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = path.with_suffix(".txt")
    table.write_text(render(report) + "\n", encoding="utf-8")
    return path, table


# --------------------------------------------------------------------------- CLI


def score_windows(predictor: Any, dataset: Any, device: str | torch.device) -> tuple[Any, Any]:
    """The calibrated predictions and the labels over ``dataset``, with the model on ``device``.

    ``Predictor`` loads on the CPU for serving, and :func:`score_logits` scores wherever the model
    lives, so a run has to move it first.
    """
    predictor.model.to(device)
    logits, truth = score_logits(predictor.model, dataset)
    return apply_calibration(logits, predictor.calibration), truth


def _utc(text: str) -> datetime:
    """An ISO-8601 time with its offset, as UTC: a naive time would depend on the machine."""
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO-8601 time: {text!r}") from exc
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f"give a UTC offset: {text!r}")
    return moment.astimezone(UTC)


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate the bot's exits on a checkpoint's held-out windows (read-only)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pt checkpoint")
    source.add_argument("--champion", action="store_true", help="the registry's champion")
    parser.add_argument("--out", help="write the JSON report here (+ the table as .txt)")
    parser.add_argument("--horizon", default=DEFAULT_HORIZON, help="the entry rule's horizon")
    parser.add_argument("--pump-thresholds", type=_floats, default=list(DEFAULT_PUMP_THRESHOLDS))
    parser.add_argument("--lookback-days", type=float, help="override data.lookback_days")
    parser.add_argument("--max-rows", type=int, help="override data.max_rows")
    parser.add_argument(
        "--until",
        type=_utc,
        help="end of the read, ISO-8601 with an offset (default: the newest row)",
    )
    parser.add_argument(
        "--window-stride-seconds",
        type=float,
        default=0.0,
        help="decisions per token: 0 = every snapshot, as the bot decides (default)",
    )
    parser.add_argument(
        "--ideal", action="store_true", help="fill at the mark, at once, for free (for comparison)"
    )
    parser.add_argument(
        "--grid",
        choices=("bot", "variants"),
        default="bot",
        help="bot = the #221 grid; variants = partial sales and trailing stops vs today's setting",
    )
    parser.add_argument("--min-trades", type=int, default=DEFAULT_MIN_TRADES)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--ci", type=float, default=DEFAULT_CI)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="torch device (default: cuda if available, else cpu)")
    args = parser.parse_args(argv)

    try:  # fail on bad settings before reading or scoring anything
        check_settings(args.pump_thresholds, args.min_trades, args.bootstrap_samples, args.ci)
        if args.window_stride_seconds < 0:
            raise ValueError(f"window stride must be >= 0, got {args.window_stride_seconds!r}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from None

    from .config import horizon_seconds, load_config
    from .db import DataSettings, make_engine, read_training_features
    from .evaluate import holdout_dataset
    from .inference import Predictor

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = make_engine()
    if args.champion:
        try:
            predictor = Predictor.from_champion(engine)
        except LookupError as exc:
            raise SystemExit(str(exc)) from None
    else:
        predictor = Predictor.from_checkpoint(args.checkpoint)
    horizons = list(predictor.horizons)
    if args.horizon not in horizons:
        raise SystemExit(f"horizon {args.horizon!r} is not one of the model's: {horizons}")
    h = horizons.index(args.horizon)

    config = load_config()
    config = {
        **config,
        "data": {**config.get("data", {}), "window_stride_seconds": args.window_stride_seconds},
    }
    settings_read = DataSettings.from_config(
        config, lookback_days=args.lookback_days, max_rows=args.max_rows
    )
    print("reading features")
    df, bounds = read_training_features(engine, settings_read, end=args.until)
    print(bounds.describe())
    print("building the held-out windows")
    dataset = holdout_dataset(df, config, predictor)
    print(f"scoring {len(dataset)} windows on {device}")
    predictions, truth = score_windows(predictor, dataset, device)
    print("simulating the exits")

    execution = IDEAL if args.ideal else Execution()
    floor = LiquidityFloor.from_config(config)
    paths = Paths.from_frame(df, floor=floor, execution=execution)
    samples, indices = dataset.samples, dataset.indices
    windows = Windows.align(
        paths,
        samples.mint[indices],
        samples.end_ts[indices],
        predictions["pump_prob"][:, h],
        predictions["expected_return"][:, h],
        truth["mask"][:, h] == 1,
    )
    settings = {
        "pump_thresholds": args.pump_thresholds,
        "window_stride_seconds": args.window_stride_seconds,
        "grid": args.grid,
        "read": bounds.as_dict(),
        "min_trades": args.min_trades,
        "bootstrap_samples": args.bootstrap_samples,
        "ci": args.ci,
        "seed": args.seed,
        "execution": asdict(execution),
    }
    variants = args.grid == "variants"
    rules = simulate_rules(
        paths,
        windows,
        pump_thresholds=args.pump_thresholds,
        horizon_s=horizon_seconds(args.horizon),
        pump_threshold=float(config["labeling"]["pump_threshold"][args.horizon]),
        grid=variant_grid() if variants else None,
        execution=execution,
        reference=REFERENCE_RULE if variants else None,
        min_trades=args.min_trades,
        bootstrap_samples=args.bootstrap_samples,
        ci=args.ci,
        seed=args.seed,
    )
    report = build_report(predictor.model_version, len(dataset), args.horizon, rules, settings)
    print(render(report))
    if args.out:
        json_path, table_path = write_report(report, args.out)
        print(f"\nreport written to {json_path} and {table_path}")


if __name__ == "__main__":
    main()
