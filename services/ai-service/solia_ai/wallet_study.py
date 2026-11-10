"""Winning wallets: who makes money on Pump.fun, how fast, does it persist, could a follower copy
it (#269). Descriptive; nothing is served.

**Data** (``extract``): the raw pumpapi replay archives (``YYYY/MM/DD/HH.jsonl.zst``), one frame per
on-chain event. Kept, per hour, as parquet:

* *trades*: SOL-quoted ``buy`` / ``sell`` frames — wallet (``txSigner``), token, side, token and SOL
  amounts, the pool's SOL reserve after the trade (the curve's virtual reserve when it has one), the
  price, venue, block, time and the transaction's priority fee. A ``create`` frame carries the
  creator's dev buy: kept as a buy unless the same transaction also has its own buy frame;
* *creates*: token, creator, block, time;
* *transfers*: token movements between wallets (SOL ones skipped).

**Positions** (:func:`positions`): per wallet x token over a period, its buys and sells. *Closed*
when it sold 99-101 % of the tokens it bought; its P&L is SOL out - SOL in - the priority fees it
paid. Positions touched by a token transfer (tokens that didn't come from or go to the pool) or
selling more than they bought are left out; the still-open ones are marked at the token's last
price, apart. Jito tips and other off-trade payments are not in the frames: bots' P&L is overstated.

**Persistence** (:func:`persistence`): wallets with >= :data:`MIN_POSITIONS` closed positions in
period A, ranked by A's P&L; A's top :data:`TOP_N` against the rest, on period B.

**Copy test** (:func:`copy_trades`, fixed before looking at period B): for each of A's top
wallets' positions in B, enter at the token's first trade >= ``latency`` (3 s / 10 s) after its
first buy, exit at the first trade >= :data:`EXIT_LATENCY` s after its first sell (none: the
token's last trade in the data), net of the bot's costs (:class:`solia_ai.survivor_study.Costs`:
1 SOL, pool impact, venue fee each side, network fees and rent). The same with
:data:`TOP_N` random eligible wallets as a control.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rug_labels import concat_categorical
from .survivor_study import Costs, bootstrap_ci, net_return, partition_of

logger = logging.getLogger(__name__)

WSOL = "So11111111111111111111111111111111111111112"
VENUES = {
    "pump": "pumpfun",
    "pump-amm": "pumpswap",
    "raydium-launchpad": "raydium-launchpad",
    "raydium-cpmm": "raydium",
    "raydium-clmm": "raydium",
    "raydium-v4": "raydium",
    "meteora-launchpad": "meteora-launchpad",
    "meteora-damm-v1": "meteora",
    "meteora-damm-v2": "meteora",
    "meteora-dlmm": "meteora",
    "orca-whirlpool": "orca",
    "bonk": "bonk",
    "sunrise": "sunrise",
}
MIN_POSITIONS = 10
TOP_N = 100
LATENCIES = (3.0, 10.0)
EXIT_LATENCY = 2.0
CLOSED_LOW, CLOSED_HIGH = 0.99, 1.01


def _utc(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.to_pydatetime().astimezone(UTC)


# Fixed before the analysis (#269): the two mirrored blocks of hours.
PERIOD_A = (_utc("2026-10-29"), _utc("2026-11-01"))
PERIOD_B = (_utc("2026-11-03"), _utc("2026-11-05"))


# --------------------------------------------------------------------------- extraction

_TRADE_COLUMNS = ("ts", "block", "mint", "wallet", "side", "tokens", "sol", "reserve", "price",
                  "venue", "priority_fee", "signature")  # fmt: skip


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def parse_lines(lines: Iterable[bytes]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(trades, creates, transfers)`` of raw pumpapi frames (module doc)."""
    import orjson

    trades: list[tuple] = []
    creates: list[tuple] = []
    transfers: list[tuple] = []
    for line in lines:
        is_trade = b'"action":"buy"' in line or b'"action":"sell"' in line
        is_create = b'"action":"create"' in line
        is_transfer = b'"action":"transfer"' in line and b'"isSolana":false' in line
        if not (is_trade or is_create or is_transfer):
            continue
        try:
            e = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        action, mint, wallet = e.get("action"), e.get("mint"), e.get("txSigner")
        ts, block = e.get("timestamp"), e.get("block")
        if action == "transfer":
            for t in e.get("transfers") or ():
                if not t.get("isSolana") and t.get("mint") and t.get("from") and t.get("to"):
                    transfers.append((ts, t["mint"], t["from"], t["to"], _num(t.get("amount"))))
            continue
        if not mint or not wallet or ts is None:
            continue
        if action == "create":
            creates.append((ts, block, mint, wallet))
        sol_quoted = e.get("quoteMint", WSOL) == WSOL
        if action == "create":
            tokens, side = _num(e.get("initialBuy")), "buy"
        else:
            tokens, side = _num(e.get("tokenAmount")), action
        sol = _num(e.get("quoteAmount", e.get("solAmount")))
        if not sol_quoted or not (tokens > 0) or not (sol > 0):
            continue
        reserve = _num(e.get("vQuoteInBondingCurve"))
        if math.isnan(reserve):
            reserve = _num(e.get("quoteInPool", e.get("solInPool")))
        trades.append((ts, block, mint, wallet, side, tokens, sol, reserve, _num(e.get("price")),
                       VENUES.get(str(e.get("pool")), str(e.get("pool"))),
                       _num(e.get("priorityFee")), e.get("signature") or "",
                       action == "create"))  # fmt: skip
    t = pd.DataFrame(trades, columns=[*_TRADE_COLUMNS, "from_create"])
    if len(t):
        # A create's dev buy also present as its own buy frame: keep the buy frame.
        real = t[~t["from_create"]]
        dup = t["from_create"] & t.set_index(["signature", "wallet", "mint"]).index.isin(
            real.set_index(["signature", "wallet", "mint"]).index
        )
        t = t[~dup]
    t = t.drop(columns=["from_create", "signature"]).astype(
        {"ts": "int64", "block": "Int64", "tokens": "float64", "sol": "float64"}
    )
    c = pd.DataFrame(creates, columns=["ts", "block", "mint", "creator"])
    x = pd.DataFrame(transfers, columns=["ts", "mint", "from", "to", "amount"])
    return t.reset_index(drop=True), c, x


def iter_lines(path: Path) -> Iterator[bytes]:
    import zstandard

    with open(path, "rb") as raw:
        reader = zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20)
        yield from io.BufferedReader(reader, buffer_size=1 << 22)


def hour_of(path: Path) -> datetime:
    """The hour of an archive ``.../YYYY/MM/DD/HH.jsonl.zst``."""
    hh = path.name.split(".")[0]
    dd, mm, yyyy = path.parent.name, path.parent.parent.name, path.parent.parent.parent.name
    return datetime(int(yyyy), int(mm), int(dd), int(hh), tzinfo=UTC)


def extract_hour(path: Path, out_dir: Path, parts: int = 1) -> str:
    """Parse one archive hour into ``out_dir/p<k>/{trades,creates,transfers}-YYYYMMDDHH.parquet``,
    split into ``parts`` token partitions (:func:`partition_of`): a position (wallet x token) and a
    copy never span two partitions."""
    stamp = f"{hour_of(path):%Y%m%d%H}"
    done = out_dir / f"done-{stamp}"
    if done.exists():
        return f"{stamp} already extracted"
    frames = dict(zip(("trades", "creates", "transfers"), parse_lines(iter_lines(path)),
                      strict=True))  # fmt: skip
    for k in range(parts):
        (out_dir / f"p{k}").mkdir(parents=True, exist_ok=True)
    for kind, frame in frames.items():
        part = partition_of(frame["mint"], parts) if len(frame) else np.zeros(0, np.int64)
        for k in range(parts):
            frame[part == k].to_parquet(out_dir / f"p{k}" / f"{kind}-{stamp}.parquet", index=False)
    done.touch()  # last: marks the hour done
    sizes = ", ".join(f"{len(f)} {kind}" for kind, f in frames.items())
    return f"{stamp}: {sizes}"


def extract(archives: Path, out_dir: Path, jobs: int = 1, parts: int = 1) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(archives.glob("*/*/*/*.jsonl.zst"))
    if not paths:
        raise ValueError(f"no .jsonl.zst archives under {archives}")
    n = len(paths)
    if jobs <= 1:
        return [extract_hour(p, out_dir, parts) for p in paths]
    with ProcessPoolExecutor(jobs) as pool:
        out = []
        for msg in pool.map(extract_hour, paths, [out_dir] * n, [parts] * n):
            logger.info(msg)
            out.append(msg)
        return out


def partitions(data: Path) -> list[Path]:
    """The token partition directories of an extraction."""
    return sorted(p for p in data.glob("p*") if p.is_dir())


def load_period(part_dir: Path, start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """One partition's extracted hours of ``[start, end)``: ``trades`` (by time), ``creates``,
    ``transfers``; wallet / token columns categorical."""
    out = {}
    for kind in ("trades", "creates", "transfers"):
        files = [
            f
            for f in sorted(part_dir.glob(f"{kind}-*.parquet"))
            if start
            <= datetime.strptime(f.stem.split("-")[1], "%Y%m%d%H").replace(tzinfo=UTC)
            < end
        ]
        frames = [pd.read_parquet(f) for f in files]
        cats = [c for c in ("wallet", "mint", "venue", "side", "creator", "from", "to")
                if frames and c in frames[0].columns]  # fmt: skip
        for f in frames:
            for col in cats:
                f[col] = f[col].astype(object).astype("category")
        out[kind] = concat_categorical(frames, cats) if frames else pd.DataFrame()
    if len(out["trades"]):
        out["trades"] = out["trades"].sort_values("ts", kind="stable").reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- positions


def positions(trades: pd.DataFrame, creates: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Per wallet x token (module doc): ``first_buy_ts`` / ``block``, ``first_sell_ts``,
    ``last_sell_ts``, ``buy_sol``, ``sell_sol``, ``buy_tokens``, ``sell_tokens``, ``fees``,
    ``status`` (closed / open / transferred / unbacked), ``pnl`` (closed: realized; open: marked at
    the token's last price), ``hold_s``, ``entry_delay_blocks`` / ``_s`` (from the create, when
    seen), ``is_creator``."""
    t = trades
    buy = t["side"] == "buy"
    g = t.assign(
        buy_sol=np.where(buy, t["sol"], 0.0),
        sell_sol=np.where(buy, 0.0, t["sol"]),
        buy_tokens=np.where(buy, t["tokens"], 0.0),
        sell_tokens=np.where(buy, 0.0, t["tokens"]),
        buy_ts=np.where(buy, t["ts"], np.iinfo(np.int64).max),
        sell_ts=np.where(buy, np.iinfo(np.int64).max, t["ts"]),
        sell_ts_max=np.where(buy, np.iinfo(np.int64).min, t["ts"]),
        buy_block=np.where(buy, t["block"].astype("float64"), np.inf),
        fee=t["priority_fee"].fillna(0.0),
    ).groupby(["wallet", "mint"], sort=False, observed=True)
    p = g.agg(
        buy_sol=("buy_sol", "sum"),
        sell_sol=("sell_sol", "sum"),
        buy_tokens=("buy_tokens", "sum"),
        sell_tokens=("sell_tokens", "sum"),
        first_buy_ts=("buy_ts", "min"),
        first_sell_ts=("sell_ts", "min"),
        last_sell_ts=("sell_ts_max", "max"),
        first_buy_block=("buy_block", "min"),
        fees=("fee", "sum"),
        trades=("sol", "size"),
    ).reset_index()
    big, small = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    p["first_sell_ts"] = p["first_sell_ts"].where(p["first_sell_ts"] != big)
    p["last_sell_ts"] = p["last_sell_ts"].where(p["last_sell_ts"] != small)
    p["first_buy_ts"] = p["first_buy_ts"].where(p["first_buy_ts"] != big)
    p["first_buy_block"] = p["first_buy_block"].where(np.isfinite(p["first_buy_block"]))

    moved = set()
    if len(transfers):
        moved = set(zip(transfers["from"], transfers["mint"], strict=True)) | set(
            zip(transfers["to"], transfers["mint"], strict=True)
        )
    key = list(zip(p["wallet"].astype(object), p["mint"].astype(object), strict=True))
    transferred = np.array([k in moved for k in key], bool)
    ratio = np.where(p["buy_tokens"] > 0, p["sell_tokens"] / p["buy_tokens"], np.inf)
    status = np.select(
        [
            transferred,
            (p["buy_tokens"] <= 0) | (ratio > CLOSED_HIGH),
            ratio >= CLOSED_LOW,
        ],
        ["transferred", "unbacked", "closed"],
        default="open",
    )
    p["status"] = status
    last_price = t.groupby("mint", observed=True)["price"].last()
    remaining = (p["buy_tokens"] - p["sell_tokens"]).clip(lower=0)
    mark = remaining * p["mint"].map(last_price).astype(float).fillna(0.0)
    realized = p["sell_sol"] - p["buy_sol"] - p["fees"]
    p["pnl"] = np.where(status == "closed", realized, np.where(status == "open",
                        realized + mark, np.nan))  # fmt: skip
    p["hold_s"] = (p["last_sell_ts"] - p["first_buy_ts"]) / 1000.0

    if len(creates):
        c = creates.drop_duplicates("mint").set_index("mint")
        p["is_creator"] = (
            p["wallet"].astype(object).to_numpy()
            == p["mint"].map(c["creator"]).astype(object).to_numpy()
        )
        p["entry_delay_blocks"] = p["first_buy_block"] - p["mint"].map(c["block"]).astype(float)
        p["entry_delay_s"] = (p["first_buy_ts"] - p["mint"].map(c["ts"]).astype(float)) / 1000.0
    else:
        p["is_creator"] = False
        p["entry_delay_blocks"] = np.nan
        p["entry_delay_s"] = np.nan
    return p


def wallet_profiles(pos: pd.DataFrame) -> pd.DataFrame:
    """Per wallet over its closed positions: count, total / mean P&L, win rate, median hold, median
    entry delay, share entered in the create's block, share of its own tokens; open positions'
    count and marked P&L apart."""
    closed = pos[pos["status"] == "closed"]
    g = closed.groupby("wallet", observed=True)
    prof = pd.DataFrame(
        {
            "positions": g.size(),
            "pnl": g["pnl"].sum(),
            "pnl_mean": g["pnl"].mean(),
            "win_rate": g["pnl"].apply(lambda x: float((x > 0).mean())),
            "sol_in": g["buy_sol"].sum(),
            "hold_median_s": g["hold_s"].median(),
            "delay_median_s": g["entry_delay_s"].median(),
            # among positions whose token create is in the data (others: created earlier)
            "create_seen": g["entry_delay_blocks"].apply(lambda x: float(x.notna().mean())),
            "same_block": g["entry_delay_blocks"].apply(
                lambda x: float((x.dropna() == 0).mean()) if x.notna().any() else np.nan
            ),
            "creator_share": g["is_creator"].mean(),
        }
    )
    opened = pos[pos["status"] == "open"].groupby("wallet", observed=True)["pnl"]
    prof["open_positions"] = opened.size().reindex(prof.index).fillna(0).astype(int)
    prof["open_pnl"] = opened.sum().reindex(prof.index).fillna(0.0)
    return prof


# --------------------------------------------------------------------------- persistence


def top_wallets(
    prof_a: pd.DataFrame, n: int | None = None, min_positions: int | None = None
) -> pd.Index:
    """A's top ``n`` (:data:`TOP_N`) wallets by total P&L among those with >= ``min_positions``
    (:data:`MIN_POSITIONS`) closed ones."""
    n = TOP_N if n is None else n
    min_positions = MIN_POSITIONS if min_positions is None else min_positions
    eligible = prof_a[prof_a["positions"] >= min_positions]
    return eligible.sort_values("pnl", ascending=False, kind="stable").index[:n]


def group_stats(prof_b: pd.DataFrame, wallets: pd.Index, samples: int = 1000) -> dict[str, Any]:
    """How ``wallets`` did in period B: how many traded, their summed and per-position P&L, the
    share of them in profit, the pooled win rate, and a bootstrap CI (by wallet) of the mean P&L
    per position."""
    active = prof_b.reindex(wallets).dropna(subset=["positions"])
    out: dict[str, Any] = {"wallets": int(len(wallets)), "active_in_b": int(len(active))}
    if not len(active):
        return out
    per_pos = active["pnl"].to_numpy() / active["positions"].to_numpy()
    out.update(pnl_total=float(active["pnl"].sum()), positions=int(
        active["positions"].sum()
    ), pnl_per_position=float(
        active["pnl"].sum() / active["positions"].sum()
    ), wallet_mean_per_position=float(
        per_pos.mean()
    ), wallet_mean_per_position_ci=bootstrap_ci(
        per_pos,
        lambda x: x.mean(axis=1),
        samples=samples),  # fmt: skip
        share_profitable=float((active["pnl"] > 0).mean()),
        win_rate=float(
            (active["win_rate"] * active["positions"]).sum() / active["positions"].sum()
        ),
    )
    return out


def persistence(
    prof_a: pd.DataFrame, prof_b: pd.DataFrame, *, seed: int = 0, samples: int = 1000
) -> dict[str, Any]:
    """A's top wallets vs the rest of A's eligible wallets, on period B; and the rank
    correlation of the per-position P&L for wallets eligible in both periods."""
    from scipy.stats import spearmanr

    eligible = prof_a[prof_a["positions"] >= MIN_POSITIONS]
    top = top_wallets(prof_a)
    rest = eligible.index.difference(top)
    both = eligible.index.intersection(prof_b[prof_b["positions"] >= 5].index)
    rho = None
    if len(both) >= 3:
        rho = float(spearmanr(eligible.loc[both, "pnl_mean"], prof_b.loc[both, "pnl_mean"])[0])
    return {
        "eligible_in_a": int(len(eligible)),
        "top_in_a": _profile_summary(prof_a.loc[top]),
        "top": group_stats(prof_b, top, samples),
        "rest": group_stats(prof_b, rest, samples),
        "spearman_pnl_mean_a_b": rho,
        "wallets_in_both": int(len(both)),
    }


def _profile_summary(prof: pd.DataFrame) -> dict[str, Any]:
    if not len(prof):
        return {"wallets": 0}
    return {
        "wallets": int(len(prof)),
        "pnl_total": float(prof["pnl"].sum()),
        "positions_median": float(prof["positions"].median()),
        "win_rate_median": float(prof["win_rate"].median()),
        "hold_median_s": float(prof["hold_median_s"].median()),
        "delay_median_s": float(prof["delay_median_s"].median()),
        "same_block_median": float(prof["same_block"].median()),
        "create_seen_median": float(prof["create_seen"].median()),
        "creator_share_median": float(prof["creator_share"].median()),
    }


def profile_by_pnl(prof: pd.DataFrame, min_positions: int = MIN_POSITIONS) -> dict[str, Any]:
    """Who wins: eligible wallets in P&L buckets (top 1 %, top 10 %, middle, bottom 10 %), with
    their speed (entry delay, same-block share), hold time and creator share."""
    eligible = prof[prof["positions"] >= min_positions].sort_values("pnl", ascending=False)
    n = len(eligible)
    if not n:
        return {}
    cuts = {
        "top1%": eligible.iloc[: max(1, n // 100)],
        "top10%": eligible.iloc[: max(1, n // 10)],
        "middle": eligible.iloc[n // 10 : n - n // 10],
        "bottom10%": eligible.iloc[n - max(1, n // 10) :],
    }
    return {name: _profile_summary(rows) for name, rows in cuts.items()}


# --------------------------------------------------------------------------- copy test


def copy_trades(
    trades: pd.DataFrame,
    leaders: pd.Index,
    latency_s: float,
    costs: Costs,
    exit_latency_s: float = EXIT_LATENCY,
) -> pd.DataFrame:
    """One copy per leader x token of period ``trades`` (module doc): ``wallet``, ``mint``,
    ``entry_ts`` / ``exit_ts`` (ms), ``exit_kind`` (sell / end), ``net``. A token with no trade
    ``latency_s`` after the leader's buy is not copied."""
    t = trades
    mints = t["mint"].astype(object).to_numpy()
    ts = t["ts"].to_numpy(np.int64)
    order = np.lexsort((ts, mints))  # trades by token, then time
    m_sorted, ts_sorted = mints[order], ts[order]
    codes, uniq = pd.factorize(m_sorted)
    starts = np.searchsorted(codes, np.arange(len(uniq)))
    ends = np.r_[starts[1:], len(codes)]
    code_of = dict(zip(uniq, range(len(uniq)), strict=True))

    mine = t[t["wallet"].isin(leaders)].assign(
        wallet=lambda d: d["wallet"].astype(object), mint=lambda d: d["mint"].astype(object)
    )
    buys = mine[mine["side"] == "buy"].groupby(["wallet", "mint"])["ts"].min()
    sells = mine[mine["side"] == "sell"][["wallet", "mint", "ts"]].merge(
        buys.rename("t_buy").reset_index(), on=["wallet", "mint"]
    )
    first_sell = sells[sells["ts"] >= sells["t_buy"]].groupby(["wallet", "mint"])["ts"].min()
    rows = []
    price = t["price"].to_numpy(np.float64)[order]
    reserve = t["reserve"].to_numpy(np.float64)[order]
    venue = t["venue"].astype(object).to_numpy()[order]
    for (wallet, mint), t_buy in buys.items():
        c = code_of[mint]
        lo, hi = starts[c], ends[c]
        i = lo + np.searchsorted(ts_sorted[lo:hi], t_buy + latency_s * 1000, side="left")
        if i >= hi:
            continue
        sold = first_sell.get((wallet, mint))
        if sold is not None:
            target = max(int(sold) + exit_latency_s * 1000, ts_sorted[i])
            j = lo + np.searchsorted(ts_sorted[lo:hi], target, side="left")
            kind = "sell"
            if j >= hi:
                j, kind = hi - 1, "end"
        else:
            j, kind = hi - 1, "end"
        rows.append((wallet, mint, ts_sorted[i], ts_sorted[j], kind, i, j))
    out = pd.DataFrame(
        rows, columns=["wallet", "mint", "entry_ts", "exit_ts", "exit_kind", "i", "j"]
    )
    if not len(out):
        return out.assign(net=pd.Series(dtype=float)).drop(columns=["i", "j"])
    i, j = out["i"].to_numpy(), out["j"].to_numpy()
    out["net"] = net_return(
        price[i], reserve[i], costs.fee(venue[i]), price[j], reserve[j], costs.fee(venue[j]), costs
    )
    return out.drop(columns=["i", "j"])


def copy_summary(copies: pd.DataFrame, samples: int = 1000) -> dict[str, Any]:
    if not len(copies):
        return {"copies": 0}
    net = copies["net"].to_numpy(np.float64)
    return {
        "copies": int(len(net)),
        "tokens": int(copies["mint"].nunique()),
        "mean": float(net.mean()),
        "mean_ci": bootstrap_ci(net, lambda x: x.mean(axis=1), samples=samples),
        "median": float(np.median(net)),
        "win_rate": float((net > 0).mean()),
        "sol_total_at_1sol": float(net.sum()),
        "exited_on_leader_sell": float((copies["exit_kind"] == "sell").mean()),
    }


# --------------------------------------------------------------------------- the study


_POSITION_COLUMNS = ("wallet", "mint", "status", "pnl", "buy_sol", "hold_s", "entry_delay_s",
                     "entry_delay_blocks", "is_creator")  # fmt: skip


def _period_meta(trades: int, tokens: int, pos: pd.DataFrame) -> dict[str, Any]:
    return {
        "trades": int(trades),
        "wallets": int(pos["wallet"].nunique()) if len(pos) else 0,
        "tokens": int(tokens),
        "positions": {str(k): int(v) for k, v in pos["status"].value_counts().items()},
        "closed_pnl_total": float(pos.loc[pos["status"] == "closed", "pnl"].sum()),
    }


def _slim(pos: pd.DataFrame) -> pd.DataFrame:
    out = pos[list(_POSITION_COLUMNS)].copy()
    for col in ("wallet", "mint", "status"):
        out[col] = out[col].astype(object)
    return out


def study(
    period_a: Mapping[str, pd.DataFrame],
    period_b: Mapping[str, pd.DataFrame],
    *,
    costs: Costs | None = None,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The whole report from both periods' frames held at once (one partition)."""
    costs = costs or Costs(virtual_sol={})
    pos_a = positions(period_a["trades"], period_a["creates"], period_a["transfers"])
    pos_b = positions(period_b["trades"], period_b["creates"], period_b["transfers"])
    meta = {
        "A": _period_meta(len(period_a["trades"]), period_a["trades"]["mint"].nunique(), pos_a),
        "B": _period_meta(len(period_b["trades"]), period_b["trades"]["mint"].nunique(), pos_b),
    }

    def run(combos: Sequence[tuple[str, pd.Index, float]]) -> dict[str, pd.DataFrame]:
        return {name: copy_trades(period_b["trades"], who, lat, costs) for name, who, lat in combos}

    return analyze(_slim(pos_a), _slim(pos_b), meta, run, samples=samples, seed=seed)


def study_partitions(
    data: Path,
    a: tuple[datetime, datetime],
    b: tuple[datetime, datetime],
    *,
    costs: Costs | None = None,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """:func:`study` over the token partitions of an extraction: positions per partition (a
    position never spans two), wallet profiles over all of them, copies per partition."""
    costs = costs or Costs(virtual_sol={})
    parts = partitions(data)
    if not parts:
        raise ValueError(f"no token partitions under {data}")
    pos = {"A": [], "B": []}
    counts = {"A": [0, 0], "B": [0, 0]}
    for part in parts:
        for name, window in (("A", a), ("B", b)):
            frames = load_period(part, *window)
            if not len(frames["trades"]):
                continue
            pos[name].append(_slim(positions(frames["trades"], frames["creates"],
                                             frames["transfers"])))  # fmt: skip
            counts[name][0] += len(frames["trades"])
            counts[name][1] += frames["trades"]["mint"].nunique()
        logger.info("positions of %s done", part.name)
    if not pos["A"] or not pos["B"]:
        raise ValueError("no extracted trades in period A or B")
    pos_a = pd.concat(pos["A"], ignore_index=True)
    pos_b = pd.concat(pos["B"], ignore_index=True)
    meta = {n: _period_meta(*counts[n], p) for n, p in (("A", pos_a), ("B", pos_b))}

    def run(combos: Sequence[tuple[str, pd.Index, float]]) -> dict[str, pd.DataFrame]:
        out: dict[str, list[pd.DataFrame]] = {name: [] for name, _, _ in combos}
        for part in parts:
            trades = load_period(part, *b)["trades"]
            if not len(trades):
                continue
            for name, who, lat in combos:
                out[name].append(copy_trades(trades, who, lat, costs))
            logger.info("copies of %s done", part.name)
        return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in out.items()}

    return analyze(pos_a, pos_b, meta, run, samples=samples, seed=seed)


def analyze(
    pos_a: pd.DataFrame,
    pos_b: pd.DataFrame,
    meta: Mapping[str, Any],
    run_copies: Any,
    *,
    samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """The report from both periods' positions; ``run_copies(combos)`` runs the copy test of
    ``(name, leaders, latency)`` combos on period B."""
    prof_a, prof_b = wallet_profiles(pos_a), wallet_profiles(pos_b)
    report: dict[str, Any] = {
        "periods": dict(meta),
        "who_wins_a": profile_by_pnl(prof_a),
        "persistence": persistence(prof_a, prof_b, seed=seed, samples=samples),
    }
    eligible = prof_a[prof_a["positions"] >= MIN_POSITIONS].index
    top = top_wallets(prof_a)
    rest = eligible.difference(top)
    rng = np.random.default_rng(seed)
    control = (
        pd.Index(rng.choice(rest.to_numpy(object), size=min(TOP_N, len(rest)), replace=False))
        if len(rest)
        else rest
    )
    combos = [
        (f"{lat:g}s/{group}", who, lat)
        for lat in LATENCIES
        for group, who in (("top", top), ("random", control))
    ]
    copies = run_copies(combos)
    report["copy"] = {}
    for name, _, _ in combos:
        latency, group = name.split("/")
        report["copy"].setdefault(latency, {})[group] = copy_summary(copies[name], samples)
    report["top_wallets_a"] = [
        {"wallet": str(w), **{k: _jsonable(v) for k, v in prof_a.loc[w].items()},
         "b_pnl": _jsonable(prof_b["pnl"].get(w)),
         "b_positions": _jsonable(prof_b["positions"].get(w))}
        for w in top[:20]
    ]  # fmt: skip
    return report


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, np.integer | np.bool_):
        return v.item()
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if math.isfinite(f) else None


def render(report: Mapping[str, Any]) -> str:
    lines = ["Winning wallets (#269): per-trade replay archives, period A vs period B", ""]
    for name, p in report["periods"].items():
        lines.append(
            f"period {name}: {p['trades']:,} trades, {p['wallets']:,} wallets, {p['tokens']:,} "
            f"tokens; positions {p['positions']}; closed P&L {p['closed_pnl_total']:+,.1f} SOL"
        )
    lines += ["", "Who wins in A (wallets with >= 10 closed positions, by total P&L):",
              f"{'bucket':<10} {'wallets':>7} {'P&L SOL':>10} {'pos':>5} {'win':>5} "
              f"{'hold':>8} {'delay':>8} {'seen':>5} {'same blk':>8} {'creator':>8}"]  # fmt: skip
    for name, s in report["who_wins_a"].items():
        lines.append(
            f"{name:<10} {s['wallets']:>7} {s['pnl_total']:>+10.1f} {s['positions_median']:>5.0f} "
            f"{_p(s['win_rate_median']):>5} {_dur(s['hold_median_s']):>8} "
            f"{_dur(s['delay_median_s']):>8} {_p(s['create_seen_median']):>5} "
            f"{_p(s['same_block_median']):>8} "
            f"{_p(s['creator_share_median']):>8}"
        )
    per = report["persistence"]
    lines += ["", f"Persistence: {per['eligible_in_a']} eligible wallets in A; on B:"]
    for name in ("top", "rest"):
        s = per[name]
        if not s.get("active_in_b"):
            lines.append(f"  {name}: {s['active_in_b']} of {s['wallets']} active in B")
            continue
        ci = s["wallet_mean_per_position_ci"]
        lines.append(
            f"  {name:<5} {s['active_in_b']:>5}/{s['wallets']:<5} active, P&L "
            f"{s['pnl_total']:+.1f} "
            f"SOL over {s['positions']} positions ({s['pnl_per_position']:+.4f}/pos), wallet mean "
            f"{s['wallet_mean_per_position']:+.4f}/pos [{_ci(ci)}], profitable "
            f"{_p(s['share_profitable'])}, win rate {_p(s['win_rate'])}"
        )
    rho = per["spearman_pnl_mean_a_b"]
    lines.append(
        f"  Spearman(A, B) of P&L per position, {per['wallets_in_both']} wallets: "
        f"{'n/a' if rho is None else f'{rho:+.3f}'}"
    )
    lines += ["", "Copy test on B (1 SOL, bot costs), A's top wallets vs random eligible ones:"]
    for latency, groups in report["copy"].items():
        for name, s in groups.items():
            if not s.get("copies"):
                lines.append(f"  {latency:>4} {name:<6} no copy")
                continue
            lines.append(
                f"  {latency:>4} {name:<6} {s['copies']:>6} copies, mean {s['mean']:+.1%} "
                f"[{_ci(s['mean_ci'], pct=True)}], median {s['median']:+.1%}, win "
                f"{_p(s['win_rate'])}, total {s['sol_total_at_1sol']:+.1f} SOL, exit on leader "
                f"sell {_p(s['exited_on_leader_sell'])}"
            )
    return "\n".join(lines) + "\n"


def _p(v: Any) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def _ci(ci: Sequence[float] | None, pct: bool = False) -> str:
    if ci is None:
        return "n/a"
    return f"{ci[0]:+.1%}, {ci[1]:+.1%}" if pct else f"{ci[0]:+.4f}, {ci[1]:+.4f}"


def _dur(seconds: Any) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "n/a"
    s = float(seconds)
    return f"{s:.0f}s" if s < 120 else f"{s / 60:.0f}m" if s < 7200 else f"{s / 3600:.1f}h"


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
    x = sub.add_parser("extract", help="parse the replay archives into per-hour parquet")
    x.add_argument("--archives", required=True, type=Path, help="mirror root (YYYY/MM/DD/HH)")
    x.add_argument("--out", required=True, type=Path)
    x.add_argument("--jobs", type=int, default=1)
    x.add_argument("--parts", type=int, default=8, help="token partitions (bounds the memory)")
    a = sub.add_parser("analyze", help="positions, persistence and copy test")
    a.add_argument("--data", required=True, type=Path)
    a.add_argument("--report", type=Path, help="write the JSON report (and a .txt summary)")
    a.add_argument("--a", type=_window, default=PERIOD_A, help="period A START/END (UTC)")
    a.add_argument("--b", type=_window, default=PERIOD_B, help="period B START/END (UTC)")
    a.add_argument("--samples", type=int, default=1000)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.command == "extract":
        if args.jobs < 1 or args.parts < 1:
            raise SystemExit(f"--jobs and --parts must be >= 1, got {args.jobs}, {args.parts}")
        try:
            extract(args.archives, args.out, args.jobs, args.parts)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return 0
    if not args.a[1] <= args.b[0]:
        raise SystemExit("period A must end before period B starts")
    try:
        report = study_partitions(args.data, args.a, args.b, samples=args.samples)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    report["windows"] = {"A": [t.isoformat() for t in args.a], "B": [t.isoformat() for t in args.b]}
    text = render(report)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=_jsonable), encoding="utf-8")
        args.report.with_suffix(".txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
