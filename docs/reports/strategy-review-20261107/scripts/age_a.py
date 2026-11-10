"""Direction A with a minimum token age, on data no test has used. Read-only.

Fixed on 2026-11-08 ~04:40 UTC, before reading any row after 2026-11-07 03:10 UTC.

Why: on 11-05 21:08 .. 11-06 02:11, one-block dumps hit 67 % of entries in tokens under 5 min old, 5 % past 1 h (19 trades).
Population at the decision row: LP burned >= 90 %, >= 20 traders, >= 5 SOL, 0 < price < 0.001 SOL, token_age_sec >= 3600.
Entry: breakout (price above the high of the previous 30 marks, at least 6), one position per token at a time, re-entry allowed.
Exit: TP +200 %, stop vol3, 900 s, the simulator's bot fills. This one cell decides; nothing is chosen on this data.
Window: decision rows from 2026-11-07 03:10 UTC to the end given by --end (whole hours, after the 15:48-21:41 UTC gap is
replayed and its density checked); trades must close before --end.
Baseline: one random population row per token, 20 draws, same exit.

Reading: (a) at least 100 trades, (b) a 95 % CI by token above 0, (c) its lower bound above the baseline's mean. All three:
passes, then a paper session before anything else. Fewer than 100 trades: no verdict, rerun later on more data, rule unchanged.
Otherwise: fails. Also printed, not part of the verdict: the same rule under 1 h of age, and at stop 30 % / 1 h.
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
from sqlalchemy import text

from solia_ai import exit_sim as ex
from solia_ai.config import load_config
from solia_ai.db import make_engine
from solia_ai.features import DB_FEATURE_COLUMNS
from solia_ai.model_value import breakout_signal

parser = argparse.ArgumentParser()
parser.add_argument("--end", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
START = pd.Timestamp("2026-11-07 03:10+00").timestamp()
END = pd.Timestamp(args.end).timestamp()
started = time.time()
say = lambda *a: print(*a, round(time.time() - started), "s", flush=True)
engine = make_engine()
config = load_config()

# Population rows of both age groups (the young one only for the side report).
parts, t = [], START
while t < END:
    parts.append(pd.read_sql(text(
        """select mint, extract(epoch from ts) t from features where ts >= to_timestamp(:a) and ts < to_timestamp(:b)
        and burned_liquidity >= 90 and unique_traders >= 20 and liquidity >= 5 and price > 0 and price < 0.001"""),
        engine, params={"a": t, "b": min(t + 21600, END)}))
    t += 21600
pop = pd.concat(parts, ignore_index=True)
say("population rows", len(pop), "tokens", pop["mint"].nunique())

BEFORE, AFTER = 420.0, 3700.0  # breakout needs 30 marks before; the longest hold is 1 h
intervals = []
for mint, g in pop.sort_values(["mint", "t"]).groupby("mint", sort=False):
    tt = g["t"].to_numpy()
    breaks = np.flatnonzero(tt[1:] - AFTER > tt[:-1] + BEFORE) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(tt)], strict=True):
        intervals.append((mint, float(tt[a] - BEFORE), float(tt[b - 1] + AFTER)))
columns = ", ".join(f"f.{c}" for c in dict.fromkeys(["mint", "ts", "venue", "token_age_sec", *DB_FEATURE_COLUMNS]))
frames = []
for i in range(0, len(intervals), 1000):
    part = intervals[i : i + 1000]
    chunk = pd.read_sql(text(
        f"""select {columns} from unnest(cast(:m as text[]), cast(:a as float8[]), cast(:b as float8[])) w(mint, a, b)
        cross join lateral (select * from features x where x.mint = w.mint and x.ts >= to_timestamp(w.a) and x.ts <= to_timestamp(w.b)) f"""),
        engine, params={"m": [x[0] for x in part], "a": [x[1] for x in part], "b": [x[2] for x in part]})
    for c in chunk.columns:
        if c not in ("mint", "ts", "venue"):
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(np.float64 if c in ("price", "liquidity", "volatility") else np.float32)
    frames.append(chunk)
df = pd.concat(frames, ignore_index=True).drop_duplicates(["mint", "ts"], ignore_index=True)
del frames
say("rows read", len(df))
execution = ex.Execution()
paths = ex.Paths.from_frame(df, floor=ex.LiquidityFloor.from_config(config), execution=execution)
order = ex.path_rows(df)[0]
col = lambda n: pd.to_numeric(df[n], errors="coerce").to_numpy(np.float64)[order]
burned, traders, liquidity, age = col("burned_liquidity"), col("unique_traders"), col("liquidity"), col("token_age_sec")
del df
ts = paths.ts
with np.errstate(invalid="ignore"):
    base = (burned >= 90) & (traders >= 20) & (liquidity >= 5) & (paths.price > 0) & (paths.price < 0.001) & (ts >= START) & (ts < END)
    old, young = base & (age >= 3600), base & (age < 3600)
    breakout = breakout_signal(paths) > 0
mint_of = np.empty(len(ts), dtype=object)
for mint, (a, b) in paths.spans.items():
    mint_of[a:b] = mint

RULES = {"vol3/900s": ex.ExitRule(take_profit=2.0, stop_loss="vol3", timeout_s=900),
         "30%/1h": ex.ExitRule(take_profit=2.0, stop_loss=0.3, timeout_s=3600)}
rng = np.random.default_rng(0)


def simulate(rows, rule):
    usable = rows[ts[rows] + rule.timeout_s <= END]
    out, busy = [], {}
    for row in usable[np.argsort(ts[usable], kind="stable")]:
        row = int(row)
        m = mint_of[row]
        if ts[row] < busy.get(m, -math.inf):
            continue
        o = ex.simulate_position(paths, row, paths.spans[m][1], 1.0, rule, execution)
        busy[m] = o.free_at
        out.append((m, o))
    return out


def summary(entries):
    trades = [(m, o) for m, o in entries if o.status == ex.FILLED]
    if not trades:
        return {"n": 0, "tokens": 0}
    r = np.array([o.net_return for _, o in trades])
    tokens = np.array([m for m, _ in trades], dtype=object)
    low, high = ex.bootstrap_by_token(r, tokens, samples=2000, ci=0.95, rng=rng)
    return {"n": len(r), "tokens": len(set(tokens.tolist())), "mean": float(r.mean()), "median": float(np.median(r)), "ci": [low, high],
            "win": float((r > 0).mean()), "total_loss": float((r <= -0.99).mean()), "tp": float(np.mean([o.reason == "tp" and o.net_return > 0 for _, o in trades]))}


def random_mean(mask, rule):
    by = {}
    for row in np.flatnonzero(mask):
        by.setdefault(mint_of[row], []).append(int(row))
    means = []
    for _ in range(20):
        pick = np.array([v[rng.integers(len(v))] for v in by.values()], np.int64)
        f = [o.net_return for _, o in simulate(pick, rule) if o.status == ex.FILLED]
        if f:
            means.append(np.mean(f))
    return float(np.mean(means)) if means else None


report = {"window": [START, END], "population_rows": {"old": int(old.sum()), "young": int(young.sum())},
          "tokens": {"old": len(set(mint_of[old].tolist())), "young": len(set(mint_of[young].tolist()))}, "cells": {}}
for name, rule in RULES.items():
    for group, mask in (("age>=1h", old), ("age<1h", young)):
        cell = summary(simulate(np.flatnonzero(mask & breakout), rule))
        cell["random"] = random_mean(mask, rule)
        report["cells"][f"breakout {group} | {name}"] = cell
        say(f"breakout {group} | {name}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell.items()}))
s = report["cells"]["breakout age>=1h | vol3/900s"]
if s["n"] < 100:
    report["verdict"] = f"no verdict: {s['n']} trades, under 100"
else:
    ok = s["ci"][0] > 0 and s["random"] is not None and s["ci"][0] > s["random"]
    report["verdict"] = "passes" if ok else "fails"
print("VERDICT:", report["verdict"], flush=True)
report["seconds"] = round(time.time() - started)
json.dump(report, open(args.out, "w"), indent=1)
