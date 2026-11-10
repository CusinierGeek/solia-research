"""Direction A: plain rules on tokens without the operator pattern. No model, no training. Read-only.

Fixed on 2026-11-07, before any run.

Population, at the decision row: LP burned >= 90 %, >= 20 traders, >= 5 SOL of liquidity, price < 0.001
SOL. All tokens are counted, including those that never pumped.
Entries (one position per token at a time, re-entry allowed except "first"):
  first     the token's first row in the population, once
  breakout  in the population, price above the high of the previous 30 marks (at least 6)
  pressure  in the population, buy_volume_ratio >= 0.6
  both      breakout and pressure
Exits: TP +200 % with stop vol3 / 900 s (today's), stop 30 % / 1 h, stop 30 % / 4 h.
Baseline: one random row of the population per token, 20 draws, for each exit.

Reading: choose the best mean among the 12 cells on p1. That one cell, unchanged, must on each of p2
and "fresh" have (a) at least 100 trades, (b) a 95 % CI by token above 0, (c) a lower bound above the
random baseline's mean. Otherwise this family of rules shows no edge. 300 trades over two periods and
a paper session stay required before any live use.
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from solia_ai import exit_sim as ex
from solia_ai.config import load_config
from solia_ai.db import DataSettings, make_engine, read_training_features
from solia_ai.model_value import breakout_signal

parser = argparse.ArgumentParser()
parser.add_argument("--until", required=True)
parser.add_argument("--lookback-days", type=float)
parser.add_argument("--out", required=True)
args = parser.parse_args()
started = time.time()
config = load_config()
engine = make_engine()
settings = DataSettings.from_config(config, lookback_days=args.lookback_days)
df, bounds = read_training_features(engine, settings, end=ex._utc(args.until))
print(bounds.describe(), round(time.time() - started), "s", flush=True)
execution = ex.Execution()
paths = ex.Paths.from_frame(df, floor=ex.LiquidityFloor.from_config(config), execution=execution)
order = ex.path_rows(df)[0]


def col(name):
    return pd.to_numeric(df[name], errors="coerce").to_numpy(np.float64)[order]


burned, traders, liquidity, pressure = col("burned_liquidity"), col("unique_traders"), col("liquidity"), col("buy_volume_ratio")
with np.errstate(invalid="ignore"):
    population = (burned >= 90) & (traders >= 20) & (liquidity >= 5) & (paths.price < 0.001)
    pressured = pressure >= 0.6
with np.errstate(invalid="ignore"):
    breakout = breakout_signal(paths) > 0  # NaN (under 6 marks of history) is no breakout
mint_of = np.empty(len(paths.ts), dtype=object)
for mint, (a, b) in paths.spans.items():
    mint_of[a:b] = mint
del df

EXITS = {
    "vol3/900s": ex.ExitRule(take_profit=2.0, stop_loss="vol3", timeout_s=900),
    "30%/1h": ex.ExitRule(take_profit=2.0, stop_loss=0.3, timeout_s=3600),
    "30%/4h": ex.ExitRule(take_profit=2.0, stop_loss=0.3, timeout_s=14400),
}
data_end = paths.data_end
rng = np.random.default_rng(0)


def simulate(rows, rule, once=False):
    """Rows in time order per mint, one position per mint at a time."""
    usable = rows[paths.ts[rows] + rule.timeout_s <= data_end]
    out, busy, seen = [], {}, set()
    for row in usable[np.argsort(paths.ts[usable], kind="stable")]:
        row = int(row)
        mint = mint_of[row]
        if paths.ts[row] < busy.get(mint, -math.inf) or (once and mint in seen):
            continue
        outcome = ex.simulate_position(paths, row, paths.spans[mint][1], 1.0, rule, execution)
        busy[mint] = outcome.free_at
        if outcome.status == ex.FILLED:
            seen.add(mint)
        out.append((mint, outcome))
    return out


def summary(entries):
    trades = [(m, o) for m, o in entries if o.status == ex.FILLED]
    if not trades:
        return {"n": 0, "tokens": 0}
    r = np.array([o.net_return for _, o in trades])
    tokens = np.array([m for m, _ in trades], dtype=object)
    low, high = ex.bootstrap_by_token(r, tokens, samples=2000, ci=0.95, rng=rng)
    return {
        "n": len(r), "tokens": len(set(tokens.tolist())), "mean": float(r.mean()), "median": float(np.median(r)),
        "ci": [low, high], "win": float((r > 0).mean()), "total_loss": float((r <= -0.99).mean()),
        "tp": float(np.mean([o.reason == "tp" and o.net_return > 0 for _, o in trades])),
        "failed": sum(o.status == ex.FAILED for _, o in entries),
        "hold_s": float(np.median([o.hold_s for _, o in trades])),
    }


pop_rows = np.flatnonzero(population)
print("population rows", len(pop_rows), "tokens", len(set(mint_of[pop_rows].tolist())), "of", len(paths.spans), flush=True)
RULES = {
    "first": (pop_rows, True),
    "breakout": (np.flatnonzero(population & breakout), False),
    "pressure": (np.flatnonzero(population & pressured), False),
    "both": (np.flatnonzero(population & breakout & pressured), False),
}
report = {"read": bounds.as_dict(), "population_rows": int(len(pop_rows)), "cells": {}, "random": {}}
by_mint = {}
for row in pop_rows:
    by_mint.setdefault(mint_of[row], []).append(int(row))
for name, rule in EXITS.items():
    means, ns, losses = [], [], []
    for _ in range(20):
        rows = np.array([v[rng.integers(len(v))] for v in by_mint.values()], np.int64)
        trades = [o for _, o in simulate(rows, rule) if o.status == ex.FILLED]
        means.append(float(np.mean([o.net_return for o in trades])))
        ns.append(len(trades))
        losses.append(float(np.mean([o.net_return <= -0.99 for o in trades])))
    report["random"][name] = {"mean": float(np.mean(means)), "band": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], "n": float(np.mean(ns)), "total_loss": float(np.mean(losses))}
    print("random", name, json.dumps(report["random"][name]), flush=True)
    for entry, (rows, once) in RULES.items():
        s = summary(simulate(rows, rule, once=once))
        report["cells"][f"{entry} | {name}"] = s
        print(f"{entry} | {name}", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "ci"}, [round(x, 3) for x in s.get("ci", [])], flush=True)
report["seconds"] = round(time.time() - started)
json.dump(report, open(args.out, "w"), indent=1)
print("=== period done", report["seconds"], "s", flush=True)
