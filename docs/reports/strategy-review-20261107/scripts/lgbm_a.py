"""Direction A, one LightGBM attempt: can a model pay where plain rules didn't? Read-only.

Fixed on 2026-11-07, before any training.

Population at the decision row: LP burned >= 90 %, >= 20 traders, >= 5 SOL, 0 < price < 0.001 SOL.
Label, per row and exit: the take-profit (+200 %) is reached before the stop or the deadline, with
the simulator's fills. Exits: vol3 / 900 s and 30 % / 1 h. One classifier per exit.
Inputs: the row's features, plus price and activity changes over the previous 6 and 30 marks.
Time, never shuffled, per decision row (UTC):
  train  10-27 .. 11-02 02:11, labels complete before the end, every third row
  valid  11-02 02:11 .. 11-04 02:11, early stopping and the choice of the cell
  test1  11-04 02:11 .. 11-06 02:11
  test2  11-06 02:11 .. 11-07 03:10
Entries: score >= the validation quantile of its exit, one position per token at a time. Cells: 2 exits x
top 10 / 5 / 2 / 1 % of population rows. The cell with the best validation mean, at least 100 trades,
is chosen on validation only.

Reading: that cell, unchanged, must on each of test1 and test2 have (a) at least 100 trades, (b) a 95 %
CI by token above 0, (c) a lower bound above the mean of random population rows (one per token, 20
draws). Otherwise the attempt fails. No second attempt with other inputs or labels on these periods.
"""

import json
import math
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import text

from solia_ai import exit_sim as ex
from solia_ai.config import load_config
from solia_ai.db import make_engine
from solia_ai.features import DB_FEATURE_COLUMNS, build_feature_matrix

OUT = "/c"
T = {k: pd.Timestamp(v).timestamp() for k, v in {
    "start": "2026-10-27 00:00+00", "valid": "2026-11-02 02:11+00", "test1": "2026-11-04 02:11+00",
    "test2": "2026-11-06 02:11+00", "end": "2026-11-07 03:10+00"}.items()}
EXITS = {
    "vol3/900s": ex.ExitRule(take_profit=2.0, stop_loss="vol3", timeout_s=900),
    "30%/1h": ex.ExitRule(take_profit=2.0, stop_loss=0.3, timeout_s=3600),
}
TOPS = (0.10, 0.05, 0.02, 0.01)
started = time.time()
say = lambda *a: print(*a, round(time.time() - started), "s", flush=True)
engine = make_engine()
config = load_config()

cache = os.path.join(OUT, "lgbm_pop.parquet.csv")
if os.path.exists(cache):
    pop = pd.read_csv(cache)
else:
    parts, t = [], T["start"]
    while t < T["end"]:
        parts.append(pd.read_sql(
            text(
                """select mint, extract(epoch from ts) t from features where ts >= to_timestamp(:a) and ts < to_timestamp(:b)
                and burned_liquidity >= 90 and unique_traders >= 20 and liquidity >= 5 and price > 0 and price < 0.001"""
            ),
            engine, params={"a": t, "b": t + 21600},
        ))
        t += 21600
        say("population", sum(map(len, parts)), pd.Timestamp(t, unit="s"))
    pop = pd.concat(parts, ignore_index=True)
    pop.to_csv(cache, index=False)
say("population tokens", pop["mint"].nunique(), "rows", len(pop))

# Only the marks a decision can read: 7 min before it (the 30-mark changes), 1 h after it (the longest hold).
BEFORE, AFTER = 420.0, 3700.0
intervals = []
for mint, g in pop.sort_values(["mint", "t"]).groupby("mint", sort=False):
    t = g["t"].to_numpy()
    breaks = np.flatnonzero(t[1:] - AFTER > t[:-1] + BEFORE) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(t)], strict=True):
        intervals.append((mint, float(t[a] - BEFORE), float(t[b - 1] + AFTER)))
say("intervals", len(intervals), "seconds covered", round(sum(i[2] - i[1] for i in intervals)))
columns = ", ".join(f"f.{c}" for c in dict.fromkeys(["mint", "ts", "venue", *DB_FEATURE_COLUMNS]))
frames = []
for i in range(0, len(intervals), 1000):
    part = intervals[i : i + 1000]
    chunk = pd.read_sql(
        text(
            f"""select {columns} from unnest(cast(:m as text[]), cast(:a as float8[]), cast(:b as float8[])) w(mint, a, b)
            cross join lateral (select * from features x where x.mint = w.mint and x.ts >= to_timestamp(w.a) and x.ts <= to_timestamp(w.b)) f"""
        ),
        engine, params={"m": [x[0] for x in part], "a": [x[1] for x in part], "b": [x[2] for x in part]},
    )
    for c in chunk.columns:
        if c in ("price", "liquidity", "volatility", "unique_traders", "volume", "burned_liquidity"):
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(np.float64)
        elif c not in ("mint", "ts", "venue"):
            chunk[c] = pd.to_numeric(chunk[c], errors="coerce").astype(np.float32)
    chunk["mint"] = chunk["mint"].astype("category")
    chunk["venue"] = chunk["venue"].astype("category")
    frames.append(chunk)
    if i % 2000 == 0:
        say("read", i, "of", len(intervals), sum(map(len, frames)), "rows", sum(f.memory_usage(deep=True).sum() for f in frames) // 2**20, "MiB")
df = pd.concat(frames, ignore_index=True)
del frames
df["mint"] = df["mint"].astype(object)
df["venue"] = df["venue"].astype(object)
df = df.drop_duplicates(["mint", "ts"], ignore_index=True)
say("rows", len(df))
execution = ex.Execution()
paths = ex.Paths.from_frame(df, floor=ex.LiquidityFloor.from_config(config), execution=execution)
order = ex.path_rows(df)[0]
df = df.iloc[order].reset_index(drop=True)
matrix = build_feature_matrix(df).astype(np.float32)
col = lambda n: pd.to_numeric(df[n], errors="coerce").to_numpy(np.float64)
burned, traders, liquidity, volume = col("burned_liquidity"), col("unique_traders"), col("liquidity"), col("volume")
del df
ts = paths.ts
with np.errstate(invalid="ignore"):
    population = (burned >= 90) & (traders >= 20) & (liquidity >= 5) & (paths.price < 0.001) & (ts >= T["start"]) & (ts < T["end"])
mint_of = np.empty(len(ts), dtype=object)
start_of = np.zeros(len(ts), np.int64)
for mint, (a, b) in paths.spans.items():
    mint_of[a:b] = mint
    start_of[a:b] = a


def lagged(values, k):
    out = np.full(len(values), np.nan)
    idx = np.arange(len(values))
    ok = idx - k >= start_of
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = values[ok] / values[idx[ok] - k] - 1.0
    return out.astype(np.float32)


extra = np.column_stack([lagged(s, k) for s in (paths.price, traders, liquidity, volume) for k in (6, 30)])
say("population rows", int(population.sum()))
periods = {"train": (T["start"], T["valid"]), "valid": (T["valid"], T["test1"]), "test1": (T["test1"], T["test2"]), "test2": (T["test2"], T["end"])}
rows_in = {p: np.flatnonzero(population & (ts >= a) & (ts < b)) for p, (a, b) in periods.items()}
print({p: (len(r), len(set(mint_of[r].tolist()))) for p, r in rows_in.items()}, flush=True)


def label(rows, rule):
    y = np.zeros(len(rows), np.int8)
    keep = np.zeros(len(rows), bool)
    for k, row in enumerate(rows):
        o = ex.simulate_position(paths, int(row), paths.spans[mint_of[row]][1], 1.0, rule, execution)
        if o.status == ex.FILLED:
            keep[k] = True
            y[k] = o.reason == "tp" and o.net_return > 0
    return y, keep


def inputs(rows):
    return np.hstack([matrix[rows], extra[rows]])


def simulate(rows, rule, end):
    usable = rows[ts[rows] + rule.timeout_s <= end]
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


rng = np.random.default_rng(0)


def summary(entries):
    trades = [(m, o) for m, o in entries if o.status == ex.FILLED]
    if not trades:
        return {"n": 0, "tokens": 0}
    r = np.array([o.net_return for _, o in trades])
    tokens = np.array([m for m, _ in trades], dtype=object)
    low, high = ex.bootstrap_by_token(r, tokens, samples=2000, ci=0.95, rng=rng)
    return {"n": len(r), "tokens": len(set(tokens.tolist())), "mean": float(r.mean()), "median": float(np.median(r)), "ci": [low, high],
            "win": float((r > 0).mean()), "total_loss": float((r <= -0.99).mean()), "tp": float(np.mean([o.reason == "tp" and o.net_return > 0 for _, o in trades]))}


def random_mean(rows, rule, end):
    by = {}
    for row in rows:
        by.setdefault(mint_of[row], []).append(int(row))
    means = []
    for _ in range(20):
        pick = np.array([v[rng.integers(len(v))] for v in by.values()], np.int64)
        means.append(np.mean([o.net_return for _, o in simulate(pick, rule, end) if o.status == ex.FILLED]))
    return float(np.mean(means))


report = {"periods": {p: {"rows": len(r), "tokens": len(set(mint_of[r].tolist()))} for p, r in rows_in.items()}, "exits": {}}
for name, rule in EXITS.items():
    train = rows_in["train"]
    train = train[ts[train] + rule.timeout_s <= T["valid"]][::3]
    valid = rows_in["valid"]
    valid_lab = valid[ts[valid] + rule.timeout_s <= T["test1"]][::3]
    say(name, "labels", len(train), len(valid_lab))
    y_tr, k_tr = label(train, rule)
    y_va, k_va = label(valid_lab, rule)
    say(name, "positives train", int(y_tr[k_tr].sum()), "of", int(k_tr.sum()), "valid", int(y_va[k_va].sum()), "of", int(k_va.sum()))
    model = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
         "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 0, "num_threads": 6, "verbose": -1},
        lgb.Dataset(inputs(train[k_tr]), label=y_tr[k_tr]), num_boost_round=2000,
        valid_sets=[lgb.Dataset(inputs(valid_lab[k_va]), label=y_va[k_va])], callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    say(name, "best iteration", model.best_iteration, "valid logloss", model.best_score["valid_0"]["binary_logloss"])
    scores = {p: model.predict(inputs(rows_in[p])) for p in ("valid", "test1", "test2")}
    gains = model.feature_importance("gain")
    top = np.argsort(-gains)[:10]
    names = [*DB_FEATURE_COLUMNS]
    entry = {"best_iteration": model.best_iteration, "cells": {}, "random": {}, "top_gain_idx": [int(i) for i in top]}
    for p, end in (("valid", T["test1"]), ("test1", T["test2"]), ("test2", T["end"])):
        entry["random"][p] = random_mean(rows_in[p], rule, end)
    for q in TOPS:
        threshold = float(np.quantile(scores["valid"], 1 - q))
        cell = {"threshold": threshold}
        for p, end in (("valid", T["test1"]), ("test1", T["test2"]), ("test2", T["end"])):
            cell[p] = summary(simulate(rows_in[p][scores[p] >= threshold], rule, end))
        entry["cells"][f"top {q:.0%}"] = cell
        say(name, f"top {q:.0%}", json.dumps({p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cell[p].items() if k in ("n", "mean", "ci", "total_loss", "tp")} for p in ("valid", "test1", "test2")})[:420])
    report["exits"][name] = entry
    json.dump(report, open(os.path.join(OUT, "lgbm-a.json"), "w"), indent=1)

cells = [(n, c, v) for n, e in report["exits"].items() for c, v in e["cells"].items() if v["valid"].get("n", 0) >= 100]
if not cells:
    print("VERDICT: failed, no validation cell with 100 trades")
    sys.exit(0)
n, c, v = max(cells, key=lambda x: x[2]["valid"]["mean"])
report["chosen"] = f"{n} | {c}"
print("chosen on validation:", n, c, "valid mean", round(v["valid"]["mean"], 3), "n", v["valid"]["n"])
passes = True
for p in ("test1", "test2"):
    s, base = v[p], report["exits"][n]["random"][p]
    a = s.get("n", 0) >= 100
    b = s.get("n", 0) > 0 and s["ci"][0] > 0
    d = s.get("n", 0) > 0 and s["ci"][0] > base
    passes &= a and b and d
    print(p, json.dumps({k: s.get(k) for k in ("n", "tokens", "mean", "ci", "median", "win", "total_loss", "tp")}), "random", round(base, 3), "| >=100", a, "CI>0", b, "above random", d)
report["verdict"] = "passes" if passes else "failed"
print("VERDICT:", report["verdict"])
json.dump(report, open(os.path.join(OUT, "lgbm-a.json"), "w"), indent=1)
print("=== done", flush=True)
