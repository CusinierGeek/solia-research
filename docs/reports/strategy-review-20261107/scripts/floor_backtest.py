"""Backtest of the live bot's setting: floor of 20 traders / 5 SOL, a look at every row (about 10 s),
pump >= 0.5, TP +200 %, SL vol3, 900 s, re-entries allowed. Read-only.

Arms, with the same exits and the same simulated fills as exit_sim:
- no floor at the same cadence, and older cadences with and without the floor;
- a pause before re-entering a token, no re-entry at all, a minimum age.
A look at cadence C falls on the token's first window at or after first window + k x C.
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch

from solia_ai import exit_sim as ex
from solia_ai.config import horizon_seconds, load_config
from solia_ai.db import DataSettings, make_engine, read_training_features
from solia_ai.evaluate import holdout_dataset
from solia_ai.inference import Predictor

parser = argparse.ArgumentParser()
parser.add_argument("--until", required=True)
parser.add_argument("--lookback-days", type=float)
parser.add_argument("--out", required=True)
args = parser.parse_args()
PUMP, TRADERS, LIQUIDITY, HORIZON = 0.5, 20, 5.0, "15m"
RULE = ex.REFERENCE_RULE
started = time.time()
engine = make_engine()
predictor = Predictor.from_champion(engine)
h = list(predictor.horizons).index(HORIZON)
config = load_config()
config = {**config, "data": {**config.get("data", {}), "window_stride_seconds": 0.0}}
settings = DataSettings.from_config(config, lookback_days=args.lookback_days)
print("reading", flush=True)
df, bounds = read_training_features(engine, settings, end=ex._utc(args.until))
print(bounds.describe(), round(time.time() - started), "s", flush=True)
dataset = holdout_dataset(df, config, predictor)
print("scoring", len(dataset), flush=True)
predictions, truth = ex.score_windows(predictor, dataset, "cuda" if torch.cuda.is_available() else "cpu")
execution = ex.Execution()
paths = ex.Paths.from_frame(df, floor=ex.LiquidityFloor.from_config(config), execution=execution)
samples, indices = dataset.samples, dataset.indices
windows = ex.Windows.align(
    paths, samples.mint[indices], samples.end_ts[indices], predictions["pump_prob"][:, h],
    predictions["expected_return"][:, h], truth["mask"][:, h] == 1,
)
windows = ex.uncensored(windows, paths, max(horizon_seconds(HORIZON), ex._longest_hold(RULE)))
order = ex.path_rows(df)[0]
traders = pd.to_numeric(df["unique_traders"], errors="coerce").to_numpy(np.float64)[order]
liquidity = pd.to_numeric(df["liquidity"], errors="coerce").to_numpy(np.float64)[order]
age = pd.to_numeric(df["token_age_sec"], errors="coerce").to_numpy(np.float64)[order]
with np.errstate(invalid="ignore"):
    passes = (traders >= TRADERS) & (liquidity >= LIQUIDITY)
floor_mask = passes[windows.row]
print("windows", len(windows), "over the floor", int(floor_mask.sum()), "tokens over it",
      len(set(windows.mint[floor_mask].tolist())), "of", len(set(windows.mint.tolist())), flush=True)
del predictions, truth, dataset, samples


def at_cadence(w, cadence):
    """Windows the bot would look at, at one look per token per ``cadence`` seconds."""
    if not cadence:
        return w
    t = paths.ts[w.row]
    keep, first, nxt = [], {}, {}
    for i in np.lexsort((w.row,)):
        m = w.mint[i]
        if m not in first:
            first[m] = t[i]
            nxt[m] = t[i] + cadence
            continue
        if t[i] >= nxt[m]:
            keep.append(i)
            steps = math.floor((t[i] - first[m]) / cadence) + 1
            nxt[m] = first[m] + steps * cadence
    return w.take(np.array(sorted(keep), np.int64))


def simulate(w, pause=0.0, once=False):
    """exit_sim.simulate_rule, plus a pause after a position and a one-entry-per-token variant."""
    out, rows = [], []
    busy, seen = {}, set()
    for i in np.lexsort((w.row,)):
        mint, row = w.mint[i], int(w.row[i])
        if paths.ts[row] < busy.get(mint, -math.inf) or (once and mint in seen):
            continue
        outcome = ex.simulate_position(
            paths, row, paths.spans[mint][1], float(w.expected_return[i]), RULE, execution
        )
        busy[mint] = outcome.free_at + (pause if outcome.status == ex.FILLED else 0.0)
        if outcome.status == ex.FILLED:
            seen.add(mint)
        out.append((mint, outcome))
        rows.append(row)
    return out, np.array(rows, np.int64)


rng = np.random.default_rng(0)


def summary(entries, reference=None):
    trades = [(m, o) for m, o in entries if o.status == ex.FILLED]
    if not trades:
        return {"n": 0}
    r = np.array([o.net_return for _, o in trades])
    tokens = np.array([m for m, _ in trades], dtype=object)
    low, high = ex.bootstrap_by_token(r, tokens, samples=2000, ci=0.95, rng=rng)
    out = {
        "n": len(r), "tokens": len(set(tokens.tolist())), "mean": float(r.mean()), "median": float(np.median(r)),
        "ci": [low, high], "win": float((r > 0).mean()), "total_loss": float((r <= -0.99).mean()),
        "tp": float(np.mean([o.reason == "tp" and o.net_return > 0 for _, o in trades])),
        "failed": sum(o.status == ex.FAILED for _, o in entries),
    }
    if reference is not None:
        out["vs_bot"] = ex.paired_difference(entries, reference, samples=2000, ci=0.95, rng=rng)
    return out


selected = lambda w: w.take(w.pump_prob >= PUMP)
floor_all = windows.take(floor_mask)
bot_w = selected(floor_all)
bot, bot_rows = simulate(bot_w)
report = {"read": bounds.as_dict(), "rule": RULE.label, "windows": len(windows), "arms": {}}
report["arms"]["bot: floor, 10 s"] = summary(bot)
arms = {
    "no floor, 10 s": lambda: simulate(selected(windows)),
    "floor, 60 s": lambda: simulate(selected(at_cadence(floor_all, 60))),
    "floor, 180 s": lambda: simulate(selected(at_cadence(floor_all, 180))),
    "floor, 300 s": lambda: simulate(selected(at_cadence(floor_all, 300))),
    "no floor, 180 s (old bot, quiet)": lambda: simulate(selected(at_cadence(windows, 180))),
    "no floor, 300 s (old bot, busy)": lambda: simulate(selected(at_cadence(windows, 300))),
    "floor, 10 s, pause 5 min": lambda: simulate(bot_w, pause=300.0),
    "floor, 10 s, pause 15 min": lambda: simulate(bot_w, pause=900.0),
    "floor, 10 s, one entry per token": lambda: simulate(bot_w, once=True),
    "floor, 10 s, age >= 120 s": lambda: simulate(bot_w.take(age[bot_w.row] >= 120)),
    "floor, 10 s, age >= 180 s": lambda: simulate(bot_w.take(age[bot_w.row] >= 180)),
}
for name, run in arms.items():
    report["arms"][name] = summary(run()[0], reference=bot)
    print(name, json.dumps(report["arms"][name])[:200], flush=True)

seen, first_flags = set(), []
filled = [(m, o, r) for (m, o), r in zip(bot, bot_rows, strict=True) if o.status == ex.FILLED]
for m, _, _ in filled:
    first_flags.append(m not in seen)
    seen.add(m)
first_flags = np.array(first_flags)


def cut(mask):
    return summary([(m, o) for (m, o, _), keep in zip(filled, mask, strict=True) if keep])


ages = np.array([age[r] for _, _, r in filled])
report["splits"] = {"first entry": cut(first_flags), "re-entry": cut(~first_flags)}
for a, b in ((0, 120), (120, 180), (180, 300), (300, 600), (600, 1800), (1800, math.inf)):
    report["splits"][f"age {a}-{b} s"] = cut((ages >= a) & (ages < b))
report["seconds"] = round(time.time() - started)
json.dump(report, open(args.out, "w"), indent=1)
print(f"\n{report['read']}")
for section in ("arms", "splits"):
    print(section)
    for name, s in report[section].items():
        if not s.get("n"):
            print(f"  {name:<36} no trades")
            continue
        vs = s.get("vs_bot") or {}
        print(
            f"  {name:<36} n {s['n']:>4} tok {s['tokens']:>4} mean {s['mean']:+.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}] med {s['median']:+.3f} "
            f"win {s['win']:.2f} loss {s['total_loss']:.2f} tp {s['tp']:.2f} {'' if not vs or vs.get('mean_diff') is None else 'vs bot %+.3f [%+.3f,%+.3f] on %d tok' % (vs['mean_diff'], vs['mean_diff_ci'][0], vs['mean_diff_ci'][1], vs['n_tokens'])}"
        )
print("=== period done", round(time.time() - started), "s", flush=True)
