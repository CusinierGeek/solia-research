"""Confirmation of two leads on unused data. Read-only. Fixed on 2026-11-09 04:55 UTC, before reading any row after
2026-11-08 05:00 UTC.

Bot setting, unchanged (champion, floor 20 traders / 5 SOL, every row, pump >= 0.5, TP +200 %, SL vol3, 900 s, re-entries).
Leads, from earlier periods:
  one entry per token: +20.0 % vs +15.1 % for the bot on 11-07 03:12 .. 11-08 05:00 (re-entries +4.9 %)
  partial exit: at +200 % sell what recovers the stake, the rest at +300 % with its stop moved to the entry mark, 900 s
                (+7.8 [+1.6, +14.2] and +4.0 [-4.0, +12.6] points vs the bot on two periods)
Arms: bot; one entry; partial; one entry + partial. Window: decisions 2026-11-08 05:00 .. --until (whole hours).
Reading, per lead: confirmed if its paired difference against the bot (per token, tokens drawn) has a 95 % CI above 0.
Otherwise not confirmed on this window: no change; it can be read again on a longer window, with this rule unchanged.
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



def simulate(w, rule, once=False):
    out, busy, seen = [], {}, set()
    for i in np.lexsort((w.row,)):
        mint, row = w.mint[i], int(w.row[i])
        if paths.ts[row] < busy.get(mint, -math.inf) or (once and mint in seen):
            continue
        o = ex.simulate_position(paths, row, paths.spans[mint][1], float(w.expected_return[i]), rule, execution)
        busy[mint] = o.free_at
        if o.status == ex.FILLED:
            seen.add(mint)
        out.append((mint, o))
    return out


rng = np.random.default_rng(0)
START = pd.Timestamp("2026-11-08 05:00+00").timestamp()
PARTIAL = ex.ExitRule(take_profit=3.0, stop_loss="vol3", timeout_s=900, partial_at=2.0, partial_share=ex.STAKE, rest_stop="entry")


def summary(entries, reference=None):
    trades = [(m, o) for m, o in entries if o.status == ex.FILLED]
    if not trades:
        return {"n": 0}
    r = np.array([o.net_return for _, o in trades])
    tokens = np.array([m for m, _ in trades], dtype=object)
    low, high = ex.bootstrap_by_token(r, tokens, samples=2000, ci=0.95, rng=rng)
    out = {"n": len(r), "tokens": len(set(tokens.tolist())), "mean": float(r.mean()), "median": float(np.median(r)), "ci": [low, high],
           "win": float((r > 0).mean()), "total_loss": float((r <= -0.99).mean()), "tp": float(np.mean([o.reason == "tp" for _, o in trades]))}
    if reference is not None:
        out["vs_bot"] = ex.paired_difference(entries, reference, samples=2000, ci=0.95, rng=rng)
    return out


bot_w = windows.take(floor_mask)
bot_w = bot_w.take((bot_w.pump_prob >= PUMP) & (paths.ts[bot_w.row] >= START))
bot = simulate(bot_w, RULE)
report = {"read": bounds.as_dict(), "arms": {"bot": summary(bot)}}
for name, (rule, once) in {"one entry": (RULE, True), "partial": (PARTIAL, False), "one entry + partial": (PARTIAL, True)}.items():
    report["arms"][name] = summary(simulate(bot_w, rule, once=once), reference=bot)
for name, s in report["arms"].items():
    vs = s.get("vs_bot") or {}
    print(f"{name:<22} n {s.get('n')} tok {s.get('tokens')} mean {s.get('mean', 0):+.3f} ci {[round(x, 3) for x in s.get('ci', [])]} med {s.get('median', 0):+.3f} "
          f"loss {s.get('total_loss', 0):.2f} tp {s.get('tp', 0):.2f}", "" if not vs else f"vs bot {vs.get('mean_diff')} {vs.get('mean_diff_ci')}", flush=True)
for lead in ("one entry", "partial"):
    vs = report["arms"][lead].get("vs_bot") or {}
    ok = vs.get("mean_diff_ci") is not None and vs["mean_diff_ci"][0] > 0
    report.setdefault("verdict", {})[lead] = "confirmed" if ok else "not confirmed on this window"
    print("VERDICT", lead, report["verdict"][lead], flush=True)
report["seconds"] = round(time.time() - started)
json.dump(report, open(args.out, "w"), indent=1)
