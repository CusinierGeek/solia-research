"""The user's manual checks as filters on the bot's setting. Read-only. Fixed on 2026-11-08, before any run.

Bot setting, unchanged: champion, floor 20 traders / 5 SOL, a look at every row, pump >= 0.5, TP +200 %, SL vol3, 900 s,
re-entries allowed. Each filled trade keeps, from its decision row: hs_top10_share, hs_top1_share, hs_top10_net_flow,
hs_holders, meta_has_twitter, meta_narrative_grads_3h. A filter keeps a trade when its condition holds; unknown fails.
Filters are applied to the bot's trades as simulated (a refused entry doesn't free the token for another one earlier).

Cells (fixed): top10 <= 0.3 / 0.5 / 0.7; top1 <= 0.1 / 0.2; top10 flow >= -0.1 / >= 0; X page; narrative >= 1.
Choose: 2026-11-03 14:00 .. 2026-11-07 03:10 UTC (link flags live only), read as two windows. The cell with the best mean
among those keeping >= 100 trades. Confirm, unchanged, on >= 24 h after 2026-11-08 05:00 UTC: kept trades (a) >= 100,
(b) 95 % CI by token above 0, (c) mean at least 5 points above all the bot's trades in that window. Otherwise it fails.
This script writes the trades; read_filters.py applies the reading.
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
COLS = ["hs_top10_share", "hs_top1_share", "hs_top10_net_flow", "hs_holders", "meta_has_twitter", "meta_narrative_grads_3h"]
started = time.time()
engine = make_engine()
predictor = Predictor.from_champion(engine)
h = list(predictor.horizons).index(HORIZON)
config = load_config()
config = {**config, "data": {**config.get("data", {}), "window_stride_seconds": 0.0}}
settings = DataSettings.from_config(config, lookback_days=args.lookback_days)
df, bounds = read_training_features(engine, settings, end=ex._utc(args.until))
print(bounds.describe(), round(time.time() - started), "s", flush=True)
missing = [c for c in COLS if c not in df.columns]
assert not missing, f"columns not read: {missing}"
dataset = holdout_dataset(df, config, predictor)
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
num = lambda n: pd.to_numeric(df[n], errors="coerce").to_numpy(np.float64)[order]
traders, liquidity = num("unique_traders"), num("liquidity")
extra = {c: num(c) for c in COLS}
venue = df["venue"].to_numpy(object)[order]
del df, predictions, truth, dataset, samples
with np.errstate(invalid="ignore"):
    passes = (traders >= TRADERS) & (liquidity >= LIQUIDITY)
bot_w = windows.take(passes[windows.row])
bot_w = bot_w.take(bot_w.pump_prob >= PUMP)

trades, busy = [], {}
for i in np.lexsort((bot_w.row,)):
    mint, row = bot_w.mint[i], int(bot_w.row[i])
    if paths.ts[row] < busy.get(mint, -math.inf):
        continue
    o = ex.simulate_position(paths, row, paths.spans[mint][1], float(bot_w.expected_return[i]), RULE, execution)
    busy[mint] = o.free_at
    if o.status == ex.FILLED:
        rec = {"mint": mint, "ts": float(paths.ts[row]), "net": o.net_return, "reason": o.reason, "venue": venue[row]}
        rec.update({c: (None if not np.isfinite(extra[c][row]) else float(extra[c][row])) for c in COLS})
        trades.append(rec)
r = np.array([t["net"] for t in trades])
print("trades", len(trades), "mean", round(float(r.mean()), 3) if len(r) else None,
      "known:", {c: sum(t[c] is not None for t in trades) for c in COLS}, flush=True)
json.dump({"read": bounds.as_dict(), "trades": trades, "seconds": round(time.time() - started)}, open(args.out, "w"), indent=1)
print("=== done", round(time.time() - started), "s", flush=True)
