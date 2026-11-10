"""Why do trades under a 30 % stop lose about 50 %? Read-only decomposition on direction A's population.

Entries: one random population row per token (seed 0) and the breakout rule, exit TP +200 % / stop 30 % / 1 h.
For each filled trade, the net return splits into (log terms, they add up):
  entry cost   mark at fill -> our buy price (pool impact + fee)
  move         mark at fill -> mark at the trigger (for a stop: how far past -30 % the first mark under it was)
  latency      trigger mark -> mark the sale used (2 s latency, slippage retries, stale mark)
  exit cost    that mark -> our sell price (impact + fee)
  fixed        network fees + account rent, over the size
Stop exits also keep the gap before the trigger mark, the reserve and size, for the archive check.
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
settings = DataSettings.from_config(config, lookback_days=args.lookback_days)
df, bounds = read_training_features(make_engine(), settings, end=ex._utc(args.until))
print(bounds.describe(), round(time.time() - started), "s", flush=True)
execution = ex.Execution()
paths = ex.Paths.from_frame(df, floor=ex.LiquidityFloor.from_config(config), execution=execution)
order = ex.path_rows(df)[0]
col = lambda n: pd.to_numeric(df[n], errors="coerce").to_numpy(np.float64)[order]
burned, traders, liquidity = col("burned_liquidity"), col("unique_traders"), col("liquidity")
venue = df["venue"].to_numpy(object)[order]
with np.errstate(invalid="ignore"):
    population = (burned >= 90) & (traders >= 20) & (liquidity >= 5) & (paths.price > 0) & (paths.price < 0.001)
    breakout = breakout_signal(paths) > 0
mint_of = np.empty(len(paths.ts), dtype=object)
for mint, (a, b) in paths.spans.items():
    mint_of[a:b] = mint
del df

RULE = ex.ExitRule(take_profit=2.0, stop_loss=0.3, timeout_s=3600)
ts, price, reserve, fee = paths.ts, paths.price, paths.reserve, paths.fee

# Record every sale the simulator fills.
sales = []
_orig = ex._fill_sale


def _spy(p, stop, k, trigger_at, trigger_row, sold, entry_price, execution):
    sale = _orig(p, stop, k, trigger_at, trigger_row, sold, entry_price, execution)
    sales.append((trigger_row, sale.next_row - 1, sale.price, sold, entry_price, sale.attempts))
    return sale


ex._fill_sale = _spy


def fill_row(row, stop):
    due = ts[row] + execution.entry_latency_s
    return int(np.searchsorted(ts[row + 1 : stop], due, side="left")) + row + 1


def run(rows):
    usable = rows[ts[rows] + RULE.timeout_s <= paths.data_end]
    out, busy = [], {}
    for row in usable[np.argsort(ts[usable], kind="stable")]:
        row = int(row)
        m = mint_of[row]
        if ts[row] < busy.get(m, -math.inf):
            continue
        sales.clear()
        o = ex.simulate_position(paths, row, paths.spans[m][1], 1.0, RULE, execution)
        busy[m] = o.free_at
        if o.status != ex.FILLED or len(sales) != 1:
            continue
        trig, used, sell_price, size, entry_price, attempts = sales[0]
        f = fill_row(row, paths.spans[m][1])
        mark = float(price[f])
        lg = lambda a, b: math.log(a / b) if a > 0 and b > 0 else -math.inf
        rec = {
            "mint": m, "decided": float(ts[row]), "fill_ts": float(ts[f]), "reason": o.reason, "net": o.net_return,
            "size": size, "reserve_fill": float(reserve[f]), "venue": venue[f], "fee": float(fee[f]),
            "entry_cost": lg(entry_price, mark),
            "move": lg(float(price[trig]), mark),
            "latency": lg(float(price[used]), float(price[trig])),
            "exit_cost": lg(sell_price, float(price[used])) if sell_price > 0 else -math.inf,
            "fixed": (execution.network_fee * (1 + attempts) + execution.account_rent) / size,
            "trigger_ts": float(ts[trig]), "prev_ts": float(ts[trig - 1]), "prev_move": lg(float(price[trig - 1]), mark),
            "used_ts": float(ts[used]), "reserve_used": float(reserve[used]), "attempts": attempts,
            "sell_price": sell_price, "used_mark": float(price[used]), "trigger_mark": float(price[trig]), "fill_mark": mark,
        }
        out.append(rec)
    return out


rng = np.random.default_rng(0)
pop = np.flatnonzero(population)
by = {}
for r in pop:
    by.setdefault(mint_of[r], []).append(int(r))
random_rows = np.array([v[rng.integers(len(v))] for v in by.values()], np.int64)
report = {"read": bounds.as_dict(), "random": run(random_rows), "breakout": run(np.flatnonzero(population & breakout))}


def describe(recs, name):
    d = pd.DataFrame(recs)
    print(f"\n== {name}: {len(d)} trades, mean net {d.net.mean():+.3f}, median {d.net.median():+.3f}")
    print(d.reason.value_counts().to_dict())
    for reason, g in d.groupby("reason"):
        q = lambda c: np.nanmedian(np.where(np.isfinite(g[c]), g[c], np.nan))
        print(f"  {reason:<8} n {len(g):>5} net med {g.net.median():+.3f} | log medians: entry {q('entry_cost'):+.3f} move {q('move'):+.3f} "
              f"latency {q('latency'):+.3f} exit {q('exit_cost'):+.3f} fixed {g.fixed.median():.3f} | size med {g['size'].median():.3f} reserve {g.reserve_fill.median():.1f} "
              f"| sale=0 {(~np.isfinite(g.exit_cost)).mean():.2f}")
    s = d[d.reason == "sl"]
    if len(s):
        gap = s.trigger_ts - s.prev_ts
        print("  sl: move past stop, pct of trades with trigger mark below -40/-50/-70 %:",
              [round(float((np.exp(s.move) - 1 < -x).mean()), 2) for x in (0.4, 0.5, 0.7)],
              "gap before trigger s median/p90", round(gap.median(), 1), round(gap.quantile(0.9), 1),
              "prev mark med", round(float(np.expm1(s.prev_move).median()), 3))
        print("  sl by venue:", s.groupby("venue").net.agg(["count", "median"]).round(3).to_dict("index"))


for k in ("random", "breakout"):
    describe(report[k], k)
report["seconds"] = round(time.time() - started)
json.dump(report, open(args.out, "w"), default=float)
print("=== done", report["seconds"], "s", flush=True)
