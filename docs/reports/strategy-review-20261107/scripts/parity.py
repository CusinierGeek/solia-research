"""Does the simulator reproduce the bot? Read-only, CPU only, reads tokens by mint.

Part 1, exits: every closed trade of sessions 5 and 6, replayed by exit_sim from its own decision row.
Part 2, entries: session 6's period scored offline on every row over the floor, with exit_sim's entry
loop, against the trades the bot really took.
"""

import collections
import json
import statistics as st
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from solia_ai import exit_sim as ex
from solia_ai.config import load_config
from solia_ai.db import make_engine
from solia_ai.features import DB_FEATURE_COLUMNS, build_feature_matrix
from solia_ai.inference import Predictor

torch.set_num_threads(1)
engine = make_engine()
predictor = Predictor.from_champion(engine)
config = load_config()
execution = ex.Execution()
floor = ex.LiquidityFloor.from_config(config)
RULE = ex.REFERENCE_RULE
columns = ", ".join(dict.fromkeys(["mint", "ts", "venue", *DB_FEATURE_COLUMNS]))


def read(mints):
    frames = []
    for i in range(0, len(mints), 200):
        frames.append(pd.read_sql(
            text(f"select {columns} from features where mint = any(:m) and ts >= '2026-11-06 02:00+00' order by mint, ts"),
            engine, params={"m": list(mints[i : i + 200])},
        ))
    return pd.concat(frames, ignore_index=True)


trades = pd.read_sql(text(
    """select t.id, t.session_id s, t.mint, p.ts decision, t.opened_at, t.closed_at, t.pnl_pct, t.exit_reason, t.status,
    t.size::float8 sz, p.pump_prob pump from trades t join predictions p on p.id=t.prediction_id
    where t.session_id in (5,6) order by t.id"""), engine)
session6 = pd.read_sql(text("select started_at from bot_sessions where id=6"), engine)["started_at"][0]
end = pd.read_sql(text("select max(ts) m from features"), engine)["m"][0] - timedelta(minutes=16)
candidates = pd.read_sql(text(
    """select distinct mint from features where ts >= :s and ts < :e and unique_traders >= 20 and liquidity >= 5"""),
    engine, params={"s": session6, "e": end})["mint"].tolist()
print("session 6 from", session6, "to", end, "tokens over the floor", len(candidates), flush=True)
mints = sorted(set(trades["mint"]) | set(candidates))
df = read(mints)
print("rows", len(df), flush=True)
paths = ex.Paths.from_frame(df, floor=floor, execution=execution)
order = ex.path_rows(df)[0]
frame = df.iloc[order].reset_index(drop=True)
del df
stamps = pd.to_datetime(frame["ts"], utc=True)


def row_of(mint, when):
    a, b = paths.spans[mint]
    t = when.timestamp()
    i = a + int(np.searchsorted(paths.ts[a:b], t + 1e-3, side="right")) - 1
    return i if i >= a and abs(paths.ts[i] - t) < 1 else None


print("\nPART 1: the same decision through the simulator")
rows = []
for t in trades.itertuples():
    if t.status != "closed":
        continue
    i = row_of(t.mint, t.decision)
    if i is None:
        print("no row", t.id)
        continue
    o = ex.simulate_position(paths, i, paths.spans[t.mint][1], 1.0, RULE, execution)
    rows.append((t.id, t.s, t.mint[:6], t.pnl_pct, t.exit_reason, o.status, o.reason, o.net_return if o.status == ex.FILLED else None, o.hold_s, (t.closed_at - t.opened_at).total_seconds()))
for s in (5, 6):
    part = [r for r in rows if r[1] == s]
    both = [r for r in part if r[7] is not None]
    d = [r[7] - r[3] for r in both]
    print(
        f"session {s}: {len(part)} trades, {len(both)} filled in the sim | booked mean {st.mean(r[3] for r in both):+.3f} sim mean {st.mean(r[7] for r in both):+.3f}"
        f" | diff median {st.median(d):+.3f} | same reason {sum(r[4] == r[6] for r in both)} | total losses booked {sum(r[3] <= -0.99 for r in both)} sim {sum(r[7] <= -0.99 for r in both)}"
    )
    not_filled = collections.Counter(r[5] for r in part if r[7] is None)
    print("  not filled in the sim:", dict(not_filled), "booked mean of those", round(st.mean([r[3] for r in part if r[7] is None] or [0]), 3))
print("over 25 points apart:")
for r in rows:
    if r[7] is None or abs(r[7] - r[3]) > 0.25:
        print("  ", r[0], f"s{r[1]}", r[2], f"booked {r[3]:+.2f} {r[4]} {r[9]:.0f}s | sim {r[5]} {r[6]} {'' if r[7] is None else f'{r[7]:+.2f}'} {r[8]:.0f}s")

print("\nPART 2: entries over session 6's period, every row over the floor scored offline")
matrix = build_feature_matrix(frame)
traders = pd.to_numeric(frame["unique_traders"], errors="coerce").to_numpy(float)
liq = pd.to_numeric(frame["liquidity"], errors="coerce").to_numpy(float)
start_s, end_s = session6.timestamp(), end.timestamp()
with np.errstate(invalid="ignore"):
    over = (traders >= 20) & (liq >= 5) & (paths.ts >= start_s) & (paths.ts < end_s)
want = []
span_of = {}
for m, (a, b) in paths.spans.items():
    for i in a + np.flatnonzero(over[a:b]):
        i = int(i)
        if i + 1 - a >= predictor.min_seq_len:
            span_of[i] = a
            want.append(i)
print("windows", len(want), flush=True)
pump, ret, rug = np.zeros(len(want)), np.zeros(len(want)), np.zeros(len(want))
for b in range(0, len(want), 256):
    part = want[b : b + 256]
    out = predictor.predict([matrix[max(span_of[i], i + 1 - predictor.seq_len) : i + 1] for i in part], ["15m"])
    for k, p in enumerate(out):
        pump[b + k] = p.predictions[0].pump_prob
        ret[b + k] = p.predictions[0].expected_return
want = np.array(want)
mint_of = frame["mint"].to_numpy(dtype=object)
sel = pump >= 0.5
windows = ex.Windows(mint_of[want[sel]], want[sel], pump[sel], ret[sel], np.ones(int(sel.sum()), bool))
entries = []
busy = {}
for i in np.lexsort((windows.row,)):
    m, r = windows.mint[i], int(windows.row[i])
    if paths.ts[r] < busy.get(m, -1e18):
        continue
    o = ex.simulate_position(paths, r, paths.spans[m][1], float(windows.expected_return[i]), RULE, execution)
    busy[m] = o.free_at
    entries.append((m, r, o, float(windows.pump_prob[i])))
filled = [e for e in entries if e[2].status == ex.FILLED]
print("high windows", int(sel.sum()), "sim entries", len(entries), "filled", len(filled), dict(collections.Counter(e[2].status for e in entries)))
live = trades[(trades["s"] == 6) & (trades["opened_at"] < end)]
print("live trades in the period", len(live), "tokens", live["mint"].nunique(), "| sim tokens", len({e[0] for e in filled}), "| common tokens", len(set(live["mint"]) & {e[0] for e in filled}))
closed = live[live["status"] == "closed"]
print(f"live mean {closed['pnl_pct'].mean():+.3f} losses {(closed['pnl_pct'] <= -0.99).mean():.2f} | sim mean {st.mean(e[2].net_return for e in filled):+.3f} losses {np.mean([e[2].net_return <= -0.99 for e in filled]):.2f}")
matched = 0
live_times = collections.defaultdict(list)
for t in live.itertuples():
    live_times[t.mint].append(t.decision.timestamp())
sim_times = collections.defaultdict(list)
for m, r, o, _ in filled:
    sim_times[m].append(paths.ts[r])
live_only = [t for t in live.itertuples() if not any(abs(x - t.decision.timestamp()) <= 30 for x in sim_times.get(t.mint, []))]
sim_only = [e for e in filled if not any(abs(x - paths.ts[e[1]]) <= 30 for x in live_times.get(e[0], []))]
print(f"within 30 s: {len(live) - len(live_only)} of {len(live)} live trades match | live only {len(live_only)} | sim only {len(sim_only)}")
for t in live_only[:20]:
    print("  live only", t.id, t.mint[:6], t.decision.strftime("%H:%M:%S"), f"pump {t.pump:.3f}", t.status, t.pnl_pct, t.exit_reason)
for m, r, o, p in sim_only[:25]:
    print("  sim only ", m[:6], stamps[r].strftime("%H:%M:%S"), f"pump {p:.3f}", o.reason, f"{o.net_return:+.2f}", "live tokens" if m in live_times else "")
print("=== done", flush=True)
