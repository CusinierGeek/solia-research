"""Session 5's closed trades, exits replayed later on the real pool state from pumpapi's archives.

The sale runs on the state after the last event of the trade's pool at or before the exit time:
SOL out = R * q / (T + q) * (1 - fee), q = size / entry price, in the pool the trade was in, so a
drained pool pays nothing. Entries stay as booked. Check: delay 0 against the booked result.
"""

import bisect
import collections
import csv
import glob
import io
import json
import os
import random
import statistics as st
import subprocess

D = os.path.dirname(os.path.abspath(__file__))
SQL = """copy (select id, mint, extract(epoch from closed_at) c, entry_price::float8 ep, size::float8 sz,
 pnl_pct, exit_reason from trades where session_id=5 and status='closed' and closed_at<'2026-11-06 19:30+00' order by id)
 to stdout csv header"""
out = subprocess.run(
    ["docker", "exec", "-i", "solia-postgres-1", "sh", "-c", "psql -U $POSTGRES_USER -d $POSTGRES_DB -q"],
    input=SQL, capture_output=True, text=True, check=True,
).stdout
trades = list(csv.DictReader(io.StringIO(out)))
mints = {t["mint"] for t in trades}

pools = collections.defaultdict(lambda: collections.defaultdict(list))
for path in sorted(glob.glob(os.path.join(D, "h??.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("mint")
        if m in mints and e.get("action") in ("buy", "sell", "add", "remove", "create", "createPool", "migrate"):
            if e.get("quoteInPool") is not None and e.get("tokensInPool") is not None and e.get("poolId"):
                pools[m][e["poolId"]].append((e["timestamp"] / 1000, e["quoteInPool"], e["tokensInPool"]))
for by_pool in pools.values():
    for v in by_pool.values():
        v.sort()


def trade_pool(mint, t):
    """The pool with the latest event at or before ``t``."""
    best, best_ts = None, -1.0
    for pid, v in pools[mint].items():
        i = bisect.bisect_right(v, (t, float("inf"), float("inf"))) - 1
        if i >= 0 and v[i][0] > best_ts:
            best, best_ts = pid, v[i][0]
    return best


def sale(mint, pid, t, q, fee):
    v = pools[mint][pid]
    i = bisect.bisect_right(v, (t, float("inf"), float("inf"))) - 1
    if i < 0:
        return 0.0
    _, r, tok = v[i]
    if r < 1 or tok <= 0:  # the bot's own rule for an empty pool
        return 0.0
    return r * q / (tok + q) * (1 - fee)


DELAYS = (0, 5, 10, 20, 30, 60, 120)
FEE = 0.0025
rows = []
for t in trades:
    c, ep, sz = float(t["c"]), float(t["ep"]), float(t["sz"])
    pid = trade_pool(t["mint"], c)
    if pid is None:
        continue
    rows.append({
        "id": t["id"], "mint": t["mint"], "booked": float(t["pnl_pct"]), "why": t["exit_reason"], "pid": pid,
        "r": {
            d: {
                "real": sale(t["mint"], pid, c + d, sz / ep, FEE) / sz - 1,
                "small": sale(t["mint"], pid, c + d, 0.1 / ep, FEE) / 0.1 - 1,
                "fees": sale(t["mint"], pid, c + d, sz / ep, FEE + 0.01) / sz - 1,
            }
            for d in DELAYS
        },
    })
print("trades replayed", len(rows), "of", len(trades))
gaps = [r["r"][0]["real"] - r["booked"] for r in rows]
print(
    "check, delay 0 minus booked: median %+.3f p10 %+.3f p90 %+.3f; mean booked %+.3f, replay %+.3f"
    % (st.median(gaps), sorted(gaps)[len(gaps) // 10], sorted(gaps)[9 * len(gaps) // 10],
       st.mean(r["booked"] for r in rows), st.mean(r["r"][0]["real"] for r in rows))
)
far = [(r["id"], r["why"], round(r["booked"], 2), round(r["r"][0]["real"], 2)) for r in rows if abs(r["r"][0]["real"] - r["booked"]) > 0.15]
print("over 15 points from booked:", far)

random.seed(0)
by_token = collections.defaultdict(list)
for r in rows:
    by_token[r["mint"]].append(r)
tokens = list(by_token)


def ci(value):
    means = []
    for _ in range(4000):
        pick = [x for m in random.choices(tokens, k=len(tokens)) for x in by_token[m]]
        means.append(sum(value(x) for x in pick) / len(pick))
    means.sort()
    return means[100], means[3899]


def table(name, key, part=None):
    part = part or rows
    print(f"\n{name}")
    print("delay   mean           95% CI     win  zero  drained since delay 0")
    base = [x for x in part if x["r"][0][key] > -0.999]
    for d in DELAYS:
        v = [x["r"][d][key] for x in part]
        lo, hi = ci(lambda x: x["r"][d][key]) if part is rows else (float("nan"), float("nan"))
        print(
            f"{d:>4}s {st.mean(v):+7.3f} [{lo:+.3f},{hi:+.3f}] {sum(a > 0 for a in v):>4} {sum(a <= -0.999 for a in v):>5}"
            f" {sum(x['r'][d][key] <= -0.999 for x in base):>5}"
        )


table("all trades, booked size", "real")
table("all trades, 0.1 SOL stake (separates size from timing)", "small")
table("all trades, booked size, 1 point more fee on the sale", "fees")
table("winners only, booked size", "real", [r for r in rows if r["booked"] > 0])
winners_lost = [(r["id"], round(r["booked"], 2)) for r in rows if r["booked"] > 0 and r["r"][30]["real"] <= -0.999]
print("\nwinners paying nothing 30 s later:", winners_lost)
json.dump(rows, open(os.path.join(D, "stress.json"), "w"))
