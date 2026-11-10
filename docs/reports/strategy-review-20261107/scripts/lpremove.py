"""How liquidity left the pools of session 5's closed trades (pumpapi archives, 2026-11-06 03-19 UTC). Read-only.

Per trade, on its mint's pools: the liquidity removals after the entry. Was the pool emptied in one transaction, or did a
smaller removal come first, and how long before? Could an exit at the first removal (+ 2 s latency) have beaten the drain?
"""

import collections
import csv
import glob
import io
import json
import os
import statistics as st
import subprocess

D = os.path.dirname(os.path.abspath(__file__))
SQL = """copy (select id, mint, extract(epoch from opened_at) o, extract(epoch from closed_at) c, pnl_pct::float8 pnl, exit_reason
 from trades where session_id=5 and status='closed' and closed_at<'2026-11-06 19:30+00' order by id) to stdout csv header"""
out = subprocess.run(["docker", "exec", "-i", "solia-postgres-1", "sh", "-c", "psql -U $POSTGRES_USER -d $POSTGRES_DB -q"],
                     input=SQL, capture_output=True, text=True, check=True).stdout
trades = list(csv.DictReader(io.StringIO(out)))
mints = {t["mint"] for t in trades}
ev = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(D, "h??.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("mint") in mints and e.get("action") in ("buy", "sell", "remove", "add"):
            ev[e["mint"]].append((e["timestamp"] / 1000, e["action"], float(e.get("quoteAmount") or 0), float(e.get("quoteInPool") or 0),
                                  e.get("poolId"), e.get("txSigner"), float(e.get("priceSol") or 0)))
for v in ev.values():
    v.sort()

rows = []
for t in trades:
    o, c, pnl = float(t["o"]), float(t["c"]), float(t["pnl"])
    es = ev.get(t["mint"], [])
    rem = [e for e in es if e[1] == "remove" and e[0] >= o]
    if not rem:
        rows.append({"id": t["id"], "pnl": pnl, "removes": 0})
        continue
    pool = rem[0][4]
    pr = [e for e in rem if e[4] == pool]
    first = pr[0]
    before = first[3] + first[2]
    drained = next((e for e in pr if e[3] < 1), None)
    big = max(pr, key=lambda e: e[2])
    # price 2 s after the first removal, vs at our entry, on that pool
    pe = [e for e in es if e[4] == pool and e[6] > 0]
    entry_p = next((e[6] for e in pe if e[0] >= o), None)
    at_first2 = [e for e in pe if e[0] <= first[0] + 2]
    rows.append({
        "id": t["id"], "pnl": pnl, "reason": t["exit_reason"], "removes": len(pr), "first_after_entry_s": first[0] - o,
        "first_vs_exit_s": first[0] - c, "first_share": first[2] / before if before else None, "first_sol": first[2],
        "big_share": big[2] / (big[3] + big[2]) if big[2] else None, "warning_s": big[0] - first[0] if first is not big else 0.0,
        "drained_after_first_s": (drained[0] - first[0]) if drained else None,
        "same_signer": len({e[5] for e in pr}) == 1,
        "exit_at_first_vs_entry": (at_first2[-1][6] / entry_p - 1) if at_first2 and entry_p else None,
    })

json.dump(rows, open(os.path.join(D, "lpremove.json"), "w"), indent=1)
with_r = [r for r in rows if r["removes"]]
print("trades", len(rows), "with a removal after entry", len(with_r))
loss = [r for r in with_r if r["pnl"] <= -0.99]
win = [r for r in with_r if r["pnl"] > 0]
for name, xs in (("total losses", loss), ("paying", win), ("all", with_r)):
    if not xs:
        continue
    one = sum(r["removes"] == 1 or (r["first_share"] or 0) >= 0.9 for r in xs)
    print(f"\n{name}: {len(xs)}")
    print(f"  emptied by the first removal (>= 90 %): {one} | removals per pool median {st.median(r['removes'] for r in xs)}")
    warned = [r for r in xs if (r["first_share"] or 0) < 0.9]
    if warned:
        print(f"  first removal small: {len(warned)}, its share median {st.median(r['first_share'] for r in warned):.3f}, "
              f"time to the big one median {st.median(r['warning_s'] for r in warned):.0f} s, min {min(r['warning_s'] for r in warned):.0f} s")
    print(f"  first removal after entry, s, median {st.median(r['first_after_entry_s'] for r in xs):.0f}; vs our exit, s, median {st.median(r['first_vs_exit_s'] for r in xs):.0f}")
    print(f"  removal began before our exit: {sum(r['first_vs_exit_s'] < 0 for r in xs)}")
    ex_ = [r["exit_at_first_vs_entry"] for r in xs if r["exit_at_first_vs_entry"] is not None and r["first_vs_exit_s"] < 0]
    if ex_:
        print(f"  price 2 s after the first removal vs our entry (removal during the hold): median {st.median(ex_):+.3f}, n {len(ex_)}")
for r in loss:
    print("  loss", r["id"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("id",)})
