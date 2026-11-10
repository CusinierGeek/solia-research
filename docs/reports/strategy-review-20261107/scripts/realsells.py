"""Session 5's closed trades against the real events of pumpapi's archives (2026-11-06, 03-19 UTC).

For each simulated exit: did real sells of a similar size go through, near that time and price? Who
created the pool, who removed its liquidity, and how concentrated was the trading?
Only reads the filtered archive lines and the trades table.
"""

import collections
import csv
import glob
import io
import json
import os
import statistics as st
import subprocess
import sys

D = os.path.dirname(os.path.abspath(__file__))
C = "solia-postgres-1"
SQL = """copy (select id, mint, extract(epoch from opened_at) o, extract(epoch from closed_at) c,
 entry_price::float8 ep, exit_price::float8 xp, size::float8 sz, pnl::float8 pnl, pnl_pct, exit_reason
 from trades where session_id=5 and status='closed' and closed_at<'2026-11-06 19:30+00' order by id) to stdout csv header"""
out = subprocess.run(
    ["docker", "exec", "-i", C, "sh", "-c", 'psql -U $POSTGRES_USER -d $POSTGRES_DB -q'],
    input=SQL, capture_output=True, text=True, check=True,
).stdout
trades = list(csv.DictReader(io.StringIO(out)))
mints = {t["mint"] for t in trades}

events = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(D, "h??.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("mint")
        if m in mints and e.get("action") in ("buy", "sell", "add", "remove", "create", "createPool", "migrate"):
            events[m].append((
                e["timestamp"] / 1000, e["action"], e.get("txSigner"), e.get("quoteAmount") or 0.0,
                e.get("priceSol") or e.get("price"), e.get("quoteInPool"), e.get("poolId"),
            ))
for v in events.values():
    v.sort(key=lambda x: x[0])
print("trades", len(trades), "tokens with events", len(events), "events", sum(map(len, events.values())))

rows = []
creators = collections.Counter()
removers = collections.Counter()
for t in trades:
    tid, mint = t["id"], t["mint"]
    o, c, ep, xp, sz, pnl = (float(t[k]) for k in ("o", "c", "ep", "xp", "sz", "pnl"))
    proceeds = max(sz + pnl, 0.0)
    ev = events.get(mint, [])
    near = [e for e in ev if e[1] == "sell" and abs(e[0] - c) <= 30]
    big = [e for e in near if e[3] >= 0.5 * proceeds and proceeds > 0]
    full = [e for e in near if e[3] >= proceeds and proceeds > 0]
    after = [e for e in ev if e[1] == "sell" and c <= e[0] <= c + 300]
    last_sell = max((e[0] for e in ev if e[1] == "sell"), default=None)
    closest = min(near, key=lambda e: abs(e[0] - c), default=None)
    gap = None if not closest or not closest[4] or not xp else closest[4] / xp - 1
    before = [e for e in ev if e[1] in ("buy", "sell") and e[0] <= c and e[5] is not None]
    pool = before[-1][5] if before else None
    made = [e for e in ev if e[1] in ("create", "createPool")]
    creator = made[0][2] if made else None
    removes = [e for e in ev if e[1] == "remove" and e[3] > 1]
    held = [e for e in ev if e[1] in ("buy", "sell") and o - 120 <= e[0] <= c]
    vol = collections.Counter()
    sides = collections.defaultdict(set)
    for e in held:
        vol[e[2]] += e[3]
        sides[e[2]].add(e[1])
    tot = sum(vol.values()) or 1.0
    top5 = sum(v for _, v in vol.most_common(5)) / tot
    both = sum(v for s, v in vol.items() if len(sides[s]) == 2) / tot
    rm_first = removes[0] if removes else None
    rm_by_creator = bool(rm_first and creator and rm_first[2] == creator)
    if creator:
        creators[creator] += 0
    rows.append({
        "id": tid, "mint": mint[:8], "pct": float(t["pnl_pct"]), "why": t["exit_reason"], "proceeds": proceeds,
        "n_near": len(near), "n_big": len(big), "n_full": len(full), "max_near": max((e[3] for e in near), default=0.0),
        "gap": gap, "pool": pool, "after": len(after), "last_sell_min": None if last_sell is None else (last_sell - c) / 60,
        "creator": creator, "n_remove": len(removes),
        "rm_min": None if not rm_first else (rm_first[0] - c) / 60, "rm_sol": rm_first[3] if rm_first else None,
        "rm_by_creator": rm_by_creator, "rm_signer": rm_first[2] if rm_first else None,
        "signers": len(vol), "top5": top5, "both": both, "has_events": bool(ev),
    })

byc = {}
for r in {x["mint"]: x for x in rows}.values():
    if r["creator"]:
        creators[r["creator"]] += 1
    if r["rm_signer"]:
        removers[r["rm_signer"]] += 1

fmt = lambda x, f="{:+.0f}": "n/a" if x is None else f.format(x)
print("id pnl why | proceeds near30s big full maxsell | price_vs_exit | pool | rm_min rm_sol by_creator | signers top5 both")
for r in rows:
    print(
        r["id"], f'{r["pct"]:+.2f}', r["why"], "|", f'{r["proceeds"]:.2f}', r["n_near"], r["n_big"], r["n_full"],
        f'{r["max_near"]:.2f}', "|", fmt(None if r["gap"] is None else 100 * r["gap"]), "|", fmt(r["pool"], "{:.0f}"), "|",
        fmt(r["rm_min"], "{:+.1f}"), fmt(r["rm_sol"], "{:.0f}"), r["rm_by_creator"], "|", r["signers"],
        f'{100 * r["top5"]:.0f}', f'{100 * r["both"]:.0f}', r["mint"],
    )


def summary(name, part):
    if not part:
        return
    payers = [r for r in part if r["proceeds"] > 0.05]
    gaps = [r["gap"] for r in payers if r["gap"] is not None]
    print(
        f"\n{name}: n={len(part)} paying exits={len(payers)}",
        f"| real sell within 30s: {sum(r['n_near'] > 0 for r in payers)}",
        f"| at >=50% of our size: {sum(r['n_big'] > 0 for r in payers)}",
        f"| at >=100%: {sum(r['n_full'] > 0 for r in payers)}",
        f"| price gap median {100 * st.median(gaps):+.1f}% p10 {100 * sorted(gaps)[len(gaps) // 10]:+.1f}% p90 {100 * sorted(gaps)[9 * len(gaps) // 10]:+.1f}%" if gaps else "",
    )
    print(
        "  liquidity removed:", sum(r["n_remove"] > 0 for r in part),
        "by the pool's creator:", sum(r["rm_by_creator"] for r in part),
        "| within 15 min after our exit:", sum(r["rm_min"] is not None and 0 <= r["rm_min"] <= 15 for r in part),
        "| median signers", st.median(r["signers"] for r in part),
        "top5 share", round(100 * st.median(r["top5"] for r in part)),
        "both-sides share", round(100 * st.median(r["both"] for r in part)),
    )


known = [r for r in rows if r["has_events"]]
summary("all", known)
summary("winners", [r for r in known if r["pct"] > 0])
summary("tp", [r for r in known if r["why"] == "tp" and r["pct"] > 0])
summary("losers", [r for r in known if r["pct"] <= 0])
print("\nno archive events:", [r["id"] for r in rows if not r["has_events"]])
print("creators with >1 token:", [(k[:8], v) for k, v in creators.most_common(8) if v > 1], "of", len(creators))
print("removers with >1 token:", [(k[:8], v) for k, v in removers.most_common(8) if v > 1], "of", len(removers))
json.dump(rows, open(os.path.join(D, "realsells.json"), "w"), indent=1)
