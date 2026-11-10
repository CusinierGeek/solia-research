"""Stop exits of stopdiag-smoke.json (11-05 21:08 .. 11-06 02:11 UTC) against pumpapi's real events.

For each stop: does the real price agree with our marks at the fill and at the trigger? When the
price first crossed the stop (fill mark x 0.7), how far below did that one block leave it, and which
transaction did it: size, share of the pool, signer, a buy-and-sell wallet or not?
"""

import collections
import glob
import json
import os
import statistics as st
import sys

D = os.path.dirname(os.path.abspath(__file__))
A = sys.argv[1]
r = json.load(open(os.path.join(D, "stopdiag-smoke.json")))
stops = {(t["mint"], t["fill_ts"]): t for k in ("random", "breakout") for t in r[k] if t["reason"] == "sl"}
mints = {m for m, _ in stops}

ev = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(A, "*.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("mint") in mints and e.get("priceSol"):
            ev[e["mint"]].append((e["timestamp"] / 1000, e["action"], e.get("txSigner"), float(e.get("quoteAmount") or 0),
                                  float(e.get("quoteInPool") or 0), float(e["priceSol"]), e.get("poolId"), e.get("block"), e.get("pool")))
for v in ev.values():
    v.sort(key=lambda x: (x[0], x[7] or 0))


def last_before(es, t):
    out = None
    for e in es:
        if e[0] > t:
            break
        out = e
    return out


rows = []
for (m, fill_ts), t in sorted(stops.items(), key=lambda x: x[0][1]):
    es = ev.get(m, [])
    if not es:
        rows.append({"mint": m, "missing": True})
        continue
    # The pool our marks follow: the one whose price at the fill is closest to our fill mark.
    pools = {e[6] for e in es}
    best = None
    for p in pools:
        b = last_before([e for e in es if e[6] == p], fill_ts)
        if b and (best is None or abs(b[5] / t["fill_mark"] - 1) < abs(best[1] / t["fill_mark"] - 1)):
            best = (p, b[5])
    if best is None:
        rows.append({"mint": m, "missing": True})
        continue
    pe = [e for e in es if e[6] == best[0]]
    level = t["fill_mark"] * 0.7
    cross = next((i for i, e in enumerate(pe) if e[0] > fill_ts and e[5] <= level), None)
    real_trig = last_before(pe, t["trigger_ts"])
    d = {"mint": m, "fill_ts": fill_ts, "net": t["net"], "sim_trigger": t["trigger_mark"] / t["fill_mark"] - 1,
         "real_fill_diff": best[1] / t["fill_mark"] - 1,
         "real_at_trigger": real_trig[5] / t["fill_mark"] - 1 if real_trig else None, "pool": pe[0][8]}
    if cross is not None:
        c = pe[cross]
        block = [e for e in pe if e[7] == c[7]]
        after_block = block[-1][5] / t["fill_mark"] - 1
        before = pe[cross - 1] if cross > 0 else None
        sells_wallets = {e[2] for e in pe if e[1] == "sell"}
        buys_wallets = {e[2] for e in pe if e[1] == "buy"}
        d.update({
            "cross_dt": c[0] - fill_ts, "cross_action": c[1], "cross_sol": c[3],
            "cross_pool_share": c[3] / (c[4] + c[3]) if c[1] == "sell" and c[4] else None,
            "before_cross": before[5] / t["fill_mark"] - 1 if before else None,
            "after_tx": c[5] / t["fill_mark"] - 1, "after_block": after_block, "block_txs": len(block),
            "signer_also_bought": c[2] in buys_wallets, "signer_sells_total": sum(e[3] for e in pe if e[2] == c[2] and e[1] == "sell"),
            "signer_buys_total": sum(e[3] for e in pe if e[2] == c[2] and e[1] == "buy"),
            "cross_vs_trigger_s": t["trigger_ts"] - c[0],
        })
        if c[1] == "sell" and c[2] in sells_wallets:
            first = min(e[0] for e in pe if e[2] == c[2])
            d["signer_first_seen_s_before"] = c[0] - first
    rows.append(d)

json.dump(rows, open(os.path.join(D, "slcheck.json"), "w"), indent=1, default=str)
ok = [x for x in rows if not x.get("missing")]
print("stops", len(rows), "with events", len(ok))
med = lambda k, xs=ok: st.median([x[k] for x in xs if x.get(k) is not None]) if any(x.get(k) is not None for x in xs) else None
print("real vs our fill mark, median diff", round(med("real_fill_diff"), 4),
      "share within 2 %:", round(sum(abs(x["real_fill_diff"]) < 0.02 for x in ok) / len(ok), 2))
print("trigger: sim median", round(med("sim_trigger"), 3), "real price at that time median", round(med("real_at_trigger"), 3))
c = [x for x in ok if "after_block" in x]
print("crossed in archive", len(c))
print(" price before crossing tx median", round(med("before_cross", c), 3), "| after that tx", round(med("after_tx", c), 3),
      "| after its block", round(med("after_block", c), 3), "| sim trigger", round(med("sim_trigger", c), 3))
for lim in (-0.4, -0.5, -0.7):
    print(f"  one block takes it below {lim:.0%}: {sum(x['after_block'] < lim for x in c) / len(c):.2f}")
print(" crossing action", collections.Counter(x["cross_action"] for x in c).most_common())
s = [x for x in c if x["cross_action"] == "sell"]
if s:
    print(" crossing sell: SOL median", round(med("cross_sol", s), 2), "pool share median", round(med("cross_pool_share", s), 3),
          "| signer also bought", round(sum(x["signer_also_bought"] for x in s) / len(s), 2),
          "| signer sold >= its buys", round(sum(x["signer_sells_total"] >= x["signer_buys_total"] for x in s) / len(s), 2),
          "| first seen before (s) median", round(med("signer_first_seen_s_before", s) or 0))
print(" crossing to sim trigger mark, s median", round(med("cross_vs_trigger_s", c), 1), "| block txs median", med("block_txs", c))
