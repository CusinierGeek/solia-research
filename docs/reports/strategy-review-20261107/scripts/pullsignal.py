"""What triggers the drain? Session 5's tokens in pumpapi's archives (2026-11-06 03-19 UTC). Read-only, exploratory.

For each pool emptied by one removal (>= 85 % of its SOL), measured at the removal and, as a baseline, at every buy/sell
event of the same pool between its first trade and the removal:
  age       s since the pool's first event in the archive (createPool when seen)
  growth    SOL in the pool / SOL at its first event
  price     price / first price, and / highest price so far (drawdown from the top)
  flow      buys minus sells (SOL) over the previous 60 s and 300 s, and buy count over 60 s vs the 300 s rate
Where does the removal fall in each token's own distribution (percentile)? A trigger shows up as removals sitting at
the top (or bottom) of their tokens' ranges. Then, per creator with several pools: age and growth at removal.
"""

import bisect
import collections
import glob
import json
import os
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
mints = {m.strip() for m in open(os.path.join(D, "mints.txt")) if m.strip()}
ev = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(D, "h??.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("mint") in mints and e.get("poolId"):
            ev[e["poolId"]].append((e["timestamp"] / 1000, e["action"], float(e.get("quoteAmount") or 0), float(e.get("quoteInPool") or 0),
                                    float(e.get("priceSol") or 0), e.get("txSigner"), e["mint"]))
for v in ev.values():
    v.sort()


class Pool:
    """Prefix sums over one pool's events, so a state costs O(log n)."""

    def __init__(self, es):
        import itertools
        self.es = es
        self.times = [e[0] for e in es]
        self.flow = list(itertools.accumulate(((e[2] if e[1] == "buy" else -e[2]) if e[1] in ("buy", "sell") else 0.0 for e in es), initial=0.0))
        self.buys = list(itertools.accumulate((1 if e[1] == "buy" else 0 for e in es), initial=0))
        top, last_p, last_q, self.top, self.p, self.q = 0.0, 0.0, 0.0, [], [], []
        for e in es:  # values before each event
            self.top.append(top)
            self.p.append(last_p)
            self.q.append(last_q)
            if e[4] > 0:
                top, last_p = max(top, e[4]), e[4]
            if e[3] > 0:
                last_q = e[3]

    def state(self, i, t0, q0, p0):
        t = self.times[i]
        j60, j300 = bisect.bisect_left(self.times, t - 60), bisect.bisect_left(self.times, t - 300)
        b60, b300 = self.buys[i] - self.buys[j60], self.buys[i] - self.buys[j300]
        p, q, top = self.p[i] or p0, self.q[i] or q0, self.top[i] or p0
        return {"age": t - t0, "growth": q / q0 if q0 else None, "price_x": p / p0 if p0 else None, "off_top": p / top - 1 if top else None,
                "flow60": self.flow[i] - self.flow[j60], "flow300": self.flow[i] - self.flow[j300],
                "buy_pace": (b60 / 60) / (b300 / 300) if b300 else None}


rows = []
for pool, es in ev.items():
    rem = [i for i, e in enumerate(es) if e[1] == "remove" and e[3] + e[2] > 0 and e[2] / (e[3] + e[2]) >= 0.85]
    if not rem:
        continue
    k = rem[0]
    first = next((i for i, e in enumerate(es) if e[1] in ("buy", "sell") and e[3] > 0 and e[4] > 0), None)
    if first is None or first >= k - 5:
        continue
    created = next((e for e in es if e[1] == "createPool"), None)
    t0 = created[0] if created else es[first][0]
    q0, p0 = es[first][3], es[first][4]
    P = Pool(es)
    at = P.state(k, t0, q0, p0)
    idx = [i for i in range(first + 1, k) if es[i][1] in ("buy", "sell")]
    step = max(1, len(idx) // 400)
    base = [P.state(i, t0, q0, p0) for i in idx[::step]]
    pct = {}
    for key, v in at.items():
        vals = [b[key] for b in base if b[key] is not None]
        if v is None or len(vals) < 5:
            continue
        pct[key] = sum(x < v for x in vals) / len(vals)
    creator = created[5] if created else None
    rows.append({"pool": pool, "mint": es[0][6], "creator": creator, "remover": es[k][5], "creator_seen": bool(created),
                 "events": k - first, "at": at, "pct": pct, "sol_removed": es[k][2]})

json.dump(rows, open(os.path.join(D, "pullsignal.json"), "w"), indent=1, default=str)
print("pools emptied by one removal", len(rows), "| creator seen", sum(r["creator_seen"] for r in rows),
      "| remover = pool creator", sum(r["creator"] == r["remover"] for r in rows if r["creator"]))
for key in ("age", "growth", "price_x", "off_top", "flow60", "flow300", "buy_pace"):
    vals = [r["at"][key] for r in rows if r["at"].get(key) is not None]
    pcts = [r["pct"][key] for r in rows if key in r["pct"]]
    if not vals:
        continue
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    print(f"{key:<9} at removal: p10 {q(vals, .1):+.3f} med {st.median(vals):+.3f} p90 {q(vals, .9):+.3f} | percentile in own token: "
          f"med {st.median(pcts):.2f}, top 10 % {sum(p >= .9 for p in pcts)}/{len(pcts)}, bottom 10 % {sum(p <= .1 for p in pcts)}/{len(pcts)}")
by = collections.defaultdict(list)
for r in rows:
    by[r["remover"]].append(r)
rep = {w: rs for w, rs in by.items() if len(rs) >= 2}
print("\nremovers with several pools:", len(rep), "covering", sum(len(v) for v in rep.values()), "pools")
for w, rs in sorted(rep.items(), key=lambda x: -len(x[1])):
    print(f"  {w[:8]} pools {len(rs)} age s {[round(r['at']['age']) for r in rs]} growth {[round(r['at']['growth'] or 0, 2) for r in rs]} "
          f"price_x {[round(r['at']['price_x'] or 0, 2) for r in rs]} sol {[round(r['sol_removed']) for r in rs]}")
