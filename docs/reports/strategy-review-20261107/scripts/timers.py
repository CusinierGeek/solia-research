"""Do drainers run on timers? pumpapi archives of 2026-11-08 (pool creations, liquidity events only). Read-only, exploratory.

A drain = the first removal taking >= 85 % of a pool's SOL, by the wallet that created the pool (createPool seen). Life =
drain time - pool creation. Per wallet with several drains: how tight are its lives (each life vs the wallet's median), against
the same wallets with lives shuffled between them? Then, in time order: does the median of a wallet's earlier lives predict the
next one? An exit at 80 % of that prediction would leave before the drain how often, and after how long a hold?
Cross-day: wallets that also drained on 11-06 (pullsignal.json), their lives then and now.
"""

import collections
import glob
import json
import os
import random
import statistics as st
import sys

D = os.path.dirname(os.path.abspath(__file__))
A = sys.argv[1]
MIN_SEED = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0  # e.g. 50: pools launched like the bot's targets
created, drains = {}, {}
for path in sorted(glob.glob(os.path.join(A, "*.jsonl"))):
    for line in open(path, encoding="utf8", errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        pool, t = e.get("poolId"), e["timestamp"] / 1000
        if not pool:
            continue
        if e["action"] == "createPool" and pool not in created:
            created[pool] = (t, e.get("txSigner"), float(e.get("quoteInPool") or 0), e.get("pool"), e.get("mint"))
        elif e["action"] == "remove" and pool not in drains:
            q, out = float(e.get("quoteInPool") or 0), float(e.get("quoteAmount") or 0)
            if q + out > 0 and out / (q + out) >= 0.85:
                drains[pool] = (t, e.get("txSigner"), out)
rows = []
for pool, (t, w, out) in drains.items():
    c = created.get(pool)
    if c and c[1] == w and t > c[0] and c[2] >= MIN_SEED:
        rows.append({"pool": pool, "wallet": w, "created": c[0], "life": t - c[0], "seed": c[2], "out": out, "venue": c[3], "mint": c[4]})
print("min seed", MIN_SEED, "| pools created", len(created), "| drained (>= 85 % in one removal)", len(drains), "| by their creator, creation seen", len(rows))
print("share of created pools drained by their creator:", round(len(rows) / max(1, len(created)), 3))
by = collections.defaultdict(list)
for r in sorted(rows, key=lambda r: r["created"]):
    by[r["wallet"]].append(r)
rep = {w: rs for w, rs in by.items() if len(rs) >= 3}
print("wallets with >= 3 drains", len(rep), "covering", sum(len(v) for v in rep.values()), "drains of", len(rows))


def spread(groups):
    """Median over drains of |life / own wallet's median - 1|."""
    errs = []
    for lives in groups:
        m = st.median(lives)
        errs += [abs(x / m - 1) for x in lives]
    return st.median(errs) if errs else None


groups = [[r["life"] for r in rs] for rs in rep.values()]
real = spread(groups)
pool_lives = [x for g in groups for x in g]
rng = random.Random(0)
shuffled = []
for _ in range(200):
    rng.shuffle(pool_lives)
    k, gs = 0, []
    for g in groups:
        gs.append(pool_lives[k : k + len(g)])
        k += len(g)
    shuffled.append(spread(gs))
print(f"life vs own median, median gap: {real:.3f} | shuffled between wallets: {st.median(shuffled):.3f} (p2.5 {sorted(shuffled)[5]:.3f})")
print("lives (s), all drains: p10/50/90", [round(sorted(r["life"] for r in rows)[int(p * (len(rows) - 1))]) for p in (.1, .5, .9)])
for w, rs in sorted(rep.items(), key=lambda x: -len(x[1]))[:12]:
    lives = [round(r["life"]) for r in rs]
    print(f"  {w[:8]} n {len(rs):>3} median {st.median(lives):>6.0f} s  lives {lives[:14]}")

# Prediction in time order: median of the wallet's earlier lives (at least 2).
hit, n, holds, errs = 0, 0, [], []
for rs in by.values():
    for k in range(2, len(rs)):
        pred = st.median(r["life"] for r in rs[:k])
        life = rs[k]["life"]
        n += 1
        errs.append(life / pred - 1)
        if 0.8 * pred < life:
            hit += 1
            holds.append(0.8 * pred)
if n:
    print(f"\npredicted from earlier drains: {n} drains | life / prediction - 1: median {st.median(errs):+.3f}, "
          f"within +-20 %: {sum(abs(x) <= .2 for x in errs) / n:.2f}")
    print(f"exit at 80 % of the prediction leaves before the drain on {hit / n:.2f} of them, after a median {st.median(holds):.0f} s since creation")
    first_seen = sum(len(rs) >= 3 for rs in by.values())
    print("drains by a wallet with >= 2 earlier drains that day:", n, "of", len(rows), f"({n / len(rows):.2f})")

prev = os.path.join(D, "pullsignal.json")
if os.path.exists(prev):
    old = collections.defaultdict(list)
    for r in json.load(open(prev)):
        old[r["remover"]].append(r["at"]["age"])
    both = [w for w in old if w in by]
    print("\nwallets draining on 11-06 and 11-08:", len(both))
    for w in both:
        print(f"  {w[:8]} 11-06 {[round(x) for x in old[w]]}  11-08 {[round(r['life']) for r in by[w]][:12]}")
json.dump(rows, open(os.path.join(D, f"timers-seed{MIN_SEED:g}.json"), "w"), indent=1)
