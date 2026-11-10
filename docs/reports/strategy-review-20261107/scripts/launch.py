"""How session 5's 45 traded tokens were launched: the pools created in their first minutes."""

import collections
import glob
import json
import os
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
mints = set(open(os.path.join(D, "mints.txt")).read().split())
made = collections.defaultdict(list)
keys = None
size = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(D, "h??.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("mint")
        if m not in mints:
            continue
        a = e.get("action")
        if a in ("create", "createPool", "migrate"):
            if keys is None:
                keys = sorted(k for k in e if k not in ("preBalances", "postBalances", "transfers"))
            made[m].append((e["timestamp"] / 1000, a, e.get("pool"), e.get("poolId"), e.get("txSigner"), e.get("quoteInPool"), e.get("initialBuy"), e.get("marketCapSol")))
        elif a in ("buy", "sell") and e.get("quoteAmount"):
            size[m].append(e["quoteAmount"])
print("event keys", keys)
print("tokens with a pool event", len(made), "of", len(mints))
shapes = collections.Counter()
seeds = []
for m, ev in sorted(made.items(), key=lambda x: x[1][0][0]):
    ev.sort()
    t0 = ev[0][0]
    shape = " > ".join(f"{a}:{p}" for _, a, p, *_ in ev[:3])
    shapes[shape] += 1
    signers = {e[4] for e in ev}
    q = [e[5] for e in ev if e[1] == "createPool"]
    seeds += q[:1]
    s = size[m]
    print(
        m[:8], shape, "| dt", [round(e[0] - t0, 1) for e in ev[:3]], "| signers", len(signers),
        "| quote", [None if e[5] is None else round(e[5], 1) for e in ev[:3]],
        "| trades", len(s), "median_sol", round(st.median(s), 3) if s else None,
    )
print(shapes.most_common())
print("first createPool quote in pool: median", round(st.median(seeds), 1) if seeds else None, "values", sorted(round(x, 1) for x in seeds))
