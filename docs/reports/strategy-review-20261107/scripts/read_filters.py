"""Applies filters_bt.py's reading. Usage: read_filters.py choose a.json b.json | read_filters.py confirm "<cell>" c.json"""

import json
import sys

import numpy as np

CELLS = {
    "top10 <= 0.3": lambda t: t["hs_top10_share"] is not None and t["hs_top10_share"] <= 0.3,
    "top10 <= 0.5": lambda t: t["hs_top10_share"] is not None and t["hs_top10_share"] <= 0.5,
    "top10 <= 0.7": lambda t: t["hs_top10_share"] is not None and t["hs_top10_share"] <= 0.7,
    "top1 <= 0.1": lambda t: t["hs_top1_share"] is not None and t["hs_top1_share"] <= 0.1,
    "top1 <= 0.2": lambda t: t["hs_top1_share"] is not None and t["hs_top1_share"] <= 0.2,
    "top10 flow >= -0.1": lambda t: t["hs_top10_net_flow"] is not None and t["hs_top10_net_flow"] >= -0.1,
    "top10 flow >= 0": lambda t: t["hs_top10_net_flow"] is not None and t["hs_top10_net_flow"] >= 0,
    "X page": lambda t: t["meta_has_twitter"] is not None and t["meta_has_twitter"] >= 1,
    "narrative >= 1": lambda t: t["meta_narrative_grads_3h"] is not None and t["meta_narrative_grads_3h"] >= 1,
}
rng = np.random.default_rng(0)


def stats(ts):
    if not ts:
        return {"n": 0}
    r = np.array([t["net"] for t in ts])
    mints = sorted({t["mint"] for t in ts})
    by = {m: [] for m in mints}
    for t in ts:
        by[t["mint"]].append(t["net"])
    groups = [np.array(by[m]) for m in mints]
    boot = []
    for _ in range(2000):
        pick = rng.integers(len(groups), size=len(groups))
        boot.append(np.concatenate([groups[k] for k in pick]).mean())
    return {"n": len(r), "tokens": len(mints), "mean": float(r.mean()), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "median": float(np.median(r)), "total_loss": float((r <= -0.99).mean()), "tp": float(np.mean([t["reason"] == "tp" and t["net"] > 0 for t in ts]))}


def show(name, s):
    if not s["n"]:
        print(f"  {name:<20} no trades")
        return
    print(f"  {name:<20} n {s['n']:>4} tok {s['tokens']:>4} mean {s['mean']:+.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}] med {s['median']:+.3f} "
          f"loss {s['total_loss']:.2f} tp {s['tp']:.2f}")


mode = sys.argv[1]
if mode == "choose":
    trades = [t for f in sys.argv[2:] for t in json.load(open(f))["trades"]]
    trades = list({(t["mint"], t["ts"]): t for t in trades if t["ts"] >= 1793714400}.values())  # from 2026-11-03 14:00 UTC
    known = {c: sum(t[c] is not None for t in trades) for c in ("hs_top10_share", "hs_top10_net_flow", "meta_has_twitter", "meta_narrative_grads_3h")}
    print("trades", len(trades), "known", known)
    show("all (bot)", stats(trades))
    res = {}
    for name, keep in CELLS.items():
        k = [t for t in trades if keep(t)]
        res[name] = stats(k)
        show(name, res[name])
        show("  rejected", stats([t for t in trades if not keep(t)]))
    ok = {n: s for n, s in res.items() if s["n"] >= 100}
    best = max(ok, key=lambda n: ok[n]["mean"]) if ok else None
    print("CHOSEN:", best)
else:
    cell, trades = sys.argv[2], json.load(open(sys.argv[3]))["trades"]
    allb, kept = stats(trades), stats([t for t in trades if CELLS[cell](t)])
    show("all (bot)", allb)
    show(cell, kept)
    a, b, c = kept["n"] >= 100, kept["n"] > 0 and kept["ci"][0] > 0, kept["n"] > 0 and kept["mean"] >= allb["mean"] + 0.05
    print(">=100", a, "CI>0", b, "+5 pts", c, "VERDICT:", "passes" if a and b and c else "fails")
