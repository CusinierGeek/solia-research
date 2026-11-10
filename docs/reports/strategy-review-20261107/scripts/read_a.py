"""Applies direction A's reading rule, fixed before the runs."""

import json
import os

D = os.path.dirname(os.path.abspath(__file__))
R = {p: json.load(open(os.path.join(D, f"a-{p}.json"))) for p in ("p1", "p2", "fresh")}
for p, r in R.items():
    print(f"\n{p}: {r['read']['window_start'][:16]} .. {r['read']['window_end'][:16]} population rows {r['population_rows']}")
    for e, s in r["random"].items():
        print(f"  random {e:<10} n {s['n']:>6.0f} mean {s['mean']:+.3f} band [{s['band'][0]:+.3f},{s['band'][1]:+.3f}] loss {s['total_loss']:.2f}")
    for name, s in r["cells"].items():
        if s["n"]:
            print(f"  {name:<22} n {s['n']:>5} tok {s['tokens']:>4} mean {s['mean']:+.3f} [{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}] med {s['median']:+.3f} win {s['win']:.2f} loss {s['total_loss']:.2f} tp {s['tp']:.2f} hold {s['hold_s']:.0f}s")
        else:
            print(f"  {name:<22} no trades")
cells = {k: v for k, v in R["p1"]["cells"].items() if v["n"]}
best = max(cells, key=lambda k: cells[k]["mean"])
print("\nchosen on p1:", best, cells[best]["mean"])
exit_name = best.split(" | ")[1]
passes = True
for p in ("p2", "fresh"):
    s = R[p]["cells"][best]
    base = R[p]["random"][exit_name]["mean"]
    a, b, c = s["n"] >= 100, s["n"] > 0 and s["ci"][0] > 0, s["n"] > 0 and s["ci"][0] > base
    passes &= a and b and c
    print(p, "n", s["n"], "mean", s.get("mean"), "ci", s.get("ci"), "random", round(base, 3), "| >=100:", a, "CI>0:", b, "above random:", c)
print("VERDICT:", "passes" if passes else "no edge shown")
