"""Would leaving on a rug score beat the booked exit? Exits fall 2 s after the row that fires.

Session 5 up to 19:30 UTC sells on the real pool (pumpapi archives, as in stress.py, within 0.2 point
of booked). The later trades have no archive: they sell at the next row's price less the fee, no
price impact. Rule "new" fires only on a rise: the entry's own rug score was under the threshold.
"""

import bisect
import collections
import glob
import json
import os
import random
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(D, "rugscore.json")))
mints = {t["mint"] for t in data}
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
ARCHIVE_END = 1793990000  # replaced below


def state_at(v, t):
    i = bisect.bisect_right(v, (t, float("inf"), float("inf"))) - 1
    return v[i] if i >= 0 else None


def pool_sale(t, when):
    """Share of the stake back when selling at ``when`` in the trade's pool; None without archive."""
    by_pool = pools.get(t["mint"])
    if not by_pool or t["c"] > last_event:
        return None
    pid = max(by_pool, key=lambda p: (state_at(by_pool[p], t["c"]) or (-1,))[0])
    s = state_at(by_pool[pid], when)
    if s is None:
        return None
    _, r, tok = s
    if r < 1 or tok <= 0:
        return 0.0
    q = t["sz"] / t["ep"]
    return r * q / (tok + q) * (1 - 0.0025) / t["sz"]


last_event = max(v[-1][0] for b in pools.values() for v in b.values())


def row_sale(t, i):
    rows = t["rows"]
    j = min(i + 1, len(rows) - 1)
    return 0.0 if rows[j][4] < 1 else rows[j][3] / t["ep"] * (1 - 0.0025)


def result(t, thr, only_rise):
    if only_rise and t["rug0"] is not None and t["rug0"] >= thr:
        return t["pnl"], False, None
    for i, row in enumerate(t["rows"]):
        if row[1] >= thr:
            back = pool_sale(t, row[0] + 2)
            if back is None:
                back = row_sale(t, i)
            return back - 1, True, row[0] - t["o"]
    return t["pnl"], False, None


def ci(values):
    by = collections.defaultdict(list)
    for m, v in values:
        by[m].append(v)
    keys = list(by)
    means = []
    for _ in range(4000):
        pick = [v for m in random.choices(keys, k=len(keys)) for v in by[m]]
        means.append(sum(pick) / len(pick))
    means.sort()
    return means[100], means[3899]


random.seed(0)
for name, part in (
    ("session 5, archive", [t for t in data if t["session"] == 5 and t["c"] <= last_event]),
    ("session 5, after the archive (row prices)", [t for t in data if t["session"] == 5 and t["c"] > last_event]),
    ("session 6 (row prices)", [t for t in data if t["session"] == 6]),
):
    if not part:
        continue
    base = st.mean(t["pnl"] for t in part)
    print(f"\n{name}: n={len(part)} tokens={len({t['mint'] for t in part})} booked mean {base:+.3f}, total losses {sum(t['pnl'] <= -0.99 for t in part)}, wins {sum(t['pnl'] > 0 for t in part)}")
    print("rule           fired  mean    diff vs booked [95% CI]  losses wins  saved  winners_cut  med_s")
    for thr in (0.7, 0.8, 0.9):
        for rise in (False, True):
            res = [result(t, thr, rise) for t in part]
            diffs = [(t["mint"], r[0] - t["pnl"]) for t, r in zip(part, res, strict=True)]
            lo, hi = ci(diffs)
            saved = sum(t["pnl"] <= -0.99 and r[0] > -0.99 for t, r in zip(part, res, strict=True))
            cut = sum(t["pnl"] > 0.5 and r[0] < t["pnl"] - 0.2 for t, r in zip(part, res, strict=True))
            times = [r[2] for r in res if r[1]]
            print(
                f"rug>={thr} {'new ' if rise else 'any '}  {sum(r[1] for r in res):>4} {st.mean(r[0] for r in res):+.3f} "
                f"{st.mean(d for _, d in diffs):+.3f} [{lo:+.3f},{hi:+.3f}] {sum(r[0] <= -0.99 for r in res):>5} {sum(r[0] > 0 for r in res):>4}"
                f" {saved:>6} {cut:>6} {st.median(times) if times else 0:>9.0f}"
            )
