"""Was the dump visible at entry? Trades of stopdiag-smoke.json (11-05 21:08 .. 11-06 02:11 UTC) against pumpapi's events.

At each entry fill: the token balances of wallets seen so far in the archive (from the transactions' post-balances), the
largest one as a share of the pool's tokens, and what selling it all would do to the price on a constant-product pool.
During the hold: the largest one-block fall, and who did it. Exploratory: these periods are used up for direction A,
so anything found here is a hypothesis to test on data after 2026-11-07 03:10 UTC.
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
trades = [dict(t, rule=k) for k in ("random", "breakout") for t in r[k]]
mints = {t["mint"] for t in trades}

ev = collections.defaultdict(list)
for path in sorted(glob.glob(os.path.join(A, "*.jsonl"))):
    for line in open(path, encoding="utf8"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = e.get("mint")
        if m not in mints:
            continue
        post = {}
        for w, b in (e.get("postBalances") or {}).items():
            if isinstance(b, dict) and m in b:
                post[w] = float(b[m] or 0)
        ev[m].append({"t": e["timestamp"] / 1000, "a": e["action"], "s": e.get("txSigner"), "q": float(e.get("quoteAmount") or 0),
                      "p": float(e.get("priceSol") or 0), "T": float(e.get("tokensInPool") or 0), "pool": e.get("poolId"),
                      "b": e.get("block") or 0, "post": post})
for v in ev.values():
    v.sort(key=lambda x: (x["t"], x["b"]))


def pool_of(es, t, mark):
    best = None
    for p in {e["pool"] for e in es}:
        last = [e for e in es if e["pool"] == p and e["t"] <= t and e["p"] > 0]
        if last and (best is None or abs(last[-1]["p"] / mark - 1) < best[1]):
            best = (p, abs(last[-1]["p"] / mark - 1))
    return best[0] if best and best[1] < 0.05 else None


out = []
for t in trades:
    es = ev.get(t["mint"], [])
    pool = pool_of(es, t["fill_ts"], t["fill_mark"]) if es else None
    if pool is None:
        continue
    before = [e for e in es if e["t"] <= t["fill_ts"]]
    pe_before = [e for e in before if e["pool"] == pool and e["T"] > 0]
    T = pe_before[-1]["T"]
    bal, creators = {}, set()
    for e in before:
        bal.update(e["post"])
        if e["a"] in ("create", "createPool"):
            creators.add(e["s"])
    bal = {w: b for w, b in bal.items() if b > 0 and abs(b / T - 1) > 0.01 and w != pool}
    ranked = sorted(bal.items(), key=lambda x: -x[1])
    top_w, top_b = ranked[0] if ranked else (None, 0.0)
    ratio = top_b / T
    # The hold, on our pool: the largest fall within one block, and its biggest seller.
    hold = [e for e in es if e["pool"] == pool and t["fill_ts"] < e["t"] <= t["used_ts"] and e["p"] > 0]
    prev = pe_before[-1]["p"]
    worst, worst_seller, blocks = 0.0, None, collections.OrderedDict()
    for e in hold:
        blocks.setdefault(e["b"], []).append(e)
    for b, es_b in blocks.items():
        fall = es_b[-1]["p"] / prev - 1
        if fall < worst:
            worst = fall
            sells = [e for e in es_b if e["a"] == "sell"]
            worst_seller = max(sells, key=lambda e: e["q"])["s"] if sells else None
        prev = es_b[-1]["p"]
    out.append({
        "mint": t["mint"], "rule": t["rule"], "reason": t["reason"], "net": t["net"], "first_seen_s": t["fill_ts"] - es[0]["t"],
        "wallets_known": len(bal), "top_ratio": ratio, "dump_fall": 1 - 1 / (1 + ratio) ** 2,
        "top3_ratio": sum(b for _, b in ranked[:3]) / T, "creator_seen": bool(creators),
        "worst_block": worst, "dumper_known_at_entry": worst_seller in bal if worst_seller else None,
        "dumper_is_top": worst_seller == top_w if worst_seller else None,
        "dumper_rank": next((i + 1 for i, (w, _) in enumerate(ranked) if w == worst_seller), None),
        "dumper_is_creator": worst_seller in creators if worst_seller else None,
    })

json.dump(out, open(os.path.join(D, "holders.json"), "w"), indent=1)
print("trades matched", len(out), "of", len(trades))
crash = [x for x in out if x["worst_block"] <= -0.3]
calm = [x for x in out if x["worst_block"] > -0.3]
md = lambda k, xs: round(st.median([x[k] for x in xs]), 3) if xs else None
for name, xs in (("crash (one block <= -30 %)", crash), ("no such block", calm)):
    print(f"\n{name}: {len(xs)} trades, net mean {st.mean(x['net'] for x in xs):+.3f}")
    print("  top wallet / pool tokens median", md("top_ratio", xs), "| its dump would cut price by", md("dump_fall", xs),
          "| top3", md("top3_ratio", xs), "| wallets known", md("wallets_known", xs), "| archive history before entry s", md("first_seen_s", xs))
if crash:
    k = [x for x in crash if x["dumper_known_at_entry"] is not None]
    print("\ncrashes: dumper's balance known at entry", round(sum(x["dumper_known_at_entry"] for x in k) / len(k), 2),
          "| dumper = top wallet", round(sum(bool(x["dumper_is_top"]) for x in k) / len(k), 2),
          "| dumper rank", collections.Counter(x["dumper_rank"] for x in k).most_common(6),
          "| dumper = creator seen", round(sum(bool(x["dumper_is_creator"]) for x in k) / len(k), 2),
          "| creator seen at all", round(sum(x["creator_seen"] for x in crash) / len(crash), 2))
print("\nsplit by the dump a top wallet could do at entry:")
for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)):
    xs = [x for x in out if lo <= x["dump_fall"] < hi]
    if xs:
        print(f"  {lo:.0%}-{hi:.0%}: n {len(xs):>3} crash share {sum(x['worst_block'] <= -0.3 for x in xs) / len(xs):.2f} "
              f"net mean {st.mean(x['net'] for x in xs):+.3f} tokens {len({x['mint'] for x in xs})}")
