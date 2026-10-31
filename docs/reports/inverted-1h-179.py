"""#179 read-only analysis: the candidate's top-1% 1h pump windows (one-off, no registry writes).

Run inside the GPU ai-service container: python -u inverted-1h-179.py <checkpoint.pt> <out.json>.
The token G is the one the analysis found dominating the rug<=0.2 selection (inverted-1h-179.txt).
"""
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

from solia_ai.config import load_config
from solia_ai.dataset import SequenceDataset, config_samples, embargo_seconds
from solia_ai.db import DataSettings, make_engine, read_training_features
from solia_ai.evaluate import apply_calibration, score_logits
from solia_ai.inference import Predictor
from solia_ai.liquidity import LiquidityFloor, liquidity_buckets

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/app/models/solia-20261030T203422233817Z.pt"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/app/models/reports/inverted-1h-179.json"

pred = Predictor.from_checkpoint(CKPT)
pred.model.to("cuda")
cfg = {**load_config(), "horizons": list(pred.horizons)}
floor = LiquidityFloor.from_config(cfg)
print("ckpt floor", LiquidityFloor.of(pred.metadata.get("hyperparams")).describe(), "| cfg", floor.describe(), flush=True)
engine = make_engine()
df, _ = read_training_features(engine, DataSettings.from_config(cfg))
print("rows", len(df), "cols", list(df.columns)[:12], flush=True)
s = config_samples(df, cfg)
H = list(s.horizons)
h1, h15 = H.index("1h"), H.index("15m")
ends = [int(pred.metadata[k]) for k in ("train_end_ts", "calibration_end_ts", "data_end_ts") if pred.metadata.get(k)]
train_end = int(pred.metadata["train_end_ts"])
after = max(ends) + embargo_seconds(s.horizons)
print("windows", len(s), "train_end", train_end, "holdout after", after, flush=True)

ds = SequenceDataset(s, np.arange(len(s)))
with torch.autocast("cuda", enabled=False):
    logits, truth = score_logits(pred.model, ds, batch_size=2048)
p = apply_calibration(logits, pred.calibration)
pp, rp = p["pump_prob"], p["rug_prob"]

thr_1h = cfg["labeling"]["pump_threshold"]["1h"]
rug_thr_1h = cfg["labeling"]["rug_threshold"]["1h"]
max_gap = max(3600 * cfg["labeling"]["max_gap_fraction"], cfg["labeling"]["min_gap_seconds"])
hold = s.end_ts > after
train = s.end_ts <= train_end
lab = s.mask[:, h1] > 0
report: dict = {"checkpoint": CKPT, "windows": len(s), "holdout": int(hold.sum()), "train": int(train.sum())}


def describe(sel: np.ndarray, name: str) -> dict:
    idx = np.flatnonzero(sel)
    mints = Counter(s.mint[idx])
    top5 = sum(c for _, c in mints.most_common(5))
    out = {
        "n": len(idx),
        "distinct_tokens": len(mints),
        "top5_share": top5 / max(len(idx), 1),
        "top5": mints.most_common(5),
        "venue": Counter(map(str, s.anchor_venue[idx])).most_common(),
        "bucket": Counter(liquidity_buckets(s.anchor_liquidity[idx], s.anchor_venue[idx])).most_common(),
        "anchor_liq_median": float(np.nanmedian(s.anchor_liquidity[idx])) if len(idx) else None,
    }
    for h in ("15m", "1h"):
        k = H.index(h)
        m = s.mask[idx, k] > 0
        out[h] = {
            "masked_share": float(1 - m.mean()) if len(idx) else None,
            "pump_rate": float(s.pump[idx, k][m].mean()) if m.any() else None,
            "rug_rate": float(s.rug[idx, k][m].mean()) if m.any() else None,
            "ret_mean": float(s.expected_return[idx, k][m].mean()) if m.any() else None,
            "ret_median": float(np.median(s.expected_return[idx, k][m])) if m.any() else None,
            "pump_prob_mean": float(pp[idx, k].mean()) if len(idx) else None,
        }
    print(f"\n== {name}\n" + json.dumps(out, indent=1, default=str), flush=True)
    return out


# Top 1% of the held-out 1h-labeled windows (the sweep's rule), and the same score threshold in training.
cut = np.quantile(pp[hold & lab, h1], 0.99)
print("top1% 1h threshold", cut, flush=True)
report["cut"] = float(cut)
top_hold = hold & lab & (pp[:, h1] >= cut)
report["holdout_top"] = describe(top_hold, "holdout top1% 1h")
report["holdout_top_rug_le_0.2"] = describe(top_hold & (rp[:, h1] <= 0.2), "holdout top1% & rug<=0.2")
report["holdout_all"] = describe(hold & lab, "holdout all 1h-labeled")
report["train_top"] = describe(train & lab & (pp[:, h1] >= cut), "train windows with pump_prob>=cut")
report["train_all"] = describe(train & lab, "train all 1h-labeled")
gate = top_hold & (rp[:, h1] <= 0.2)
for name, sel in (("gate_without_top_token", gate & (s.mint != "G4vGFQjeqYdbfckvipiuyusFgY1jqAK23mbYG4H8pump")),
                  ("top_without_top_token", top_hold & (s.mint != "G4vGFQjeqYdbfckvipiuyusFgY1jqAK23mbYG4H8pump"))):
    report[name] = {k: report_v for k, report_v in describe(sel, name).items() if k in ("n", "distinct_tokens", "1h", "15m")}
for r in (None, 0.5, 0.3, 0.2):
    sel = top_hold if r is None else top_hold & (rp[:, h1] <= r)
    print("rug<=", r, "n", int(sel.sum()), "prec", float(s.pump[sel, h1].mean()), "tokens", len(set(s.mint[sel])), flush=True)

# Per-token rows, time-ordered, for path analysis + an independent (naive) relabel.
df = df.sort_values(["mint", "ts"], kind="stable")
df["t"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[s]").astype(np.int64)
groups = {m: g for m, g in df.groupby("mint", sort=False)}


def naive(m: str, t: int, use_floor: bool) -> dict:
    g = groups[m]
    tt, pr, lq = g["t"].to_numpy(), g["price"].to_numpy(float), g["liquidity"].to_numpy(float)
    ve = g["venue"].to_numpy(object)
    ok = floor.tradeable(lq, ve) if use_floor else np.ones(len(g), bool)
    a = np.flatnonzero(tt == t)[-1]
    p0, l0 = pr[a], lq[a]
    fut = (tt > t) & (tt <= t + 3600)
    fp, fl, fok, ft, fv = pr[fut], lq[fut], ok[fut], tt[fut], ve[fut]
    fin = np.isfinite(fp)
    cnt = fin & fok
    pmax_c = fp[cnt].max() if cnt.any() else -np.inf
    pmax_all = fp[fin].max() if fin.any() else np.nan
    pmin = fp[fin].min() if fin.any() else np.nan
    lmin = np.nanmin(fl) if np.isfinite(fl).any() else np.nan
    crashed = (p0 - pmin) / p0 >= rug_thr_1h or (np.isfinite(l0) and l0 > 0 and np.isfinite(lmin) and (l0 - lmin) / l0 >= rug_thr_1h)
    covered = tt[-1] >= t + 3600 and len(ft) and ft[-1] >= t + 3600 - max_gap
    labeled = bool(fin.any() and (covered or crashed))
    upto = np.flatnonzero((tt <= t + 3600) & np.isfinite(pr) & ok)
    last = pr[upto[-1]] if len(upto) else p0
    below = np.flatnonzero(~fok)
    return {
        "labeled": labeled,
        "pump": bool(labeled and pmax_c / p0 - 1 >= thr_1h),
        "rug": bool(labeled and crashed),
        "ret": float(min(last / p0 - 1, 10.0)) if labeled else 0.0,
        "max_all": float(pmax_all / p0), "max_counted": float(pmax_c / p0), "min": float(pmin / p0),
        "liq0": float(l0), "liq_min_ratio": float(lmin / l0) if l0 else None,
        "first_below_floor_s": int(ft[below[0]] - t) if len(below) else None,
        "below_floor_rows": int(len(below)), "rows": int(len(ft)),
        "venues": sorted(set(map(str, fv))), "venue0": str(ve[a]),
        "covered": bool(covered), "end_ratio_all": float(fp[fin][-1] / p0) if fin.any() else None,
        "liq_end": float(fl[-1]) if len(fl) else None,
    }


idx = np.flatnonzero(top_hold)
G = "G4vGFQjeqYdbfckvipiuyusFgY1jqAK23mbYG4H8pump"
gi = np.flatnonzero(gate & (s.mint == G))
print("G windows", len(gi), "from", pd.to_datetime(s.end_ts[gi].min(), unit="s"), "to", pd.to_datetime(s.end_ts[gi].max(), unit="s"), flush=True)
g = df[df["mint"] == G]
t0 = s.end_ts[gi].min() - 3600
gg = g[(g["t"] >= t0) & (g["t"] <= s.end_ts[gi].max() + 3600)]
print(gg.iloc[:: max(1, len(gg) // 60)][["ts", "venue", "price", "liquidity", "market_cap", "volume", "buy_events", "sell_events"]].to_string(), flush=True)
report["G_timeline"] = gg[["t", "venue", "price", "liquidity"]].values.tolist()
rows, mism = [], Counter()
for i in idx:
    m, t = s.mint[i], int(s.end_ts[i])
    a, b = naive(m, t, True), naive(m, t, False)
    ok = (a["pump"] == bool(s.pump[i, h1]) and a["rug"] == bool(s.rug[i, h1]) and abs(a["ret"] - s.expected_return[i, h1]) < 1e-3)
    mism["match" if ok else "MISMATCH"] += 1
    rows.append({"i": int(i), "mint": m, "t": t, "pp": float(pp[i, h1]), "rp": float(rp[i, h1]),
                 "stored": {"pump": float(s.pump[i, h1]), "rug": float(s.rug[i, h1]), "ret": float(s.expected_return[i, h1]),
                            "pump15": float(s.pump[i, h15]), "rug15": float(s.rug[i, h15]), "ret15": float(s.expected_return[i, h15]), "mask15": float(s.mask[i, h15])},
                 "floor": a, "nofloor": {k: b[k] for k in ("pump", "rug", "ret")}})
print("\nrelabel check (floor):", dict(mism), flush=True)
fl = pd.DataFrame([{**r["floor"], "pp": r["pp"], "mint": r["mint"], "nf_pump": r["nofloor"]["pump"], "nf_ret": r["nofloor"]["ret"]} for r in rows])
summary = {
    "relabel": dict(mism),
    "floor_changes_pump": int((fl["pump"] != fl["nf_pump"]).sum()),
    "floor_changes_ret": int((abs(fl["ret"] - fl["nf_ret"]) > 1e-3).sum()),
    "any_below_floor_share": float((fl["below_floor_rows"] > 0).mean()),
    "migrated_share": float(fl["venues"].map(lambda v: len(v) > 1).mean()),
    "rug_share": float(fl["rug"].mean()),
    "max_all_ge3_share": float((fl["max_all"] >= 3).mean()),
    "max_counted_ge3_share": float((fl["max_counted"] >= 3).mean()),
    "median_max_all": float(fl["max_all"].median()), "median_min": float(fl["min"].median()),
    "median_liq_min_ratio": float(fl["liq_min_ratio"].median()),
    "median_first_below_floor_s": float(fl["first_below_floor_s"].dropna().median()) if fl["first_below_floor_s"].notna().any() else None,
    "covered_share": float(fl["covered"].mean()),
    "liq_end_zero_share": float((fl["liq_end"].fillna(0) < 1e-9).mean()),
}
print(json.dumps(summary, indent=1), flush=True)
report["path_summary"] = summary
report["per_token"] = (fl.groupby("mint").agg(n=("pp", "size"), pp=("pp", "mean"), venue=("venue0", "first"), liq0=("liq0", "median"),
                                              max_all=("max_all", "max"), minr=("min", "min"), rug=("rug", "mean"),
                                              first_below=("first_below_floor_s", "min"))
                       .sort_values("n", ascending=False).head(15).reset_index().to_dict("records"))
print(pd.DataFrame(report["per_token"]).to_string(), flush=True)
report["windows_sample"] = rows[:40]
report["G_windows"] = [r for r in rows if r["mint"] == G][:20]
print(json.dumps(report["G_windows"][:3], indent=1, default=str), flush=True)

# Train-side: does the candidate's high-1h-pump region carry pump=1 labels in training, and for which tokens?
tidx = np.flatnonzero(train & lab & (pp[:, h1] >= cut))
tm = Counter(s.mint[tidx][s.pump[tidx, h1] > 0])
report["train_top_pump_tokens"] = tm.most_common(10)
print("train top pump=1 tokens", tm.most_common(10), "distinct", len(tm), flush=True)
with open(OUT, "w") as f:
    json.dump(report, f, indent=1, default=str)
print("written", OUT)
