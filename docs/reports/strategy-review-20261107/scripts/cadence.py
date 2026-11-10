"""What the frozen rule would see at several scoring cadences, on the same tokens and the same model.

Read-only and CPU only. Period A has the bot's 3 min cadence live, period B its 5 min cadence, so
this separates "fewer good tokens" (hits fall in B at every cadence) from "looked too late" (at a
given cadence A and B agree, and slow cadences lose hits in both).

Looks at a token fall at first-seen + k x cadence, up to 600 s, on its latest row. A token enters on
its first look scoring >= 0.5 with a live pool. The exit replay is rough: next row's price, TP +200 %,
SL vol x 3 in [5 %, 30 %], 900 s, 2.5 % round trip, an empty pool pays nothing. No slippage or size.
"""

import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from solia_ai.db import make_engine
from solia_ai.features import DB_FEATURE_COLUMNS, build_feature_matrix
from solia_ai.inference import Predictor
from solia_ai.registry import read_champion

torch.set_num_threads(1)
PERIODS = {
    "A 09-12 UTC (live 3 min)": ("2026-11-06 09:00+00", "2026-11-06 12:00+00"),
    "B 21-23:40 UTC (live 5 min)": ("2026-11-06 21:00+00", "2026-11-06 23:40+00"),
}
CADENCES = (10, 30, 60, 120, 172, 280, 420)
THRESHOLD = 0.5
MAX_AGE = 600
OUT = sys.argv[1]
SEED = np.random.default_rng(0)

engine = make_engine()
record = read_champion(engine)
predictor = Predictor.from_record(record)
print("champion", record.version, "seq", predictor.seq_len, "min", predictor.min_seq_len, flush=True)
columns = ", ".join(dict.fromkeys(["mint", "ts", "venue", *DB_FEATURE_COLUMNS]))


def replay(price, liq, vol, ts, entry):
    """Rough result of an entry on row ``entry``: filled on the next row."""
    i = entry + 1
    if i >= len(price) or not price[i] > 0 or liq[i] < 1:
        return None
    p0, t0 = price[i], ts[i]
    v = vol[i]
    sl = 0.3 if not math.isfinite(v) or v <= 0 else min(max(3 * v, 0.05), 0.3)
    last = i
    for j in range(i + 1, len(price)):
        if ts[j] >= t0 + 900:
            break
        last = j
        if price[j] > 0 and (price[j] <= p0 * (1 - sl) or price[j] >= p0 * 3):
            k = min(j + 1, len(price) - 1)
            why = "sl" if price[j] <= p0 * (1 - sl) else "tp"
            return why, (-1.0 if liq[k] < 1 else price[k] / p0 - 1.025)
    return "time", (-1.0 if liq[last] < 1 else price[last] / p0 - 1.025)


report = {}
for name, (start, end) in PERIODS.items():
    t = time.time()
    mints = pd.read_sql(
        text(
            """select distinct mint from features where ts >= :s and ts < :e
            and token_age_sec <= 600 and unique_traders >= 100 and liquidity >= 50 and price > 0
            and price < 0.001 and venue in ('meteora','pumpswap')"""
        ),
        engine,
        params={"s": start, "e": end},
    )["mint"].tolist()
    print(name, "tokens", len(mints), round(time.time() - t), "s", flush=True)
    frames = []
    for i in range(0, len(mints), 200):
        frames.append(
            pd.read_sql(
                text(
                    f"""select {columns} from features where mint = any(:m)
                    and ts >= cast(:s as timestamptz) - interval '20 minutes'
                    and ts <= cast(:e as timestamptz) + interval '40 minutes' order by mint, ts"""
                ),
                engine,
                params={"m": mints[i : i + 200], "s": start, "e": end},
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    print("rows", len(rows), round(time.time() - t), "s", flush=True)
    matrix = build_feature_matrix(rows)
    epoch = pd.to_datetime(rows["ts"], utc=True).astype("int64").to_numpy() / 1e9
    if epoch.max() > 1e11:  # another resolution than ns would not need this
        epoch = epoch / 1.0
    unit = pd.to_datetime(rows["ts"], utc=True).dt.as_unit("ms").astype("int64").to_numpy() / 1e3
    epoch = unit
    price = pd.to_numeric(rows["price"], errors="coerce").to_numpy(float)
    liq = pd.to_numeric(rows["liquidity"], errors="coerce").fillna(0).to_numpy(float)
    vol = pd.to_numeric(rows["volatility"], errors="coerce").to_numpy(float)
    traders = pd.to_numeric(rows["unique_traders"], errors="coerce").fillna(0).to_numpy(float)
    s_e = pd.Timestamp(start).timestamp(), pd.Timestamp(end).timestamp()

    # Every row of a token's first 600 s (from its first row in the read) is scored once.
    tokens = {}
    for mint, idx in rows.groupby("mint", sort=False).indices.items():
        a, b = int(idx[0]), int(idx[-1]) + 1
        t0 = epoch[a]
        if not s_e[0] <= t0 < s_e[1]:
            continue  # first seen outside the period: its youth isn't in this read
        stop = a + int(np.searchsorted(epoch[a:b], t0 + MAX_AGE, side="right"))
        tokens[mint] = (a, b, stop, t0)
    want = [(m, i) for m, (a, _, stop, _) in tokens.items() for i in range(a, stop)
            if i + 1 - a >= predictor.min_seq_len]  # fmt: skip
    print("young tokens", len(tokens), "windows", len(want), flush=True)
    scores = np.full(len(rows), np.nan)
    started = time.time()
    for b in range(0, len(want), 256):
        part = want[b : b + 256]
        windows = [matrix[max(tokens[m][0], i + 1 - predictor.seq_len) : i + 1] for m, i in part]
        out = predictor.predict(windows, ["15m"])
        for (_, i), p in zip(part, out, strict=True):
            scores[i] = p.predictions[0].pump_prob
        if b % 12800 == 0:
            print("scored", b, round(time.time() - started), "s", flush=True)

    result = {"tokens": len(tokens), "windows": len(want), "cadences": {}}
    for c in CADENCES:
        looks = hits = active_hits = 0
        outcomes, ages = [], []
        for m, (a, end_i, stop, t0) in tokens.items():
            k = 1
            while t0 + k * c <= t0 + MAX_AGE:
                i = a + int(np.searchsorted(epoch[a:stop], t0 + k * c, side="right")) - 1
                k += 1
                if i < a or np.isnan(scores[i]):
                    continue
                looks += 1
                if scores[i] >= THRESHOLD and price[i] > 0 and price[i] < 0.001 and liq[i] >= 1:
                    hits += 1
                    active_hits += traders[i] >= 100
                    ages.append(epoch[i] - t0)
                    done = replay(price[a:end_i], liq[a:end_i], vol[a:end_i], epoch[a:end_i], i - a)
                    if done:
                        outcomes.append(done)
                    break
        pnl = np.array([o[1] for o in outcomes]) if outcomes else np.array([np.nan])
        boot = SEED.choice(pnl, (2000, len(pnl))).mean(1)
        result["cadences"][c] = {
            "looks": looks,
            "tokens_entered": hits,
            "with_100_traders": int(active_hits),
            "median_age_s": float(np.median(ages)) if ages else None,
            "replayed": len(outcomes),
            "mean_net": float(pnl.mean()),
            "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "tp": sum(o[0] == "tp" for o in outcomes),
            "sl": sum(o[0] == "sl" for o in outcomes),
            "time": sum(o[0] == "time" for o in outcomes),
            "total_loss": int((pnl <= -0.999).sum()),
        }
        print(name, c, json.dumps(result["cadences"][c]), flush=True)
    report[name] = result
    json.dump(report, open(OUT, "w"), indent=1)
print("=== done", flush=True)
