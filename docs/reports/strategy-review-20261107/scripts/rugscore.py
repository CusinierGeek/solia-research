"""Rug score on every feature row of each closed position of sessions 5 and 6, with the served model.

Read-only, CPU only. Also compares the offline scores with the live predictions at the same rows.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch
from sqlalchemy import text

from solia_ai.db import make_engine
from solia_ai.features import DB_FEATURE_COLUMNS, build_feature_matrix
from solia_ai.inference import Predictor
from solia_ai.registry import read_champion

torch.set_num_threads(1)
engine = make_engine()
predictor = Predictor.from_record(read_champion(engine))
trades = pd.read_sql(
    text(
        """select t.id, t.session_id, t.mint, t.opened_at, t.closed_at, t.entry_price::float8 ep,
        t.size::float8 sz, t.pnl_pct, t.exit_reason, p.rug_prob rug0
        from trades t left join predictions p on p.id=t.prediction_id
        where t.session_id in (5,6) and t.status='closed' order by t.id"""
    ),
    engine,
)
columns = ", ".join(dict.fromkeys(["mint", "ts", "venue", *DB_FEATURE_COLUMNS]))
out = []
live_pairs = []
for tr in trades.itertuples():
    rows = pd.read_sql(
        text(
            f"""select {columns} from features where mint = :m and ts <= :c
            and ts >= cast(:o as timestamptz) - interval '30 minutes' order by ts"""
        ),
        engine,
        params={"m": tr.mint, "c": tr.closed_at, "o": tr.opened_at},
    )
    matrix = build_feature_matrix(rows)
    ts = pd.to_datetime(rows["ts"], utc=True)
    held = np.flatnonzero((ts > tr.opened_at).to_numpy())
    wanted = [i for i in held if i + 1 >= predictor.min_seq_len]
    rug, pump = {}, {}
    for b in range(0, len(wanted), 128):
        part = wanted[b : b + 128]
        windows = [matrix[max(0, i + 1 - predictor.seq_len) : i + 1] for i in part]
        for i, p in zip(part, predictor.predict(windows, ["15m"]), strict=True):
            rug[i] = p.predictions[0].rug_prob
            pump[i] = p.predictions[0].pump_prob
    live = pd.read_sql(
        text(
            """select ts, rug_prob, pump_prob from predictions where mint=:m and horizon='15m' and ts>:o and ts<=:c"""
        ),
        engine,
        params={"m": tr.mint, "o": tr.opened_at, "c": tr.closed_at},
    )
    by_ts = {t: i for i, t in enumerate(ts)}
    for row in live.itertuples():
        i = by_ts.get(pd.Timestamp(row.ts))
        if i in rug:
            live_pairs.append((rug[i], row.rug_prob, pump[i], row.pump_prob))
    price = pd.to_numeric(rows["price"], errors="coerce").to_numpy(float)
    liq = pd.to_numeric(rows["liquidity"], errors="coerce").fillna(0).to_numpy(float)
    out.append({
        "id": int(tr.id), "session": int(tr.session_id), "mint": tr.mint, "ep": tr.ep, "sz": tr.sz,
        "pnl": float(tr.pnl_pct), "why": tr.exit_reason, "rug0": None if pd.isna(tr.rug0) else float(tr.rug0),
        "o": tr.opened_at.timestamp(), "c": tr.closed_at.timestamp(),
        "rows": [
            [ts.iloc[i].timestamp(), rug[i], pump[i], float(price[i]), float(liq[i])] for i in wanted
        ],
    })
    print(tr.id, len(wanted), flush=True)

if live_pairs:
    a = np.array(live_pairs)
    d = np.abs(a[:, 0] - a[:, 1])
    print("live check rug: n", len(a), "median abs diff", float(np.median(d)), "p99", float(np.percentile(d, 99)),
          "pump median abs diff", float(np.median(np.abs(a[:, 2] - a[:, 3]))), flush=True)
json.dump(out, open(sys.argv[1], "w"))
print("=== done", flush=True)
