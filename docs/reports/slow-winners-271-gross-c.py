import json
from pathlib import Path

import pandas as pd

from solia_ai import wallet_study as w
from solia_ai.survivor_study import Costs

sel = json.loads(Path("/scratch/slow-winners-271.json").read_text())
free = Costs(size=1e-9, network_fee=0.0, account_rent=0.0, fee_pct={"default": 0.0}, virtual_sol={})
small = Costs(size=0.1, virtual_sol={})
combos = [("ideal/0s", pd.Index(sel["leaders"]), 0.0), ("ideal/3s", pd.Index(sel["leaders"]), 3.0)]
out = {}
for label, costs in (("no costs", free), ("0.1 SOL", small)):
    copies = w.partition_copies(Path("/scratch/wallets-c"), w.PERIOD_C, combos, costs)
    for name, df in copies.items():
        s = w.copy_summary(df, 500)
        s["mean_ci_by_token"] = w.cluster_mean_ci(df["net"], df["mint"], 500)
        out[f"{label} {name}"] = s
        print(label, name, {k: s[k] for k in ("copies", "mean", "median", "win_rate",
                                               "mean_ci_by_token")})
Path("/scratch/gross-c-271.json").write_text(json.dumps(out, indent=2, default=str))
