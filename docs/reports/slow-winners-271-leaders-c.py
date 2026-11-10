import json
from pathlib import Path

import pandas as pd

from solia_ai import wallet_study as w

sel = json.loads(Path("/scratch/slow-winners-271.json").read_text())
groups = {"leaders": set(sel["leaders"]), "control": set(sel["control"])}
pos = []
for part in w.partitions(Path("/scratch/wallets-c")):
    f = w.load_period(part, *w.PERIOD_C)
    p = w._slim(w.positions(f["trades"], f["creates"], f["transfers"]))
    pos.append(p[p["wallet"].isin(groups["leaders"] | groups["control"])])
pos = pd.concat(pos, ignore_index=True)
prof = w.wallet_profiles(pos)
out = {}
for name, who in groups.items():
    s = w.group_stats(prof, pd.Index(sorted(who)), samples=1000)
    out[name] = s
    print(name, {k: v for k, v in s.items()})
Path("/scratch/leaders-c-271.json").write_text(json.dumps(out, indent=2, default=str))
