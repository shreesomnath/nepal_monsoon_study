"""Step 76: moving-block bootstrap (blocks of 4 years, 4,000 draws) intervals for the composite trends of 4.3,
alongside the Theil-Sen intervals used earlier, for WiKS and two-knot, onset and withdrawal, three station sets.
Output: outputs/composite_slopes_block_bootstrap.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

rng = np.random.default_rng(76)
NB, L = 4000, 4
yrs = np.arange(1981, 2018)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")
cb = pd.read_csv(OUT / "cb_station_dates.csv", dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")
SETS = {"All": lambda d: d, "East provinces": lambda d: d[d.region == "East"], "East of 86E": lambda d: d[d.lon >= 86.0]}


def block_idx(n):
    nb = int(np.ceil(n / L))
    s = rng.integers(0, n - L + 1, nb)
    return np.concatenate([np.arange(a, a + L) for a in s])[:n]


rows = []
for name, df, pre in (("WiKS", wk, ""), ("two-knot", cb, "cb_")):
    for sname, fn in SETS.items():
        for ev in ("onset", "withdrawal"):
            P = fn(df).pivot(index="station_id", columns="year", values=pre + ev).reindex(columns=yrs)
            P = P[P.notna().sum(axis=1) >= 30]
            y = P.median(axis=0).to_numpy(float)
            ok = np.isfinite(y)
            s, ic, lo, hi = stats.theilslopes(y[ok], yrs[ok].astype(float))
            bs = []
            for _ in range(NB):
                i = block_idx(len(y))
                i = i[np.isfinite(y[i])]
                if len(np.unique(i)) < 15:
                    continue
                bs.append(stats.theilslopes(y[i], yrs[i].astype(float))[0] * 10)
            bs = np.array(bs)
            rows.append(dict(definition=name, set=sname, event=ev, n_stations=len(P), slope=s * 10, ts_lo=lo * 10, ts_hi=hi * 10,
                             boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5)))
R = pd.DataFrame(rows)
R.to_csv(OUT / "composite_slopes_block_bootstrap.csv", index=False)
print(R.round(2).to_string(index=False))
