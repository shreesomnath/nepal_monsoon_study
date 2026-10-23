"""Step 75: two additional checks of the composites.

A  Station-anomaly composites for 1981-2017. The raw-date composites use a fixed station list, but each station
   still misses some years, so the median can move with which stations report. Here each station-year date is
   expressed as an anomaly from the station's own 1981-2017 mean, and the composite is the median anomaly across
   reporting stations (at least 15 per year). Slope (Theil-Sen, 95% interval), 1997 step, correlation with the declared
   series, and the declared-minus-composite slope are reported for withdrawal.
B  Leave-one-year-out for 2018-2023: the mean difference (2018-2023 minus 1981-2017) of the fixed gauge set, ERA5 and
   the declared series when each of the six recent years is dropped in turn, and the median in place of the mean.
Outputs: outputs/anomaly_composites.csv, outputs/recent_leave_one_year_out.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

rng = np.random.default_rng(75)
yrs = np.arange(1981, 2018)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")
cb = pd.read_csv(OUT / "cb_station_dates.csv", dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")["withdrawal_doy"].reindex(yrs).to_numpy(float)
SETS = {"All": lambda d: d, "East provinces": lambda d: d[d.region == "East"], "East of 86E": lambda d: d[d.lon >= 86.0]}


def anomaly_composite(df, col, min_dates=30, min_stations=15):
    P = df.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
    P = P[P.notna().sum(axis=1) >= min_dates]
    A = P.sub(P.mean(axis=1), axis=0)
    c = A.median(axis=0)
    c[A.notna().sum(axis=0) < min_stations] = np.nan
    return c.to_numpy(float), len(P)


def slope_ci(y):
    ok = np.isfinite(y)
    s, ic, lo, hi = stats.theilslopes(y[ok], yrs[ok].astype(float))
    return s * 10, lo * 10, hi * 10


rows = []
for name, df, col in (("WiKS", wk, "withdrawal"), ("two-knot", cb, "cb_withdrawal")):
    for sname, fn in SETS.items():
        y, n = anomaly_composite(fn(df), col)
        ok = np.isfinite(y)
        s, lo, hi = slope_ci(y)
        d = dec[ok] - y[ok]
        ds, dlo, dhi = stats.theilslopes(d, yrs[ok].astype(float))[0] * 10, *[v * 10 for v in stats.theilslopes(d, yrs[ok].astype(float))[2:]]
        a, b = y[yrs <= 1996], y[yrs >= 1997]
        rows.append(dict(definition=name, set=sname, n_stations=n, n_years=int(ok.sum()), slope=s, lo=lo, hi=hi,
                         step_1997=np.nanmean(b) - np.nanmean(a), r_declared=np.corrcoef(y[ok], dec[ok])[0, 1],
                         diff_slope=ds, diff_lo=dlo, diff_hi=dhi))
A = pd.DataFrame(rows)
A.to_csv(OUT / "anomaly_composites.csv", index=False)
print("A. STATION-ANOMALY COMPOSITES, withdrawal 1981-2017 (d/decade):")
print(A.round(2).to_string(index=False))

# ---------------- B ----------------
by = pd.read_csv(OUT / "recent_years_withdrawal_by_year.csv", index_col=0)
cols = {"Gauge fixed": [c for c in by.columns if "fixed (>=30" in c and c.startswith("Gauge")][0],
        "ERA5 fixed": [c for c in by.columns if "fixed (>=30" in c and c.startswith("ERA5")][0],
        "Declared": [c for c in by.columns if c.startswith("DHM")][0]}
rows = []
for lab, c in cols.items():
    s = by[c]
    base = s[(s.index >= 1981) & (s.index <= 2017)]
    rec = s[s.index >= 2018]
    row = dict(series=lab, mean_diff=rec.mean() - base.mean(), median_diff=rec.median() - base.median())
    for y in rec.index:
        row[f"drop_{y}"] = rec.drop(y).mean() - base.mean()
    row["min_drop"] = min(v for k, v in row.items() if k.startswith("drop_"))
    rows.append(row)
B = pd.DataFrame(rows)
B.to_csv(OUT / "recent_leave_one_year_out.csv", index=False)
print("\nB. 2018-2023 minus 1981-2017, leave-one-year-out (d):")
print(B.round(1).to_string(index=False))
