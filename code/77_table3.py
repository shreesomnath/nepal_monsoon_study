"""Step 77: the lengthening test by series, built from the data with one method throughout.

For each series (declared, gauge WiKS and two-knot composites, ERA5 composites, OLR four-cell withdrawal):
  slope 1981-2017 (Theil-Sen, moving-block bootstrap 95% interval, blocks of 4 years, 4,000 draws);
  step = mean 1997-2017 minus mean 1981-1996 (block bootstrap within periods, blocks of 3 years);
  slope of the difference series declared minus series, with block-bootstrap interval;
  correlation with the declared series;
  2018-2023 minus 1981-2017 (Welch interval from the yearly anomalies; fixed 98-station and 24-station sets).
Outputs: outputs/table3_lengthening_test.csv and outputs/table3_lengthening_test.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

rng = np.random.default_rng(77)
NB = 4000
yrs = np.arange(1981, 2018)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))


def load(name):
    return pd.read_csv(OUT / name, dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")


wk, cbd, era = load("wiks_station_dates_qc.csv"), load("cb_station_dates.csv"), load("wiks_era5_dates.csv")
era = era[era.version == "ge1mm"]
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")["withdrawal_doy"].reindex(yrs).to_numpy(float)
olr = pd.read_csv(OUT / "olr_dates.csv")
SETS = {"All": lambda d: d, "East provinces": lambda d: d[d.region == "East"], "East of 86E": lambda d: d[d.lon >= 86.0]}


def comp(df, col):
    P = df.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
    P = P[P.notna().sum(axis=1) >= 30]
    return P.median(axis=0).to_numpy(float), len(P)


def bidx(n, L):
    nb = int(np.ceil(n / L))
    s = rng.integers(0, n - L + 1, nb)
    return np.concatenate([np.arange(a, a + L) for a in s])[:n]


def slope(y, idx=None):
    idx = np.arange(len(y)) if idx is None else idx
    idx = idx[np.isfinite(y[idx])]
    if len(np.unique(idx)) < 15:
        return np.nan
    return stats.theilslopes(y[idx], yrs[idx].astype(float))[0] * 10


def slope_ci(y):
    b = np.array([slope(y, bidx(len(y), 4)) for _ in range(NB)])
    return slope(y), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


def step_ci(y):
    a, b = y[yrs <= 1996], y[yrs >= 1997]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = [b[bidx(len(b), 3)].mean() - a[bidx(len(a), 3)].mean() for _ in range(NB)]
    return b.mean() - a.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def fmt(t, nd=1):
    return f"{t[0]:+.{nd}f} ({t[1]:+.{nd}f} to {t[2]:+.{nd}f})"


series = [("Declared (eastern Nepal)", dec, None, None)]
for name, df, col in (("Gauge WiKS", wk, "withdrawal"), ("Gauge two-knot", cbd, "cb_withdrawal"), ("ERA5 WiKS", era, "withdrawal")):
    for sname, fn in SETS.items():
        y, n = comp(fn(df), col)
        series.append((f"{name}, {sname}", y, n, None))
o4 = olr[olr.series == "east_4cells"].set_index("year")["wd_250_15"].reindex(yrs).to_numpy(float)
series.append(("OLR four cells, 250 W m-2 and 15 d", o4, None, None))

by = pd.read_csv(OUT / "recent_years_withdrawal_by_year.csv", index_col=0)
olr_an = pd.read_csv(OUT / "recent_years_olr_withdrawal_anomaly.csv", index_col=0).iloc[:, 0]
recent_src = {
    "Declared (eastern Nepal)": by[[c for c in by.columns if c.startswith("DHM")][0]],
    "Gauge WiKS, All": by[[c for c in by.columns if c.startswith("Gauge") and "fixed (>=30" in c][0]],
    "Gauge WiKS, East of 86E": by[[c for c in by.columns if c.startswith("Gauge") and "east of 86E" in c][0]],
    "ERA5 WiKS, All": by[[c for c in by.columns if c.startswith("ERA5") and "fixed (>=30" in c][0]],
    "ERA5 WiKS, East of 86E": by[[c for c in by.columns if c.startswith("ERA5") and "east of 86E" in c][0]],
    "OLR four cells, 250 W m-2 and 15 d": olr_an,
}


def recent(s):
    s = s.dropna()
    a, b = s[s.index <= 2017].to_numpy(), s[s.index >= 2018].to_numpy()
    d = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - 2.03 * se, d + 2.03 * se


rows = []
for name, y, n, _ in series:
    r = dict(series=name.replace("East provinces", "East province group"), n_stations=n if n else "")
    r["slope"] = fmt(slope_ci(y))
    r["step_1997"] = fmt(step_ci(y))
    if name.startswith("Declared"):
        r["diff_slope"], r["r_declared"] = "", ""
    else:
        ok = np.isfinite(y)
        d = np.where(ok, dec - y, np.nan)
        r["diff_slope"] = fmt(slope_ci(d))
        r["r_declared"] = f"{np.corrcoef(y[ok], dec[ok])[0, 1]:+.2f}"
    r["recent_diff"] = fmt(recent(recent_src[name])) if name in recent_src else ""
    rows.append(r)
T = pd.DataFrame(rows)
T.to_csv(OUT / "table3_lengthening_test.csv", index=False)
cols = ["Series", "Stations", "Slope 1981 to 2017 (days per decade)", "Step 1997 to 2017 minus 1981 to 1996 (days)",
        "Slope of declared minus series (days per decade)", "r with declared", "2018 to 2023 minus 1981 to 2017 (days)"]
md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
for _, r in T.iterrows():
    md += f"| {r.series} | {r.n_stations} | {r.slope} | {r.step_1997} | {r.diff_slope} | {r.r_declared} | {r.recent_diff} |\n"
(OUT / "table3_lengthening_test.md").write_text(md, encoding="utf-8")
print(md)
