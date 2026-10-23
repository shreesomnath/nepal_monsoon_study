"""Step 69: the recent late withdrawals (2018-2023), with within-station anomalies and decade means.

Station composition changes in 2018-2020 (many stations silent), so composites are formed from each station's
anomaly relative to its own 1981-2017 mean; composition effects (for example a west-biased subset with a
different mean date) then cancel. Series: gauge WiKS (all stations with at least 25 dates before 2018 and at
least one date in 2018-2023, and the fixed set with at least 4 dates in 2018-2023), ERA5 WiKS for the same
station-years, OLR four-cell mean (to 2022), DHM declared (eastern Nepal). A year is shown only when at least
15 stations contribute. Reported: decade means of the anomalies (with the 1981-2017 mean removed), the mean of
2018-2023 minus the mean of 1981-2017 with a bootstrap interval (stations resampled), and the slope over
1981-2023 with its interval. Outputs: outputs/recent_years_*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, WORK  # noqa: E402

rng = np.random.default_rng(69)
allyrs = np.arange(1981, 2024)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})


def rd(p):
    return pd.read_csv(p, dtype={"station_id": str})


wk = pd.concat([rd(OUT / "wiks_station_dates_qc.csv"), rd(OUT / "wiks_station_dates_2018_2023.csv")])
era = rd(OUT / "wiks_era5_dates.csv")
era = pd.concat([era[era.version == "ge1mm"], rd(OUT / "wiks_era5_dates_2018_2023.csv")])
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")
sp = sp.set_index("station_id")


def anomalies(df, ev):
    P = df.pivot(index="station_id", columns="year", values=ev).reindex(columns=allyrs)
    base = P.loc[:, 1981:2017]
    n_old = base.notna().sum(axis=1)
    n_new = P.loc[:, 2018:2023].notna().sum(axis=1)
    mu = base.mean(axis=1)
    A = P.sub(mu, axis=0)
    return A, n_old, n_new


def comp_stats(A, label, ev, rows, min_n=15):
    n = A.notna().sum(axis=0)
    comp = A.median(axis=0).where(n >= min_n)
    z = comp.dropna()
    sl, ic, lo, hi = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
    d1 = comp[(comp.index >= 2018)].dropna().mean() - comp[comp.index <= 2017].dropna().mean()
    # bootstrap over stations
    bs = []
    for _ in range(1000):
        B = A.iloc[rng.integers(len(A), size=len(A))]
        c = B.median(axis=0).where(B.notna().sum(axis=0) >= min_n)
        bs.append(c[c.index >= 2018].dropna().mean() - c[c.index <= 2017].dropna().mean())
    dec_means = {f"{a}-{b}": comp[(comp.index >= a) & (comp.index <= b)].mean() for a, b in
                 ((1981, 1990), (1991, 2000), (2001, 2010), (2011, 2017), (2018, 2023))}
    rows.append(dict(series=label, event=ev, n_stations=len(A), years_2018_2023=int(comp[comp.index >= 2018].notna().sum()),
                     slope_1981_2023=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10, d_2018_2023_vs_1981_2017=d1,
                     boot_lo=np.percentile(bs, 2.5), boot_hi=np.percentile(bs, 97.5), **dec_means))
    return comp


rows, per_year = [], {}
for ev in ("withdrawal", "onset"):
    for lab, df in (("Gauge WiKS", wk), ("ERA5 WiKS", era)):
        A, n_old, n_new = anomalies(df, ev)
        sets = {"all (>=25 before, >=1 after)": A[(n_old >= 25) & (n_new >= 1)],
                "fixed (>=30 before, >=4 after)": A[(n_old >= 30) & (n_new >= 4)],
                "east of 86E (fixed)": A[(n_old >= 30) & (n_new >= 4) & (sp.loc[A.index, "lon"].to_numpy() >= 86.0)]}
        for sname, As in sets.items():
            per_year[(lab, sname, ev)] = comp_stats(As, f"{lab}, {sname}", ev, rows)
    d = dec[f"{ev}_doy"].reindex(allyrs)
    anom = d - d.loc[1981:2017].mean()
    z = anom.dropna()
    sl, ic, lo, hi = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
    rows.append(dict(series="DHM declared", event=ev, n_stations=np.nan, years_2018_2023=6, slope_1981_2023=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                     d_2018_2023_vs_1981_2017=anom.loc[2018:2023].mean() - anom.loc[1981:2017].mean(),
                     **{f"{a}-{b}": anom.loc[a:b].mean() for a, b in ((1981, 1990), (1991, 2000), (2001, 2010), (2011, 2017), (2018, 2023))}))
    per_year[("DHM declared", "", ev)] = anom
R = pd.DataFrame(rows)
R.to_csv(OUT / "recent_years_summary.csv", index=False)
pd.set_option("display.width", 250)
print("WITHIN-STATION ANOMALY COMPOSITES (days; slope per decade; decade means relative to the 1981-2017 mean):")
print(R[["series", "event", "n_stations", "years_2018_2023", "slope_1981_2023", "ci_lo", "ci_hi", "d_2018_2023_vs_1981_2017",
         "boot_lo", "boot_hi", "1981-1990", "1991-2000", "2001-2010", "2011-2017", "2018-2023"]].round(1).to_string(index=False))
Y = pd.DataFrame({f"{k[0]} | {k[1]} | {k[2]}": v for k, v in per_year.items() if k[2] == "withdrawal"})
Y.round(2).to_csv(OUT / "recent_years_withdrawal_by_year.csv")
print("\nWITHDRAWAL ANOMALY BY YEAR, 2008-2023 (days; fixed gauge set, ERA5 fixed set, DHM):")
cols = [c for c in Y.columns if "fixed (>=30" in c or "DHM" in c]
print(Y.loc[2008:2023, cols].round(1).to_string())

# OLR withdrawal anomaly
o = pd.concat([pd.read_csv(WORK / "inputs" / "olr_nepal_1981_2017.csv"), pd.read_csv(WORK / "inputs" / "olr_nepal_2018_2022.csv")])
o["time"] = pd.to_datetime(o.time)
cells = [(25.0, 85.0), (25.0, 87.5), (27.5, 85.0), (27.5, 87.5)]
ser = pd.concat([o[(o.latitude == la) & (o.longitude == lo)].set_index("time").olr for la, lo in cells], axis=1).mean(axis=1).sort_index()
ma = ser.rolling(7).mean()
res = {}
for y in range(1981, 2023):
    s_ = ma[(ma.index >= f"{y}-09-01") & (ma.index <= f"{y}-11-30")]
    ok = (s_ >= 250.0).astype(int)
    run = ok.rolling(15).min().shift(-14)
    c = run[run == 1]
    res[y] = c.index[0].dayofyear if len(c) else np.nan
olr_w = pd.Series(res)
anom = olr_w - olr_w.loc[1981:2017].mean()
anom.rename("olr_4cell_withdrawal_anomaly").round(2).to_csv(OUT / "recent_years_olr_withdrawal_anomaly.csv")
print("\nOLR 4-cell withdrawal anomaly (days) by decade:", {f"{a}-{b}": round(anom.loc[a:b].mean(), 1) for a, b in ((1981, 1990), (1991, 2000), (2001, 2010), (2011, 2017), (2018, 2022))})
print("OLR by year 2008-2022:", anom.loc[2008:2022].round(1).to_dict())
