"""Step 64: OLR-based onset and withdrawal as a gauge-independent check (1981-2017).

OLR: NOAA PSL interpolated daily OLR (2.5 degrees; Liebmann and Smith 1996), cells covering Nepal,
fetched by code/63. Onset follows Adhikari et al. (2022, Jalawaayu 2(1):19-32): the first day after
21 May on which the 7-day moving mean OLR falls below 250 W m-2 (their 'characteristic fall'; they
add the mean offset to DHM onset, here only the dates and their correlations are used). Withdrawal is not
defined in that paper; here it is the first day on or after 1 September from which the 7-day mean stays at
or above a threshold for a persistence window (main case 250 W m-2 for 15 days), with 240/260 W m-2 and
10/20 days as sensitivity cases. Series used: the 27.5N 87.5E cell (eastern Nepal) and the mean of the
four cells 25 to 27.5N, 85 to 87.5E, and the mean of all 15 cells.
Comparisons over 1981-2017: correlation with the DHM-declared dates, with the WiKS, two-knot and ERA5 gauge-
region composites, and Sen trends with intervals.
Outputs: outputs/olr_dates.csv, outputs/olr_checks.csv and printed tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, WORK  # noqa: E402

yrs = np.arange(1981, 2018)
d = pd.read_csv(WORK / "inputs" / "olr_nepal_1981_2017.csv", parse_dates=["time"])
d = d.rename(columns={"latitude": "lat", "longitude": "lon"})
d["lat"] = d.lat.round(2)
d["lon"] = d.lon.round(2)
print("OLR rows:", len(d), "| lat", sorted(d.lat.unique()), "| lon", sorted(d.lon.unique()), "| missing", d.olr.isna().sum())

SERIES = {"cell_27.5N_87.5E": [(27.5, 87.5)],
          "east_4cells": [(25.0, 85.0), (25.0, 87.5), (27.5, 85.0), (27.5, 87.5)],
          "all_15cells": [(la, lo) for la in (25.0, 27.5, 30.0) for lo in (80.0, 82.5, 85.0, 87.5, 90.0)]}


def series(cells):
    m = pd.concat([d[(d.lat == la) & (d.lon == lo)].set_index("time").olr for la, lo in cells], axis=1)
    return m.mean(axis=1)


def fall_date(ma, year, thr=250.0):
    s = ma[(ma.index >= f"{year}-05-22") & (ma.index <= f"{year}-08-31")]
    below = s[s < thr]
    return below.index[0].dayofyear if len(below) else np.nan


def withdraw_date(ma, year, thr=250.0, persist=15):
    s = ma[(ma.index >= f"{year}-09-01") & (ma.index <= f"{year}-11-30")]
    ok = (s >= thr).astype(int)
    run = ok.rolling(persist).min().shift(-(persist - 1))     # True when the next `persist` days are all >= thr
    cand = run[run == 1]
    return cand.index[0].dayofyear if len(cand) else np.nan


rows = []
for name, cells in SERIES.items():
    ma = series(cells).sort_index().rolling(7, center=False).mean()
    for y in yrs:
        row = dict(series=name, year=int(y), onset=fall_date(ma, y))
        for thr in (240.0, 250.0, 260.0):
            for pers in (10, 15, 20):
                row[f"wd_{int(thr)}_{pers}"] = withdraw_date(ma, y, thr, pers)
        rows.append(row)
O = pd.DataFrame(rows)
O.to_csv(OUT / "olr_dates.csv", index=False)

# comparison series
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))


def comp(df, col, subset):
    df = df.merge(sp[["station_id", "region", "lon"]], on="station_id")
    if subset == "East of 86E":
        df = df[df.lon >= 86.0]
    elif subset == "East provinces":
        df = df[df.region == "East"]
    P = df.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
    P = P[P.notna().sum(axis=1) >= 30]
    return P.median(axis=0)


wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
cbd = pd.read_csv(OUT / "cb_station_dates.csv", dtype={"station_id": str})
era = pd.read_csv(OUT / "wiks_era5_dates.csv", dtype={"station_id": str})
era = era[era.version == "ge1mm"]
refs = {"DHM declared (eastern Nepal)": {"onset": dec.onset_doy.reindex(yrs), "withdrawal": dec.withdrawal_doy.reindex(yrs)}}
for sub in ("East of 86E", "East provinces", "All"):
    refs[f"Gauge WiKS, {sub}"] = {"onset": comp(wk, "onset", sub), "withdrawal": comp(wk, "withdrawal", sub)}
refs["Gauge two-knot, East of 86E"] = {"onset": comp(cbd, "cb_onset", "East of 86E"), "withdrawal": comp(cbd, "cb_withdrawal", "East of 86E")}
refs["ERA5 WiKS, East of 86E"] = {"onset": comp(era, "onset", "East of 86E"), "withdrawal": comp(era, "withdrawal", "East of 86E")}

res = []
for name in SERIES:
    s = O[O.series == name].set_index("year")
    for ev, cols in (("onset", ["onset"]), ("withdrawal", [c for c in O.columns if c.startswith("wd_")])):
        for col in cols:
            x = s[col].reindex(yrs)
            ok = x.notna()
            sl, ic, lo, hi = stats.theilslopes(x[ok].to_numpy(float), yrs[ok].astype(float)) if ok.sum() > 10 else (np.nan,) * 4
            row = dict(olr_series=name, event=ev, definition=col, n_years=int(ok.sum()), mean_doy=x.mean(),
                       sd_d=x.std(), trend_d_per_decade=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                       trend_p=stats.kendalltau(yrs[ok], x[ok])[1] if ok.sum() > 10 else np.nan)
            for rn, rv in refs.items():
                z = pd.concat([x, rv[ev].reindex(yrs)], axis=1).dropna()
                row[f"r_{rn}"] = np.corrcoef(z.iloc[:, 0], z.iloc[:, 1])[0, 1] if len(z) > 10 else np.nan
            res.append(row)
R = pd.DataFrame(res)
R.to_csv(OUT / "olr_checks.csv", index=False)
pd.set_option("display.width", 250)
print("\nOLR ONSET (first 7-day mean < 250 W m-2 after 21 May):")
print(R[R.event == "onset"].round(2).T.to_string())
print("\nOLR WITHDRAWAL, main definition (>= 250 W m-2 for 15 days) and sensitivity:")
print(R[(R.event == "withdrawal")][["olr_series", "definition", "n_years", "mean_doy", "sd_d", "trend_d_per_decade", "ci_lo", "ci_hi", "trend_p"]].round(2).to_string(index=False))
print("\nCorrelations of the main OLR withdrawal (250/15) with the references:")
print(R[(R.event == "withdrawal") & (R.definition == "wd_250_15")][["olr_series"] + [c for c in R.columns if c.startswith("r_")]].round(2).T.to_string())
