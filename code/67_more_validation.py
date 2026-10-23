"""Step 67: additional validation of the timing composites.

A  Split-half reliability: stations are split at random into two halves (2000 times); the correlation between
   the two half-composites across 1981-2017 (Spearman-Brown corrected for the full composite) measures how
   reproducible the regional year-to-year signal is.
B  Gauge versus ERA5: correlation of the interannual composites from the same station-years.
C  Step change claimed after 1997 (Gautam and Regmi 2013, from the declared series): mean of 1997-2017 minus
   mean of 1981-1996 for the declared series, gauge composites (WiKS, two-knot), ERA5 and OLR, with 95%
   bootstrap intervals, and the Pettitt change-point test.
D  Out-of-sample extension to 2018-2023 (OLR to 2022): composites from a fixed station set (at least 30 of 37
   dates before 2018 and at least 4 dates in 2018-2023), the slope over 1981-2023 against the declared slope over
   the same years, and the mean of 2018-2023 minus the mean of 1981-2017.
Outputs: outputs/validation_more_*.csv and printed tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, WORK  # noqa: E402

rng = np.random.default_rng(67)
yrs = np.arange(1981, 2018)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
SETS = {"All": lambda df: df, "East provinces": lambda df: df[df.region == "East"], "East of 86E": lambda df: df[df.lon >= 86.0]}


def load(path, cols):
    df = pd.read_csv(path, dtype={"station_id": str}).merge(sp[["station_id", "region", "lon"]], on="station_id")
    return df


wk = load(OUT / "wiks_station_dates_qc.csv", None)
cbd = load(OUT / "cb_station_dates.csv", None)
era = load(OUT / "wiks_era5_dates.csv", None)
era = era[era.version == "ge1mm"]
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")


def pivot(df, col, years):
    return df.pivot(index="station_id", columns="year", values=col).reindex(columns=years)


def composite(P, min_dates=30):
    P = P[P.notna().sum(axis=1) >= min_dates]
    return P.median(axis=0), P


# ---------------- A: split-half reliability ----------------
rows = []
for src, df, cols in (("WiKS", wk, {"onset": "onset", "withdrawal": "withdrawal"}),
                      ("two-knot", cbd, {"onset": "cb_onset", "withdrawal": "cb_withdrawal"})):
    for sname, fn in SETS.items():
        for ev, col in cols.items():
            _, P = composite(pivot(fn(df), col, yrs))
            rs = []
            for _ in range(2000):
                m = rng.permutation(len(P)) < len(P) // 2
                a, b = P[m].median(axis=0), P[~m].median(axis=0)
                ok = a.notna() & b.notna()
                rs.append(np.corrcoef(a[ok], b[ok])[0, 1])
            rs = np.array(rs)
            sb = 2 * np.median(rs) / (1 + np.median(rs))
            rows.append(dict(definition=src, set=sname, event=ev, n_stations=len(P), r_half_median=np.median(rs),
                             r_half_lo=np.percentile(rs, 2.5), r_half_hi=np.percentile(rs, 97.5), spearman_brown=sb))
A = pd.DataFrame(rows)
A.to_csv(OUT / "validation_more_split_half.csv", index=False)
print("A. SPLIT-HALF RELIABILITY OF THE REGIONAL COMPOSITES (1981-2017):")
print(A.round(2).to_string(index=False))

# ---------------- B: gauge vs ERA5 ----------------
rows = []
for sname, fn in SETS.items():
    for ev in ("onset", "withdrawal"):
        Pg = pivot(fn(wk), ev, yrs)
        Pe = pivot(fn(era), ev, yrs)
        keep = [s for s in Pg.index if s in Pe.index and Pg.loc[s].notna().sum() >= 30]
        Pg, Pe = Pg.loc[keep], Pe.loc[keep].where(Pg.loc[keep].notna())      # ERA5 only where the gauge year exists
        cg, ce = Pg.median(axis=0), Pe.median(axis=0)
        r, p = stats.pearsonr(cg, ce)
        rs, ps = stats.spearmanr(cg, ce)
        rows.append(dict(set=sname, event=ev, n_stations=len(keep), pearson=r, p=p, spearman=rs, p_spearman=ps))
B = pd.DataFrame(rows)
B.to_csv(OUT / "validation_more_gauge_vs_era5.csv", index=False)
print("\nB. INTERANNUAL CORRELATION OF THE GAUGE COMPOSITE WITH THE ERA5 COMPOSITE (same station-years):")
print(B.round(3).to_string(index=False))

# ---------------- C: step change after 1997 ----------------
olr = pd.read_csv(OUT / "olr_dates.csv")
series = {}
for ev, dcol in (("onset", "onset_doy"), ("withdrawal", "withdrawal_doy")):
    series[("DHM declared", ev)] = dec[dcol].reindex(yrs)
    for sname, fn in SETS.items():
        series[(f"Gauge WiKS, {sname}", ev)] = composite(pivot(fn(wk), ev, yrs))[0]
    series[("Gauge two-knot, All", ev)] = composite(pivot(cbd, "cb_" + ev, yrs))[0]
    series[("ERA5 WiKS, All", ev)] = composite(pivot(era, ev, yrs))[0]
    series[("ERA5 WiKS, East of 86E", ev)] = composite(pivot(SETS["East of 86E"](era), ev, yrs))[0]
series[("OLR 4 cells", "onset")] = olr[olr.series == "east_4cells"].set_index("year")["onset"].reindex(yrs)
series[("OLR 4 cells", "withdrawal")] = olr[olr.series == "east_4cells"].set_index("year")["wd_250_15"].reindex(yrs)
series[("OLR 15 cells", "withdrawal")] = olr[olr.series == "all_15cells"].set_index("year")["wd_250_15"].reindex(yrs)


def pettitt(x):
    x = np.asarray(x, float)
    n = len(x)
    U = [np.sum(np.sign(x[t + 1:, None] - x[None, :t + 1])) for t in range(n - 1)]
    K = np.max(np.abs(U))
    return int(np.argmax(np.abs(U))), 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))


rows = []
for (name, ev), s in series.items():
    s = s.dropna()
    a, b = s[s.index <= 1996], s[s.index >= 1997]
    if len(a) < 8 or len(b) < 8:
        continue
    diffs = [rng.choice(b.to_numpy(), len(b)).mean() - rng.choice(a.to_numpy(), len(a)).mean() for _ in range(3000)]
    k, pp = pettitt(s.to_numpy())
    rows.append(dict(series=name, event=ev, mean_1981_96=a.mean(), mean_1997_2017=b.mean(), diff=b.mean() - a.mean(),
                     ci_lo=np.percentile(diffs, 2.5), ci_hi=np.percentile(diffs, 97.5),
                     pettitt_year=int(s.index[k + 1]), pettitt_p=min(pp, 1.0)))
C = pd.DataFrame(rows)
C.to_csv(OUT / "validation_more_step_1997.csv", index=False)
print("\nC. MEAN OF 1997-2017 MINUS MEAN OF 1981-1996 (days), bootstrap 95% CI, Pettitt test:")
print(C.round(2).to_string(index=False))

# ---------------- D: out-of-sample extension ----------------
f_new = OUT / "wiks_station_dates_2018_2023.csv"
if f_new.exists():
    wn = load(f_new, None)
    en = load(OUT / "wiks_era5_dates_2018_2023.csv", None)
    allyrs = np.arange(1981, 2024)
    wk_all = pd.concat([wk[["station_id", "year", "onset", "withdrawal", "region", "lon"]], wn[["station_id", "year", "onset", "withdrawal", "region", "lon"]]])
    en_all = pd.concat([era[["station_id", "year", "onset", "withdrawal", "region", "lon"]], en[["station_id", "year", "onset", "withdrawal", "region", "lon"]]])
    old = pivot(wk, "withdrawal", yrs).notna().sum(axis=1)
    new = pivot(wn, "withdrawal", np.arange(2018, 2024)).notna().sum(axis=1)
    fixed = [s for s in old.index if old[s] >= 30 and new.get(s, 0) >= 4]
    print("\nD. fixed station set (>= 30 dates before 2018 and >= 4 in 2018-2023):", len(fixed))
    dd = dec.reindex(allyrs)
    rows = []
    olr_new = WORK / "inputs" / "olr_nepal_2018_2022.csv"
    olr_series = None
    if olr_new.exists():
        o = pd.concat([pd.read_csv(WORK / "inputs" / "olr_nepal_1981_2017.csv"), pd.read_csv(olr_new)])
        o["time"] = pd.to_datetime(o.time)
        cells = [(25.0, 85.0), (25.0, 87.5), (27.5, 85.0), (27.5, 87.5)]
        ser = pd.concat([o[(o.latitude == la) & (o.longitude == lo)].set_index("time").olr for la, lo in cells], axis=1).mean(axis=1).sort_index()
        ma = ser.rolling(7).mean()

        def wd(y, thr=250.0, persist=15):
            s_ = ma[(ma.index >= f"{y}-09-01") & (ma.index <= f"{y}-11-30")]
            ok = (s_ >= thr).astype(int)
            run = ok.rolling(persist).min().shift(-(persist - 1))
            c = run[run == 1]
            return c.index[0].dayofyear if len(c) else np.nan
        olr_series = pd.Series({y: wd(y) for y in range(1981, 2023)})
    for ev in ("onset", "withdrawal"):
        for lab, df in (("Gauge WiKS", wk_all), ("ERA5 WiKS", en_all)):
            for sname, fn in (("fixed set, all", lambda d_: d_[d_.station_id.isin(fixed)]),
                              ("fixed set, east of 86E", lambda d_: d_[d_.station_id.isin(fixed) & (d_.lon >= 86.0)])):
                P = pivot(fn(df), ev, allyrs)
                n_per_year = P.notna().sum(axis=0)
                comp = P.median(axis=0).where(n_per_year >= 15)
                z = comp.dropna()
                sl, ic, lo, hi = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
                before, after = comp[comp.index <= 2017].dropna(), comp[comp.index >= 2018].dropna()
                rows.append(dict(series=lab, subset=sname, event=ev, n_stations=int(P.notna().any(axis=1).sum()),
                                 years_2018_2023_used=len(after), slope_1981_2023=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                                 mean_2018_2023_minus_1981_2017=after.mean() - before.mean()))
        z = dd[f"{ev}_doy"].dropna()
        sl, ic, lo, hi = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
        rows.append(dict(series="DHM declared", subset="eastern Nepal", event=ev, n_stations=np.nan, years_2018_2023_used=6,
                         slope_1981_2023=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                         mean_2018_2023_minus_1981_2017=z[z.index >= 2018].mean() - z[z.index <= 2017].mean()))
    if olr_series is not None:
        z = olr_series.dropna()
        sl, ic, lo, hi = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
        rows.append(dict(series="OLR 4 cells (to 2022)", subset="250 W m-2, 15 d", event="withdrawal", n_stations=np.nan,
                         years_2018_2023_used=int((z.index >= 2018).sum()), slope_1981_2023=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                         mean_2018_2023_minus_1981_2017=z[z.index >= 2018].mean() - z[z.index <= 2017].mean()))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "validation_more_extension_2023.csv", index=False)
    print("\nD. EXTENSION TO 2018-2023 (slopes in days per decade; differences in days):")
    print(D.round(2).to_string(index=False))
else:
    print("\nD. skipped: extension dates not available yet")
