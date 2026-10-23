"""Step 58: robustness of the timing and lengthening results.

A  Noise ceiling: agreement of WiKS dates between neighbouring gauges in the same year, by distance.
   ERA5-versus-gauge agreement (code/54) can be no better than gauges agree with each other.
B  A second, WiKS-free measure of withdrawal and season end for the same station-years: D95 (day of
   year on which 95% of the 1 June-31 October total has fallen) and the share of that total falling
   after 20 September (late share). If the monsoon were ending later by about 6 days per decade (DHM-
   declared withdrawal, 1981-2017), the late share should rise and D95 should move later. Complete
   seasons only (95% of the 153 days observed, no infilling). Computed on gauges and on ERA5 at the
   station cells for the same station-years.
C  Fixed-network composites (stations with at least 30 of 37 dates) for the WiKS dates, gauges and ERA5,
   so that network change cannot drive the composite.
D  The declared DHM series over the same 1981-2017 window and over the full record.
Outputs: outputs/robust_*.csv and printed tables.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, YEARS  # noqa: E402

spec = importlib.util.spec_from_file_location("t55", str(Path(__file__).resolve().parent / "55_trends_budget.py"))
t55 = importlib.util.module_from_spec(spec)
sys.argv = [sys.argv[0], "all"]
spec.loader.exec_module(t55)
sen_mk, field_test = t55.sen_mk, t55.field_test

yrs = np.array(list(YEARS))
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
dates = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str}).merge(
    sp[["station_id", "region", "lat", "lon", "elev"]], on="station_id")

# ---------------- A: noise ceiling ----------------
def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))

st = sp.set_index("station_id")
ids = [s for s in dates.station_id.unique()]
lat = st.loc[ids, "lat"].to_numpy()
lon = st.loc[ids, "lon"].to_numpy()
D = hav(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
piv = {ev: dates.pivot(index="station_id", columns="year", values=ev).reindex(ids) for ev in ("onset", "withdrawal")}
rows = []
bins = [(0, 25), (25, 50), (50, 100)]
for ev in ("onset", "withdrawal"):
    P = piv[ev].to_numpy(float)
    for lo, hi in bins:
        i, j = np.where((D > lo) & (D <= hi) & (np.arange(len(ids))[:, None] < np.arange(len(ids))[None, :]))
        diff = np.abs(P[i] - P[j])
        diff = diff[np.isfinite(diff)]
        rows.append(dict(event=ev, distance_km=f"{lo}-{hi}", n_pairs=len(i), n=len(diff), within7=(diff <= 7).mean(),
                         within14=(diff <= 14).mean(), median_abs_d=np.median(diff)))
ceil = pd.DataFrame(rows)
ceil.to_csv(OUT / "robust_noise_ceiling.csv", index=False)
print("A. NEIGHBOURING-GAUGE AGREEMENT (same year; ceiling for any product comparison):")
print(ceil.round(2).to_string(index=False))

# ---------------- B: WiKS-free late-season measures ----------------
d = np.load(INP / "daily_qc.npz", allow_pickle=True)
X, status = d["X"], d["status"]
sid_all = list(d["station_id"])
yr = pd.to_datetime(d["dates"]).year.to_numpy()
V = np.where(status == 0, X, np.nan)
e = np.load(INP / "era5_station_daily.npz", allow_pickle=True)
E = e["era5_centre_mm"].astype(float)
S, F = slice(151, 304), slice(263, 304)      # 1 Jun-31 Oct (index 151..303), after 20 Sep (index 263..303)


def metrics(v):
    seg = v[S]
    if np.isfinite(seg).mean() < 0.95:
        return None
    tot = np.nansum(seg)
    if tot < 300:
        return None
    cum = np.cumsum(np.nan_to_num(seg))
    d95 = int(np.argmax(cum >= 0.95 * tot)) + 152
    d50 = int(np.argmax(cum >= 0.50 * tot)) + 152
    return dict(total=tot, d50=d50, d95=d95, late_share=100 * np.nansum(v[F]) / tot)


recs = []
for s in ids:
    i = sid_all.index(s)
    for y in yrs:
        for src, A in (("gauge", V), ("era5", E)):
            v = A[i, yr == y]
            if v.size == 366:
                v = np.delete(v, 59)
            if src == "era5":
                g = V[i, yr == y]
                if g.size == 366:
                    g = np.delete(g, 59)
                if np.isfinite(g[S]).mean() < 0.95:      # same station-years as the gauge analysis
                    continue
            m = metrics(v)
            if m:
                recs.append(dict(station_id=s, year=int(y), source=src, **m))
M = pd.DataFrame(recs).merge(sp[["station_id", "region"]], on="station_id")
M.to_csv(OUT / "robust_late_season_metrics.csv", index=False)
rows = []
for src in ("gauge", "era5"):
    for reg, sub in list(M[M.source == src].groupby("region")) + [("All", M[M.source == src])]:
        for var in ("late_share", "d95", "d50", "total"):
            P = sub.pivot(index="station_id", columns="year", values=var).reindex(columns=yrs)
            keep = P[P.notna().sum(axis=1) >= 25]
            if len(keep) < 5:
                continue
            comp = keep.median(axis=0)
            sl, p, n = sen_mk(comp.to_numpy(float), yrs.astype(float))
            st_sl = [sen_mk(r.to_numpy(float), yrs.astype(float))[0] for _, r in keep.iterrows()]
            o_neg, o_pos, pn, pp = field_test(keep.to_numpy(float), yrs.astype(float), nperm=400)
            rows.append(dict(source=src, region=reg, variable=var, n_stations=len(keep), composite_slope_per_decade=sl,
                             composite_p=p, station_median_slope=np.nanmedian(st_sl), local_sig_neg=o_neg,
                             local_sig_pos=o_pos, field_p_neg=pn, field_p_pos=pp))
R = pd.DataFrame(rows)
R.to_csv(OUT / "robust_late_season_trends.csv", index=False)
print("\nB. WiKS-FREE LATE-SEASON TRENDS 1981-2017 (late_share in % of Jun-Oct total per decade; d95/d50 in days per decade):")
print(R[R.variable.isin(["late_share", "d95", "d50"])].round(3).to_string(index=False))

# ---------------- C: fixed-network composites of WiKS dates ----------------
rows = []
era = pd.read_csv(OUT / "wiks_era5_dates.csv", dtype={"station_id": str})
era = era[era.version == "ge1mm"].merge(sp[["station_id", "region"]], on="station_id")
for src, df in (("gauge", dates), ("era5", era)):
    for ev in ("onset", "withdrawal"):
        for reg, sub in list(df.groupby("region")) + [("All", df)]:
            P = sub.pivot(index="station_id", columns="year", values=ev).reindex(columns=yrs)
            fixed = P[P.notna().sum(axis=1) >= 30]
            if len(fixed) < 5:
                continue
            comp = fixed.median(axis=0)
            sl, p, n = sen_mk(comp.to_numpy(float), yrs.astype(float))
            rows.append(dict(source=src, event=ev, region=reg, n_stations=len(fixed), slope_d_per_decade=sl, p=p))
C = pd.DataFrame(rows)
C.to_csv(OUT / "robust_fixed_network.csv", index=False)
print("\nC. FIXED-NETWORK COMPOSITE TRENDS (stations with >= 30 of 37 dates), days per decade:")
print(C.round(3).to_string(index=False))

# ---------------- D: declared series ----------------
dec = pd.read_csv(OUT / "dhm_declared_dates.csv")
print("\nD. DHM-DECLARED SERIES, Sen slope (days per decade) [Kendall p]:")
for lab, sub in (("1968-2024", dec), ("1981-2017", dec[(dec.year >= 1981) & (dec.year <= 2017)])):
    out = []
    for col in ("onset_doy", "withdrawal_doy", "length_d"):
        z = sub.dropna(subset=[col])
        out.append("%s %+.1f [p=%.3g]" % (col, stats.theilslopes(z[col], z.year)[0] * 10, stats.kendalltau(z.year, z[col])[1]))
    print(lab, "|", " | ".join(out))
