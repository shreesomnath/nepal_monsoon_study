"""Step 71: direct tests of declared versus objective timing.

A  Difference series (declared minus composite, per year, 1981-2017): Sen slope and the step
   (mean 1997-2017 minus mean 1981-96) with moving-block bootstrap intervals (block 4 years for slopes,
   3 years within periods for the step), so that serial dependence is respected. Also the block-bootstrap
   interval for the declared series alone.
B  Reliability ceiling: interannual correlation of each composite with the declared series and with the other
   independent products (ERA5, OLR), with pair-bootstrap intervals, and the bootstrap difference between
   r(gauge, independent product) and r(gauge, declared).
C  Network robustness of the gauge withdrawal composite: stricter fixed network (at least 35 of 37 dates),
   leave-one-province-out, elevation bands.
Outputs: outputs/diff_*.csv and printed tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

rng = np.random.default_rng(71)
NB = 4000
yrs = np.arange(1981, 2018)
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
SETS = {"All": lambda df: df, "East provinces": lambda df: df[df.region == "East"], "East of 86E": lambda df: df[df.lon >= 86.0]}


def load(name):
    return pd.read_csv(OUT / name, dtype={"station_id": str}).merge(sp[["station_id", "region", "lon", "elev", "province"]], on="station_id")


wk, cbd, era = load("wiks_station_dates_qc.csv"), load("cb_station_dates.csv"), load("wiks_era5_dates.csv")
era = era[era.version == "ge1mm"]
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")
olr = pd.read_csv(OUT / "olr_dates.csv")


def comp(df, col, min_dates=30):
    P = df.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
    P = P[P.notna().sum(axis=1) >= min_dates]
    return P.median(axis=0), len(P)


def slope10(y, t=yrs.astype(float)):
    ok = np.isfinite(y)
    return stats.theilslopes(y[ok], t[ok])[0] * 10


def block_idx(n, L):
    nb = int(np.ceil(n / L))
    s = rng.integers(0, n - L + 1, nb)
    return np.concatenate([np.arange(a, a + L) for a in s])[:n]


def boot_slope(y, L=4):
    n = len(y)
    return np.array([_slope_idx(y, block_idx(n, L)) for _ in range(NB)])


def _slope_idx(y, idx):
    """Theil-Sen slope of the resampled (year, value) pairs."""
    ok = np.isfinite(y[idx])
    if ok.sum() < 15 or np.ptp(idx[ok]) == 0:
        return np.nan
    return stats.theilslopes(y[idx][ok], yrs[idx][ok].astype(float))[0] * 10


def boot_step(y, L=3):
    a, b = y[yrs <= 1996], y[yrs >= 1997]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = []
    for _ in range(NB):
        out.append(b[block_idx(len(b), L)].mean() - a[block_idx(len(a), L)].mean())
    return a, b, np.array(out)


def ci(x):
    return np.nanpercentile(x, 2.5), np.nanpercentile(x, 97.5)


# ---------------- series ----------------
series = {}
for ev, dcol in (("onset", "onset_doy"), ("withdrawal", "withdrawal_doy")):
    series[("Declared", ev)] = dec[dcol].reindex(yrs)
    for sname, fn in SETS.items():
        series[(f"Gauge WiKS {sname}", ev)] = comp(fn(wk), ev)[0]
        series[(f"Gauge two-knot {sname}", ev)] = comp(fn(cbd), "cb_" + ev)[0]
        series[(f"ERA5 WiKS {sname}", ev)] = comp(fn(era), ev)[0]
series[("OLR 4 cells", "onset")] = olr[olr.series == "east_4cells"].set_index("year")["onset"].reindex(yrs)
series[("OLR 4 cells", "withdrawal")] = olr[olr.series == "east_4cells"].set_index("year")["wd_250_15"].reindex(yrs)

# ---------------- A: difference series ----------------
rows = []
for ev in ("onset", "withdrawal"):
    D = series[("Declared", ev)].to_numpy(float)
    bs = boot_slope(D)
    a, b, st = boot_step(D)
    rows.append(dict(event=ev, series="Declared alone", slope=slope10(D), slope_lo=ci(bs)[0], slope_hi=ci(bs)[1],
                     step=b.mean() - a.mean(), step_lo=ci(st)[0], step_hi=ci(st)[1]))
    for (name, e), s in series.items():
        if e != ev or name == "Declared":
            continue
        d = D - s.to_numpy(float)
        bs = boot_slope(d)
        a, b, st = boot_step(d)
        tau, p = stats.kendalltau(yrs[np.isfinite(d)], d[np.isfinite(d)])
        rows.append(dict(event=ev, series=f"Declared minus {name}", slope=slope10(d), slope_lo=ci(bs)[0], slope_hi=ci(bs)[1],
                         slope_boot_p=2 * min((bs <= 0).mean(), (bs >= 0).mean()), step=b.mean() - a.mean(),
                         step_lo=ci(st)[0], step_hi=ci(st)[1]))
A = pd.DataFrame(rows)
A.to_csv(OUT / "diff_series_tests.csv", index=False)
print("A. DECLARED MINUS COMPOSITE: Sen slope (d/decade) and step 1997-2017 minus 1981-96 (d), block-bootstrap 95% CI")
print(A.round(2).to_string(index=False))

# ---------------- B: correlations and ceiling ----------------
def pboot_r(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    bs = []
    for _ in range(NB):
        i = rng.integers(0, n, n)
        bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r, ci(np.array(bs))


rows = []
for ev in ("onset", "withdrawal"):
    names = [n for (n, e) in series if e == ev]
    S = {n: series[(n, ev)].to_numpy(float) for n in names}
    for sname in SETS:
        g = S[f"Gauge WiKS {sname}"]
        for other in ["Declared", f"ERA5 WiKS {sname}", "OLR 4 cells", f"Gauge two-knot {sname}"]:
            r, (lo, hi) = pboot_r(g, S[other])
            rows.append(dict(event=ev, gauge=f"Gauge WiKS {sname}", other=other, r=r, lo=lo, hi=hi))
B = pd.DataFrame(rows)
B.to_csv(OUT / "diff_correlations.csv", index=False)
print("\nB. INTERANNUAL CORRELATION (1981-2017) OF GAUGE WiKS COMPOSITES WITH OTHER SERIES")
print(B.round(2).to_string(index=False))
# difference r(gauge, OLR) - r(gauge, declared) with paired bootstrap
rows = []
for sname in SETS:
    g, o, d = (series[(f"Gauge WiKS {sname}", "withdrawal")].to_numpy(float), series[("OLR 4 cells", "withdrawal")].to_numpy(float),
               series[("Declared", "withdrawal")].to_numpy(float))
    e = series[(f"ERA5 WiKS {sname}", "withdrawal")].to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(o) & np.isfinite(d) & np.isfinite(e)
    g, o, d, e = g[ok], o[ok], d[ok], e[ok]
    for lab, x in (("OLR", o), ("ERA5", e)):
        bs = []
        for _ in range(NB):
            i = rng.integers(0, ok.sum(), ok.sum())
            bs.append(np.corrcoef(g[i], x[i])[0, 1] - np.corrcoef(g[i], d[i])[0, 1])
        rows.append(dict(set=sname, product=lab, r_product=np.corrcoef(g, x)[0, 1], r_declared=np.corrcoef(g, d)[0, 1],
                         diff=np.corrcoef(g, x)[0, 1] - np.corrcoef(g, d)[0, 1], lo=ci(np.array(bs))[0], hi=ci(np.array(bs))[1]))
Bd = pd.DataFrame(rows)
Bd.to_csv(OUT / "diff_correlation_contrast.csv", index=False)
print("\nB2. r(gauge, independent product) minus r(gauge, declared), withdrawal, paired bootstrap")
print(Bd.round(2).to_string(index=False))
sh = pd.read_csv(OUT / "validation_more_split_half.csv")
print("\nSplit-half reliability (Spearman-Brown) of the withdrawal composites, from step 67:")
print(sh[sh.event == "withdrawal"][["definition", "set", "n_stations", "spearman_brown"]].round(2).to_string(index=False))

# ---------------- C: network robustness of the gauge withdrawal composite ----------------
D = series[("Declared", "withdrawal")].to_numpy(float)
variants = {}
variants["All, >=30 dates (primary)"] = wk
variants["All, >=35 dates"] = wk
for prov in sorted(wk.province.dropna().unique()):
    variants[f"All without {prov}"] = wk[wk.province != prov]
for lab, lo, hi in (("below 1000 m", -1, 1000), ("1000 to 2000 m", 1000, 2000), ("2000 m and above", 2000, 9000)):
    variants[f"Elevation {lab}"] = wk[(wk.elev >= lo) & (wk.elev < hi)]
rows = []
for name, df in variants.items():
    s, n = comp(df, "withdrawal", 35 if ">=35" in name else 30)
    if n < 8:
        continue
    y = s.to_numpy(float)
    bs = boot_slope(y)
    r = np.corrcoef(y[np.isfinite(y)], D[np.isfinite(y)])[0, 1]
    rows.append(dict(variant=name, n_stations=n, slope=slope10(y), lo=ci(bs)[0], hi=ci(bs)[1], r_declared=r,
                     declared_slope_excluded=bool(not (ci(bs)[0] <= 6.2 <= ci(bs)[1]))))
C = pd.DataFrame(rows)
C.to_csv(OUT / "diff_network_robustness.csv", index=False)
print("\nC. GAUGE WiKS WITHDRAWAL COMPOSITE SLOPE (d/decade, block-bootstrap 95% CI) BY NETWORK VARIANT")
print(C.round(2).to_string(index=False))
