"""Step 68: why does the published OLR onset correlate 0.53 with DHM's declared onset and ours 0.09?

Tests how the correlation between an OLR 'characteristic fall' date and DHM's declared onset (1981-2017) varies
with the analyst's choices, using the 2.5 degree PSL OLR cells fetched for Nepal:
  - the set of cells (every contiguous rectangle of the 3 x 5 cell grid: 90 sets),
  - the threshold (240, 245, 250, 255, 260 W m-2),
  - the rule (first day below the threshold after 21 May, or Adhikari et al.'s 3-day decreasing rule with a fall of
    at least 5 W m-2).
Also reported: the share of years in which the date is stuck on the first allowed day (22 May; a floor effect that
removes interannual variance), Fisher-z confidence intervals for the two correlations, and the correlation with the
gauge composite onset. The point is to show the spread of r across defensible choices; the best combination is
not a result (it is selected on the outcome).
Outputs: outputs/olr_onset_sensitivity.csv and printed summaries.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT, WORK  # noqa: E402

yrs = np.arange(1981, 2018)
o = pd.read_csv(WORK / "inputs" / "olr_nepal_1981_2017.csv", parse_dates=["time"]).rename(columns={"latitude": "lat", "longitude": "lon"})
cube = {}
for (la, lo), g in o.groupby(["lat", "lon"]):
    cube[(la, lo)] = g.set_index("time").olr.sort_index()
LATS, LONS = [25.0, 27.5, 30.0], [80.0, 82.5, 85.0, 87.5, 90.0]
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year").onset_doy.reindex(yrs)
wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
gp = wk.pivot(index="station_id", columns="year", values="onset").reindex(columns=yrs)
gauge_on = gp[gp.notna().sum(axis=1) >= 30].median(axis=0)


def fall(ma, y, thr, rule):
    s = ma[(ma.index >= f"{y}-05-22") & (ma.index <= f"{y}-08-31")]
    v = s.to_numpy()
    for i in range(len(v) - 2):
        if rule == "simple":
            if v[i] < thr:
                return s.index[i].dayofyear
        else:
            if v[i] < thr and v[i + 1] < thr and v[i + 2] < thr and v[i] >= v[i + 1] >= v[i + 2] and (v[i] - v[i + 2]) >= 5.0:
                return s.index[i].dayofyear
    return np.nan


def fisher_ci(r, n):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


rows = []
rects = [(la0, la1, lo0, lo1) for la0 in range(3) for la1 in range(la0, 3) for lo0 in range(5) for lo1 in range(lo0, 5)]
for (la0, la1, lo0, lo1) in rects:
    cells = [(LATS[i], LONS[j]) for i in range(la0, la1 + 1) for j in range(lo0, lo1 + 1)]
    ser = pd.concat([cube[c] for c in cells], axis=1).mean(axis=1)
    ma = ser.rolling(7).mean()
    for thr in (240.0, 245.0, 250.0, 255.0, 260.0):
        for rule in ("simple", "adhikari"):
            f = pd.Series({y: fall(ma, y, thr, rule) for y in yrs})
            ok = f.notna() & dec.notna()
            if ok.sum() < 30 or f[ok].std() == 0:
                continue
            r = np.corrcoef(f[ok], dec[ok])[0, 1]
            rg = np.corrcoef(f[ok & gauge_on.notna()], gauge_on[ok & gauge_on.notna()])[0, 1]
            rows.append(dict(lat_cells=f"{LATS[la0]}-{LATS[la1]}", lon_cells=f"{LONS[lo0]}-{LONS[lo1]}", n_cells=len(cells), thr=thr,
                             rule=rule, n=int(ok.sum()), mean_doy=f[ok].mean(), sd_d=f[ok].std(),
                             floor_share=float((f[ok] == 142).mean()), r_declared=r, r_gauge=rg))
S = pd.DataFrame(rows)
S.to_csv(OUT / "olr_onset_sensitivity.csv", index=False)
print("combinations:", len(S))
print("\nr with DHM declared onset across ALL combinations (1981-2017):")
print(S.r_declared.describe(percentiles=[.05, .25, .5, .75, .95]).round(2).to_string())
print("share with r >= 0.4: %.2f; share with r <= 0.1: %.2f; share with floor_share > 0.5: %.2f" % (
    (S.r_declared >= 0.4).mean(), (S.r_declared <= 0.1).mean(), (S.floor_share > 0.5).mean()))
print("\nBy rule:")
print(S.groupby("rule").r_declared.describe().round(2).to_string())
print("\nBy threshold:")
print(S.groupby("thr").r_declared.describe().round(2).to_string())
print("\nOnly combinations whose mean date is within 5 days of the published 29 May (day 149) and not floor-limited:")
sel = S[(S.mean_doy.between(144, 154)) & (S.floor_share < 0.3)]
print(len(sel), "combinations; r with declared onset:", sel.r_declared.describe(percentiles=[.05, .5, .95]).round(2).to_dict())
print("\nFisher-z 95% CI: published r=0.53 (n=36):", np.round(fisher_ci(0.53, 36), 2), "; ours r=0.09 (n=37):", np.round(fisher_ci(0.09, 37), 2))
z1, z2 = np.arctanh(0.53), np.arctanh(0.09)
se = np.sqrt(1 / 33 + 1 / 34)
print("independent-samples test of the difference: z = %.2f, p = %.3f" % ((z1 - z2) / se, 2 * (1 - stats.norm.cdf(abs((z1 - z2) / se)))))
print("\nCorrelation of the gauge composite onset with the declared onset (1981-2017): %.2f" % np.corrcoef(gauge_on[dec.notna()], dec[dec.notna()])[0, 1])
