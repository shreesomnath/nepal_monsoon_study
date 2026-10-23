"""Step 62: power of the real gauge composites to detect a withdrawal trend, using the real noise.

The synthetic recovery test (code/59) has a composite far less noisy than the real one. Here the noise is
taken from the real gauge composites: the detrended composite residuals are resampled (years drawn with
replacement, 2000 times) after adding a linear trend of a given size, and the share of replicates with a
Mann-Kendall p below 0.05 and with the Sen slope inside the real interval is recorded. Reported: the
residual SD of the real composites, the minimal trend detectable with 80% power, and the power for the
declared +6.2 days per decade.
Outputs: outputs/power_real_noise.csv and printed table.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, YEARS  # noqa: E402

yrs = np.array(list(YEARS)).astype(float)
rng = np.random.default_rng(62)
dates = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
dates = dates.merge(sp[["station_id", "region", "lon"]], on="station_id")
rows = []
for lab, sub in (("All", dates), ("East province group", dates[dates.region == "East"]), ("East of 86E", dates[dates.lon >= 86.0])):
    P = sub.pivot(index="station_id", columns="year", values="withdrawal").reindex(columns=YEARS)
    P = P[P.notna().sum(axis=1) >= 30]
    comp = P.median(axis=0).dropna()
    t = comp.index.to_numpy(float)
    sl, ic, _, _ = stats.theilslopes(comp.to_numpy(float), t)
    resid = comp.to_numpy(float) - (ic + sl * t)
    sd = resid.std(ddof=2)
    lag1 = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    out = dict(set=lab, n_stations=len(P), resid_sd_d=sd, lag1=lag1)
    for trend in (0.0, 2.0, 3.0, 4.0, 5.0, 6.2):
        det = 0
        for _ in range(2000):
            r = rng.choice(resid, size=len(resid), replace=True)
            y = r + trend / 10.0 * (t - t.mean())
            tau, p = stats.kendalltau(t, y)
            det += p < 0.05
        out[f"power_{trend}"] = det / 2000
    rows.append(out)
res = pd.DataFrame(rows)
res.to_csv(OUT / "power_real_noise.csv", index=False)
print(res.round(3).to_string(index=False))
