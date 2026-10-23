"""Step 51b: WiKS dates for 2018-2023 (out-of-sample extension), gauges and ERA5 at the same station-years.

Same frozen rule and completeness requirement as code/51 (at least 85% of the days 22 March to 10 December
observed). Many stations were silent in 2018-2020 (see Section 3.1 of the other study), so the composites for these
years rest on the stations that kept reporting; ERA5 is analysed for exactly the same station-years.
Outputs: outputs/wiks_station_dates_2018_2023.csv, outputs/wiks_era5_dates_2018_2023.csv.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import INP, MIN_SHARE, OUT, RULE, SPAN  # noqa: E402

NEW = range(2018, 2024)


def work(args):
    sid, y, src, x = args
    r = wiks.detect(x, **RULE)
    r.update(station_id=sid, year=y, source=src)
    return r


if __name__ == "__main__":
    d = np.load(INP / "daily_qc.npz", allow_pickle=True)
    X, status = d["X"], d["status"]
    sid = list(d["station_id"])
    yr = pd.to_datetime(d["dates"]).year.to_numpy()
    V = np.where(status == 0, X, np.nan)
    e = np.load(INP / "era5_station_daily.npz", allow_pickle=True)
    E = e["era5_centre_mm"].astype(float)
    E1 = E.copy()
    E1[E1 < 1.0] = 0.0
    jobs = []
    for i, s in enumerate(sid):
        for y in NEW:
            v = V[i, yr == y]
            if v.size == 366:
                v = np.delete(v, 59)
            if np.isfinite(v[SPAN]).mean() < MIN_SHARE:
                continue
            jobs.append((s, y, "gauge", v))
            w = E1[i, yr == y]
            if w.size == 366:
                w = np.delete(w, 59)
            jobs.append((s, y, "era5", w))
    print("series:", len(jobs))
    with Pool(10) as p:
        rows = p.map(work, jobs, chunksize=20)
    df = pd.DataFrame(rows)
    df[df.source == "gauge"].drop(columns="source").to_csv(OUT / "wiks_station_dates_2018_2023.csv", index=False)
    df[df.source == "era5"].drop(columns="source").to_csv(OUT / "wiks_era5_dates_2018_2023.csv", index=False)
    g = df[df.source == "gauge"]
    print(g.groupby("year").station_id.nunique().to_string())
