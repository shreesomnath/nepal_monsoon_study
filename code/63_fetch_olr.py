"""Step 63: fetch a small Nepal subset of NOAA PSL interpolated daily OLR by HTTP range requests.

Approved by the user on 24 Oct 2026 (transfer of about 155 MB, only the Nepal cells kept).
Source: https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc (NOAA PSL, Liebmann and Smith 1996,
BAMS 77:1275-1277; 2.5 degree, daily). The file stores one global map per day (chunks 1 x 73 x 144), so the
cells are read day by day with range requests; the THREDDS/OPeNDAP and ERDDAP routes were unavailable.
Subset: 15 May to 30 November of each year 1981-2017 (the extra week lets the 7-day mean start on 22 May),
latitudes 25, 27.5 and 30 N (index 26, 25, 24), longitudes 80 to 90 E in 2.5 degree steps (index 32 to 36).
Output: inputs/olr_nepal_1981_2017.csv (a few hundred KB).
"""
import time
from multiprocessing import Pool
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd

URL = "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc"
OUT = Path(__file__).resolve().parents[1] / "inputs"
OUT.mkdir(exist_ok=True)
LAT_IDX = [24, 25, 26]               # 30.0, 27.5, 25.0 N
LON_IDX = [32, 33, 34, 35, 36]       # 80.0 ... 90.0 E
BASE = np.datetime64("1800-01-01T00:00")


def open_file():
    fs = fsspec.filesystem("https", client_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (research)"}})
    return h5py.File(fs.open(URL, mode="rb", block_size=64 * 1024, cache_type="none"), "r")


def one_year(y):
    for attempt in range(5):
        try:
            h = open_file()
            t = h["time"][:]
            lat, lon = h["lat"][:], h["lon"][:]
            dates = BASE + (t * 3600).astype("timedelta64[s]")
            sel = np.where((dates >= np.datetime64(f"{y}-05-15")) & (dates <= np.datetime64(f"{y}-11-30")))[0]
            ds = h["olr"]
            rows = []
            for i in sel:
                a = ds[i, LAT_IDX[0]:LAT_IDX[-1] + 1, LON_IDX[0]:LON_IDX[-1] + 1]
                for ja, la in enumerate(LAT_IDX):
                    for jb, lo in enumerate(LON_IDX):
                        rows.append((str(dates[i])[:10], float(lat[la]), float(lon[lo]), float(a[ja, jb])))
            print(y, "ok", len(sel), "days", flush=True)
            return rows
        except Exception as e:
            print(y, "attempt", attempt, "error", str(e)[:100], flush=True)
            time.sleep(10 * (attempt + 1))
    return []


if __name__ == "__main__":
    t0 = time.time()
    with Pool(8) as p:
        res = p.map(one_year, list(range(1981, 2018)), chunksize=1)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "olr"])
    df["olr"] = df.olr.where((df.olr > 0) & (df.olr < 500))     # fill values and nonsense to NaN
    df.to_csv(OUT / "olr_nepal_1981_2017.csv", index=False)
    print("saved", len(df), "rows,", round((OUT / "olr_nepal_1981_2017.csv").stat().st_size / 1e6, 2), "MB; missing",
          int(df.olr.isna().sum()), "; elapsed", round(time.time() - t0), "s")
    print(df.describe())
