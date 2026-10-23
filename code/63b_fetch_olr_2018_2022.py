"""Step 63b: extend the OLR subset to 2018-2022 (the PSL file ends on 31 Dec 2022), about 20 MB of range reads.

Same file, cells and method as code/63; only the years differ. The user's rule was to ask first above 50 MB;
this is about one seventh of the approved transfer. Output: inputs/olr_nepal_2018_2022.csv.
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
LAT_IDX = [24, 25, 26]
LON_IDX = [32, 33, 34, 35, 36]
BASE = np.datetime64("1800-01-01T00:00")


def one_year(y):
    for attempt in range(5):
        try:
            fs = fsspec.filesystem("https", client_kwargs={"headers": {"User-Agent": "Mozilla/5.0 (research)"}})
            h = h5py.File(fs.open(URL, mode="rb", block_size=64 * 1024, cache_type="none"), "r")
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
    with Pool(5) as p:
        res = p.map(one_year, list(range(2018, 2023)), chunksize=1)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows, columns=["time", "latitude", "longitude", "olr"])
    df["olr"] = df.olr.where((df.olr > 0) & (df.olr < 500))
    df.to_csv(OUT / "olr_nepal_2018_2022.csv", index=False)
    print("saved", len(df), "rows;", round((OUT / "olr_nepal_2018_2022.csv").stat().st_size / 1e6, 2), "MB; missing", int(df.olr.isna().sum()),
          "; elapsed", round(time.time() - t0), "s")
