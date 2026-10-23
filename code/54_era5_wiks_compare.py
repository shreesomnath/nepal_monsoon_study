"""Step 54: do gridded products reproduce gauge-based monsoon dates? ERA5 against DHM stations.

Replicates the product-dependence test of Brunello et al. (2020, Supporting Information SI.3,
where APHRODITE and IMERG gave dates within 7 days only about half the time), now with the DHM
gauges as the reference. WiKS is applied to ERA5 daily precipitation (24 h ending 03 UTC, the DHM
reading time) at each station's grid cell, for the same station-years as the gauge analysis, in three
versions: raw, ERA5 with values below 1 mm set to zero (removes ERA5 drizzle), and the 3x3
neighbourhood mean. Differences are reported as ERA5 minus gauge. ERA5 is a model product and is
not used here as truth for the gauges; the comparison measures how far a gridded product's dates
depart from the gauge dates.
Outputs: outputs/wiks_era5_dates.csv (one row per station-year and version) and
         outputs/wiks_era5_compare.csv (per version and region summary).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import RULE, YEARS, SPAN, INP, OUT  # noqa: E402

VERSIONS = ("raw", "ge1mm", "mean3x3")


def work(args):
    sid, y, ver, x = args
    r = wiks.detect(x, **RULE)
    r.update(station_id=sid, year=y, version=ver)
    return r


if __name__ == "__main__":
    g = pd.read_csv(OUT / "wiks_station_dates.csv", dtype={"station_id": str})
    e = np.load(INP / "era5_station_daily.npz", allow_pickle=True)
    sid = list(e["station_id"])
    dates = pd.to_datetime(e["dates"])
    yr = dates.year.to_numpy()
    series = {"raw": e["era5_centre_mm"].astype(float), "mean3x3": e["era5_3x3_mm"].astype(float)}
    ge = series["raw"].copy()
    ge[ge < 1.0] = 0.0
    series["ge1mm"] = ge
    jobs = []
    for s, y in zip(g.station_id, g.year):
        i = sid.index(s)
        for ver in VERSIONS:
            v = series[ver][i, yr == y]
            if v.size == 366:
                v = np.delete(v, 59)
            jobs.append((s, int(y), ver, v))
    print("ERA5 series to analyse:", len(jobs))
    with Pool(10) as p:
        rows = p.map(work, jobs, chunksize=25)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wiks_era5_dates.csv", index=False)

    m = df.merge(g[["station_id", "year", "onset", "withdrawal"]], on=["station_id", "year"], suffixes=("_era5", "_gauge"))
    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                            np.where(sp.province == "Gandaki", "Central", "East"))
    m = m.merge(sp[["station_id", "region", "elev"]], on="station_id")
    out = []
    for ev in ("onset", "withdrawal"):
        m[f"d_{ev}"] = m[f"{ev}_era5"] - m[f"{ev}_gauge"]
        for (ver, reg), q in list(m.groupby(["version", "region"])) + [((v, "All"), q) for v, q in m.groupby("version")]:
            d = q[f"d_{ev}"].dropna()
            both = q.dropna(subset=[f"{ev}_era5", f"{ev}_gauge"])
            # interannual correlation within stations (anomalies from the station mean)
            an = both.groupby("station_id").apply(
                lambda z: np.corrcoef(z[f"{ev}_era5"] - z[f"{ev}_era5"].mean(),
                                      z[f"{ev}_gauge"] - z[f"{ev}_gauge"].mean())[0, 1] if len(z) >= 12 else np.nan)
            out.append(dict(event=ev, version=ver, region=reg, n=len(d), share_found=q[f"{ev}_era5"].notna().mean(),
                            bias_d=d.median(), mae_d=d.abs().median(), within7=(d.abs() <= 7).mean(),
                            within14=(d.abs() <= 14).mean(), interannual_r_median=an.median()))
    res = pd.DataFrame(out)
    res.to_csv(OUT / "wiks_era5_compare.csv", index=False)
    print(res[res.region == "All"].round(2).to_string(index=False))
    print(res[res.region != "All"].round(2).to_string(index=False))
