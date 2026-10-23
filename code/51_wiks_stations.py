"""Step 51: WiKS monsoon onset and withdrawal at DHM stations, 1981-2017.

Uses the quality-controlled observed days only (status 0; missing, outside-operation and suspect
days are NaN). The validity rule (RULE) was frozen on synthetic data in code/50b before this
script was run on stations. A station-year is analysed when at least MIN_SHARE of the days from
22 March to 10 December (the span the windows can reach) are observed.
Outputs: outputs/wiks_station_dates.csv (one row per station-year) and
         outputs/wiks_station_summary.csv (one row per station), plus printed summaries.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]       # project root
WORK = ROOT / "monsoon_timing"
OUT = WORK / "outputs"                           # results of this study
INP = ROOT / "outputs"                           # shared inputs: daily_qc.npz, station_province.csv

from config import RULE, MIN_SHARE, YEARS, SPAN  # noqa: E402  (frozen settings, see config.py)


def work(args):
    sid, y, x = args
    r = wiks.detect(x, **RULE)
    r.update(station_id=sid, year=y)
    return r


if __name__ == "__main__":
    d = np.load(INP / "daily_qc.npz", allow_pickle=True)
    X, status = d["X"], d["status"]
    sid = d["station_id"]
    dates = pd.to_datetime(d["dates"])
    yr = dates.year.to_numpy()
    V = np.where(status == 0, X, np.nan)
    jobs = []
    for i in range(len(sid)):
        for y in YEARS:
            v = V[i, yr == y]
            if v.size == 366:
                v = np.delete(v, 59)
            if np.isfinite(v[SPAN]).mean() < MIN_SHARE:
                continue
            jobs.append((sid[i], y, v))
    print("station-years analysed:", len(jobs))
    with Pool(10) as pool:
        rows = pool.map(work, jobs, chunksize=25)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "wiks_station_dates.csv", index=False)

    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                            np.where(sp.province == "Gandaki", "Central", "East"))
    g = df.groupby("station_id")
    summ = pd.DataFrame({
        "n_years": g.size(), "n_onset": g.onset.count(), "n_withdrawal": g.withdrawal.count(),
        "onset_median": g.onset.median(), "onset_sd": g.onset.std(),
        "withdrawal_median": g.withdrawal.median(), "withdrawal_sd": g.withdrawal.std(),
        "length_median": (df.withdrawal - df.onset).groupby(df.station_id).median(),
        "wiks_onset_median": g.wiks_onset.median(), "p_wmw_onset_median": g.p_wmw_onset.median()})
    summ = summ.reset_index().merge(sp[["station_id", "lon", "lat", "elev", "province", "district", "region"]],
                                    on="station_id")
    summ.to_csv(OUT / "wiks_station_summary.csv", index=False)

    ok = summ[summ.n_years >= 15]
    print("stations with >= 15 analysed years:", len(ok), "of", len(summ))
    print("result rate (onset/withdrawal found per analysed season): %.2f / %.2f" % (
        df.onset.notna().mean(), df.withdrawal.notna().mean()))
    print("\nStation climatology by region (median of station medians, days of year):")
    print(ok.groupby("region")[["onset_median", "onset_sd", "withdrawal_median", "withdrawal_sd",
                                "length_median"]].median().round(1).to_string())
    ok = ok.assign(elev_band=pd.cut(ok.elev, [0, 500, 1000, 1500, 2000, 4000]))
    print("\nBy elevation band:")
    print(ok.groupby("elev_band", observed=True)[["onset_median", "withdrawal_median", "length_median"]]
          .agg(["median", "count"]).round(1).to_string())
    from scipy import stats
    print("\nSpearman correlation across stations: onset_median vs lon %.2f, lat %.2f, elev %.2f" % (
        stats.spearmanr(ok.onset_median, ok.lon, nan_policy="omit")[0],
        stats.spearmanr(ok.onset_median, ok.lat, nan_policy="omit")[0],
        stats.spearmanr(ok.onset_median, ok.elev, nan_policy="omit")[0]))
