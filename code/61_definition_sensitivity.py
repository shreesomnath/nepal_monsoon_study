"""Step 61: definition sensitivity with the cumulative-rainfall two-knot fit (code/cb.py).

1. Synthetic accuracy of the second definition, same generator and scenarios as code/50 (0% and 5% missing).
2. Agreement of its station dates with the WiKS dates (same station-years).
3. Composite trends 1981-2017 and the comparison with the DHM-declared withdrawal.
Outputs: outputs/cb_*.csv and printed tables.
"""
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cb  # noqa: E402
from config import INP, MIN_SHARE, OUT, SPAN, YEARS  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)
yrs = np.array(list(YEARS))


def syn_one(args):
    seed, tname, width, miss = args
    rng = np.random.default_rng(seed)
    O = int(round(rng.normal(165, 8)))
    W = int(round(rng.normal(268, 7)))
    x = syn.simulate(rng, O, W, width, syn.TYPES[tname], miss)
    r = cb.detect(x)
    return dict(type=tname, width=width, missing=miss, err_onset=r["onset"] - (O + 1), err_withdrawal=r["withdrawal"] - (W + 1))


def st_one(args):
    sid, y, x = args
    r = cb.detect(x)
    return dict(station_id=sid, year=y, cb_onset=r["onset"], cb_withdrawal=r["withdrawal"])


if __name__ == "__main__":
    jobs, seed = [], 40000
    for t in syn.TYPES:
        for w in (0, 10, 25):
            for m in (0.0, 0.05):
                for _ in range(80):
                    seed += 1
                    jobs.append((seed, t, w, m))
    with Pool(10) as p:
        s = pd.DataFrame(p.map(syn_one, jobs, chunksize=10))
    s.to_csv(OUT / "cb_synthetic.csv", index=False)
    print("1. SYNTHETIC: share of seasons within 7 days (Cook-Buckley-type fit):")
    for col in ("err_onset", "err_withdrawal"):
        print(col)
        print(s.groupby(["type", "width"])[col].apply(lambda e: (e.abs() <= 7).mean()).unstack().round(2).to_string())

    d = np.load(INP / "daily_qc.npz", allow_pickle=True)
    X, status = d["X"], d["status"]
    sid = list(d["station_id"])
    yr = pd.to_datetime(d["dates"]).year.to_numpy()
    V = np.where(status == 0, X, np.nan)
    wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
    jobs = []
    for s_, y_ in zip(wk.station_id, wk.year):
        v = V[sid.index(s_), yr == y_]
        if v.size == 366:
            v = np.delete(v, 59)
        jobs.append((s_, int(y_), v))
    print("\nstation-years:", len(jobs))
    with Pool(10) as p:
        c = pd.DataFrame(p.map(st_one, jobs, chunksize=25))
    m = wk.merge(c, on=["station_id", "year"])
    m.to_csv(OUT / "cb_station_dates.csv", index=False)
    print("2. AGREEMENT WiKS vs two-knot fit (same station-years):")
    for ev in ("onset", "withdrawal"):
        dd = (m[f"cb_{ev}"] - m[ev]).dropna()
        print(f"{ev}: n {len(dd)}, median diff {dd.median():+.1f} d, median abs {dd.abs().median():.1f} d, within 7 d {(dd.abs() <= 7).mean():.2f}")

    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                            np.where(sp.province == "Gandaki", "Central", "East"))
    m = m.merge(sp[["station_id", "region", "lon"]], on="station_id")
    dec = pd.read_csv(OUT / "dhm_declared_dates.csv")
    dec = dec[(dec.year >= 1981) & (dec.year <= 2017)]
    rows = []
    for lab, sub in (("All", m), ("East province group", m[m.region == "East"]), ("West", m[m.region == "West"]),
                     ("East of 86E", m[m.lon >= 86.0])):
        for ev, dcol in (("onset", "onset_doy"), ("withdrawal", "withdrawal_doy")):
            P = sub.pivot(index="station_id", columns="year", values=f"cb_{ev}").reindex(columns=yrs)
            keep = P[P.notna().sum(axis=1) >= 30]
            comp = keep.median(axis=0).dropna()
            sl, ic, lo, hi = stats.theilslopes(comp.to_numpy(float), comp.index.to_numpy(float))
            z = dec.dropna(subset=[dcol])
            r_ = stats.pearsonr(comp.reindex(z.year).dropna(), z.set_index("year").loc[comp.reindex(z.year).dropna().index, dcol])[0]
            rows.append(dict(set=lab, event=ev, n_stations=len(keep), slope=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                             declared_slope=stats.theilslopes(z[dcol], z.year)[0] * 10, r_with_declared=r_))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cb_trends.csv", index=False)
    print("\n3. TWO-KNOT FIT COMPOSITE TRENDS (days per decade) vs declared:")
    print(t.round(2).to_string(index=False))
