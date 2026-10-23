"""Step 59: could the gauge composite detect a lengthening like DHM's declared +6 days per decade?

Part 1: 95% intervals of the gauge composite slopes (Theil-Sen) against the DHM-declared slopes for
        1981-2017, for all stations, East, and a Koshi-like subset (stations east of 86 E, closest to
        'eastern Nepal' in the DHM bulletin).
Part 2: end-to-end recovery test. Synthetic networks of stations (humid, dry-west, high-mountain mix,
        5% missing days) over 37 years with a withdrawal that drifts later by 0 or 6.2 days per decade
        (onset fixed), plus year-to-year variability common to all stations (a regional anomaly of
        SD 6 days) and station-specific variability. The WiKS composite (median over stations per year)
        is computed exactly as for the real stations and its Theil-Sen slope and Mann-Kendall p are
        recorded over replications.
Outputs: outputs/recovery_*.csv and printed results.
"""
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import INP, OUT, RULE, YEARS  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)
yrs = np.array(list(YEARS))


def one_station(args):
    seed, tname, width, drift, regional_o, regional_w, miss = args
    rng = np.random.default_rng(seed)
    out = []
    for k, y in enumerate(yrs):
        O = int(round(165 + regional_o[k] + rng.normal(0, 6)))
        W = int(round(268 + drift * (y - 1999) / 10.0 + regional_w[k] + rng.normal(0, 6)))
        x = syn.simulate(rng, O, W, width, syn.TYPES[tname], miss)
        r = wiks.detect(x, **RULE)
        out.append((r["onset"], r["withdrawal"]))
    return out


if __name__ == "__main__":
    # ---------- Part 1 ----------
    dates = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                            np.where(sp.province == "Gandaki", "Central", "East"))
    dates = dates.merge(sp[["station_id", "region", "lon"]], on="station_id")
    dec = pd.read_csv(OUT / "dhm_declared_dates.csv")
    dec = dec[(dec.year >= 1981) & (dec.year <= 2017)]
    rows = []
    for lab, sub in (("All", dates), ("East province group", dates[dates.region == "East"]),
                     ("East of 86E", dates[dates.lon >= 86.0])):
        for ev, dcol in (("onset", "onset_doy"), ("withdrawal", "withdrawal_doy")):
            P = sub.pivot(index="station_id", columns="year", values=ev).reindex(columns=yrs)
            keep = P[P.notna().sum(axis=1) >= 30]
            comp = keep.median(axis=0).dropna()
            sl, ic, lo, hi = stats.theilslopes(comp.to_numpy(float), comp.index.to_numpy(float), alpha=0.95)
            z = dec.dropna(subset=[dcol])
            sd = stats.theilslopes(z[dcol], z.year)[0] * 10
            rows.append(dict(set=lab, event=ev, n_stations=len(keep), gauge_slope=sl * 10, ci_lo=lo * 10, ci_hi=hi * 10,
                             declared_slope=sd, declared_inside_ci=bool(lo * 10 <= sd <= hi * 10)))
    p1 = pd.DataFrame(rows)
    p1.to_csv(OUT / "recovery_ci_vs_declared.csv", index=False)
    print("PART 1. Gauge composite slope (days/decade, 95% CI) vs DHM-declared 1981-2017:")
    print(p1.round(2).to_string(index=False))

    # ---------- Part 2 ----------
    NST, NREP = 60, 4
    types = ["humid"] * 20 + ["dry_west"] * 25 + ["high_mountain"] * 15
    widths = [0, 10, 25]
    res = []
    for drift in (0.0, 6.2):
        for rep in range(NREP):
            rng = np.random.default_rng(777 + rep + int(drift * 10))
            reg_o = rng.normal(0, 6, len(yrs))
            reg_w = rng.normal(0, 6, len(yrs))
            jobs = [(10000 * rep + 100 * i + int(drift * 10), types[i], widths[i % 3], drift, reg_o, reg_w, 0.05)
                    for i in range(NST)]
            with Pool(10) as p:
                out = p.map(one_station, jobs, chunksize=2)
            A = np.array(out, dtype=float)            # stations x years x (onset, withdrawal)
            for ev, idx in (("onset", 0), ("withdrawal", 1)):
                comp = np.nanmedian(A[:, :, idx], axis=0)
                sl = stats.theilslopes(comp, yrs.astype(float))[0] * 10
                tau, pv = stats.kendalltau(yrs, comp)
                res.append(dict(true_drift=drift, rep=rep, event=ev, recovered_slope=sl, p=pv))
            print("rep", rep, "drift", drift, "done", flush=True)
    r2 = pd.DataFrame(res)
    r2.to_csv(OUT / "recovery_synthetic.csv", index=False)
    print("\nPART 2. Recovery of a built-in withdrawal drift by the WiKS composite (60 stations, 5% missing):")
    print(r2.groupby(["true_drift", "event"]).agg(mean_slope=("recovered_slope", "mean"), sd_slope=("recovered_slope", "std"),
                                                   detect_rate=("p", lambda s: (s < 0.05).mean())).round(2).to_string())
