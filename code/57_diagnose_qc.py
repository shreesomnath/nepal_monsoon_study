"""Step 57: diagnose why the null-based QC accepts few dates, especially at low elevation.

Compares the distributions of the best-change-point p-values in real station-years and in
block-shuffled seasons, by elevation band, and the power of the QC threshold on synthetic seasons
with a known transition (the generator of code/50)."""
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import OUT, RULE, INP  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)


def one(args):
    seed, tname, width, miss = args
    rng = np.random.default_rng(seed)
    O = int(round(rng.normal(165, 8)))
    W = int(round(rng.normal(268, 7)))
    x = syn.simulate(rng, O, W, width, syn.TYPES[tname], miss)
    r = wiks.detect(x, **RULE)
    return dict(type=tname, width=width, missing=miss, **{k: r[k] for k in (
        "p_ks_onset", "p_wmw_onset", "p_ks_withdrawal", "p_wmw_withdrawal")})


if __name__ == "__main__":
    real = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    real = real.merge(sp[["station_id", "elev"]], on="station_id")
    null = pd.read_csv(OUT / "wiks_null.csv", dtype={"station_id": str}).merge(sp[["station_id", "elev"]], on="station_id")
    qs = [0.05, 0.25, 0.5, 0.75]
    bands = [0, 500, 1000, 1500, 2000, 4000]
    for nm, d in (("REAL", real), ("NULL (shuffled)", null)):
        d["band"] = pd.cut(d.elev, bands)
        print(f"\n{nm}: quantiles of log10 p_wmw_onset by elevation band")
        print(d.groupby("band", observed=True).p_wmw_onset.apply(lambda s: np.log10(s.clip(1e-300)).quantile(qs)).unstack().round(1).to_string())
    # power of the QC threshold on synthetic seasons with a real transition
    thr = pd.read_csv(OUT / "wiks_null_thresholds.csv").set_index(["event", "test"])["q05"]
    jobs, seed = [], 9000
    for t in syn.TYPES:
        for w in (0, 10, 25):
            for m in (0.0, 0.05):
                for _ in range(100):
                    seed += 1
                    jobs.append((seed, t, w, m))
    with Pool(10) as p:
        rows = p.map(one, jobs, chunksize=10)
    s = pd.DataFrame(rows)
    s["pass_onset"] = (s.p_ks_onset <= thr[("onset", "ks")]) & (s.p_wmw_onset <= thr[("onset", "wmw")])
    s["pass_withdrawal"] = (s.p_ks_withdrawal <= thr[("withdrawal", "ks")]) & (s.p_wmw_withdrawal <= thr[("withdrawal", "wmw")])
    print("\nPOWER of the 5% null threshold on synthetic seasons with a true transition (share accepted):")
    print(s.groupby(["type", "width"])[["pass_onset", "pass_withdrawal"]].mean().round(2).to_string())
    print("\nmedian log10 p (onset WMW) synthetic by type/width:")
    print(s.groupby(["type", "width"]).p_wmw_onset.apply(lambda x: np.log10(x.clip(1e-300)).median()).round(1).to_string())
    s.to_csv(OUT / "wiks_qc_power_synthetic.csv", index=False)
