"""Step 72: many-replicate version of the end-to-end recovery test of code/59 (Part 2).

The full design of code/59 costs about 4 minutes per replicate (2,220 WiKS detections), so 100+ replicates
are built in two stages that reuse the same simulator and the same frozen WiKS rule:
  1. Error library: for each station type (humid, dry_west, high_mountain) and transition width (0, 10, 25 d),
     1,540 synthetic seasons with a withdrawal drawn uniformly between day 250 and 290, 5% of days missing,
     are run through wiks.detect; the error (detected minus true withdrawal) is stored, NaN if undetected.
  2. Replicates: 60 stations (20 humid, 25 dry_west, 15 high_mountain, widths cycling 0/10/25) over 37 years,
     true withdrawal = 268 + drift * (year - 1999) / 10 + regional anomaly (SD 6) + station anomaly (SD 6),
     detected date = true date + an error drawn from the library for that station's type and width. The composite
     is the median over stations, with its Theil-Sen slope and Mann-Kendall p. 1,000 replicates for each of the
     drifts 0, 2, 3, 4, 6.2 days per decade.
The approximation is that detection error does not depend on the date of the withdrawal within 250-290 d and
on the onset; code/59 (4 replicates, full simulation) is the check.
Usage: python 72_synthetic_recovery_100.py
Outputs: outputs/recovery_error_library.csv, outputs/recovery_synthetic_many.csv
"""
import importlib.util
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import OUT, RULE, YEARS  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)
yrs = np.array(list(YEARS))
NJOB = 140          # seasons per job; 11 jobs per type-width combination give 1,540


def lib_job(args):
    seed, tname, width = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(NJOB):
        O = int(rng.integers(150, 185))
        W = int(rng.integers(250, 291))
        x = syn.simulate(rng, O, W, width, syn.TYPES[tname], 0.05)
        r = wiks.detect(x, **RULE)
        ok = r["withdrawal"] is not None and np.isfinite(r["withdrawal"])
        out.append(dict(type=tname, width=width, error=(r["withdrawal"] - W) if ok else np.nan))
    return out


if __name__ == "__main__":
    lib_path = OUT / "recovery_error_library.csv"
    if lib_path.exists():
        lib = pd.read_csv(lib_path)
    else:
        t0 = time.time()
        jobs = [(500 + 10 * i + j, t, w) for i, t in enumerate(["humid", "dry_west", "high_mountain"]) for j, w in enumerate([0, 10, 25])]
        # split each combination into 11 chunks so that all cores are used
        jobs = [(s * 100 + k, t, w) for (s, t, w) in jobs for k in range(11)]
        with Pool(11) as pool:
            res = pool.map(lib_job, jobs, chunksize=1)
        lib = pd.DataFrame([r for chunk in res for r in chunk])
        lib.to_csv(lib_path, index=False)
        print(f"library built in {time.time() - t0:.0f} s, {len(lib)} seasons", flush=True)
    print(lib.groupby(["type", "width"]).error.agg(n="size", missing=lambda s: s.isna().mean(), bias="mean", sd="std", mae=lambda s: s.abs().mean()).round(2).to_string())

    types = ["humid"] * 20 + ["dry_west"] * 25 + ["high_mountain"] * 15
    widths = [[0, 10, 25][i % 3] for i in range(60)]
    errs = {(t, w): lib[(lib.type == t) & (lib.width == w)].error.to_numpy() for t in set(types) for w in (0, 10, 25)}
    rng = np.random.default_rng(72)
    NREP = 1000
    rows = []
    for drift in (0.0, 2.0, 3.0, 4.0, 6.2):
        for rep in range(NREP):
            reg = rng.normal(0, 6, len(yrs))
            A = np.empty((60, len(yrs)))
            for i in range(60):
                true = 268 + drift * (yrs - 1999) / 10.0 + reg + rng.normal(0, 6, len(yrs))
                A[i] = np.round(true) + rng.choice(errs[(types[i], widths[i])], len(yrs))
            comp = np.nanmedian(A, axis=0)
            rows.append(dict(true_drift=drift, rep=rep, recovered_slope=stats.theilslopes(comp, yrs.astype(float))[0] * 10,
                             p=stats.kendalltau(yrs, comp)[1]))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "recovery_synthetic_many.csv", index=False)
    print(R.groupby("true_drift").agg(n=("rep", "count"), mean_slope=("recovered_slope", "mean"), sd_slope=("recovered_slope", "std"),
                                      lo=("recovered_slope", lambda s: s.quantile(0.025)), hi=("recovered_slope", lambda s: s.quantile(0.975)),
                                      detect_rate=("p", lambda s: (s < 0.05).mean())).round(2).to_string())
