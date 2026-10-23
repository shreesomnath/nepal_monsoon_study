"""Step 53: data-driven null for WiKS and the quality-control threshold for station dates.

For each of N randomly chosen analysed station-years the daily values between 22 March and
10 December are shuffled in blocks of 5 days. This keeps the wet-day frequency, intensity
distribution and missing-day positions of the real season but removes any seasonal transition. The
WiKS detector is run on these shuffled seasons. The p-values of the best change point in the
shuffled seasons give the null distribution; a real station date is accepted only if both its KS and
WMW p-values are below the chosen null quantile (5% false-alarm rate by default).
Outputs: outputs/wiks_null.csv (one row per shuffled season) and outputs/wiks_null_thresholds.csv.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import RULE, MIN_SHARE, YEARS, SPAN, INP, OUT  # noqa: E402

N = 3000
BLOCK = 5


def shuffle_blocks(x, rng):
    seg = x[SPAN].copy()
    nb = int(np.ceil(len(seg) / BLOCK))
    order = rng.permutation(nb)
    blocks = [seg[i * BLOCK:(i + 1) * BLOCK] for i in range(nb)]
    out = x.copy()
    out[SPAN] = np.concatenate([blocks[i] for i in order])[:len(seg)]
    return out


def work(args):
    seed, sid, y, x = args
    rng = np.random.default_rng(seed)
    r = wiks.detect(shuffle_blocks(x, rng), **RULE)
    r.update(station_id=sid, year=y)
    return r


if __name__ == "__main__":
    d = np.load(INP / "daily_qc.npz", allow_pickle=True)
    X, status = d["X"], d["status"]
    sid = d["station_id"]
    yr = pd.to_datetime(d["dates"]).year.to_numpy()
    V = np.where(status == 0, X, np.nan)
    pool_ = []
    for i in range(len(sid)):
        for y in YEARS:
            v = V[i, yr == y]
            if v.size == 366:
                v = np.delete(v, 59)
            if np.isfinite(v[SPAN]).mean() >= MIN_SHARE:
                pool_.append((sid[i], y, v))
    rng = np.random.default_rng(53)
    pick = rng.choice(len(pool_), size=min(N, len(pool_)), replace=False)
    jobs = [(int(1000 + k), *pool_[j]) for k, j in enumerate(pick)]
    with Pool(10) as p:
        rows = p.map(work, jobs, chunksize=25)
    null = pd.DataFrame(rows)
    null.to_csv(OUT / "wiks_null.csv", index=False)
    qs = [0.01, 0.05, 0.10]
    thr = []
    for ev in ("onset", "withdrawal"):
        for test in ("ks", "wmw"):
            col = f"p_{test}_{ev}"
            s = null[col].dropna()
            thr.append(dict(event=ev, test=test, found_share=null[col].notna().mean(),
                            **{f"q{int(q * 100):02d}": s.quantile(q) for q in qs}))
    thr = pd.DataFrame(thr)
    thr.to_csv(OUT / "wiks_null_thresholds.csv", index=False)
    print("shuffled seasons:", len(null))
    print(thr.to_string(index=False))
