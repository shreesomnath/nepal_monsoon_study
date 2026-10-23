"""Step 50b: choose the missing-data rule for WiKS on synthetic data only (never on the stations).

Sweeps the window validity threshold (share of observed days needed in each half-window) and the
number of window half-widths (of 20, 30, 40 days) that must pass, for 0%, 5% and 15% randomly
missing days in blocks of 1-10 days, using the generator of code/50.
Outputs: outputs/wiks_missing_rule.csv and a printed summary (share of seasons with a result, and
median absolute error and share within 7 days among all seasons).
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)

OUT = Path(__file__).resolve().parents[1] / "outputs"      # monsoon_timing/outputs
RULES = [(0.8, 3), (0.65, 3), (0.5, 3), (0.65, 2), (0.5, 2), (0.4, 1)]
MISS = (0.0, 0.05, 0.15)
WIDTHS = (0, 10, 25)
N = 120


def one(args):
    seed, tname, width, miss = args
    rng = np.random.default_rng(seed)
    O = int(round(rng.normal(165, 8)))
    W = int(round(rng.normal(268, 7)))
    x = syn.simulate(rng, O, W, width, syn.TYPES[tname], miss)
    rows = []
    for mv, mw in RULES:
        r = wiks.detect(x, min_valid=mv, min_windows=mw, min_days=30)
        rows.append(dict(type=tname, width=width, missing=miss, min_valid=mv, min_windows=mw,
                         err_onset=r["onset"] - (O + 1), err_withdrawal=r["withdrawal"] - (W + 1)))
    return rows


if __name__ == "__main__":
    jobs, seed = [], 5000
    for tname in syn.TYPES:
        for width in WIDTHS:
            for miss in MISS:
                for _ in range(N):
                    seed += 1
                    jobs.append((seed, tname, width, miss))
    with Pool(10) as pool:
        res = pool.map(one, jobs, chunksize=10)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / "wiks_missing_rule.csv", index=False)
    for col, nm in (("err_onset", "ONSET"), ("err_withdrawal", "WITHDRAWAL")):
        g = df.groupby(["missing", "min_valid", "min_windows"])[col]
        t = pd.DataFrame({
            "result_%": 100 * g.apply(lambda s: s.notna().mean()),
            "bias_d": g.median(),
            "mae_d": g.apply(lambda s: s.abs().median()),
            "within7_%_of_all": 100 * g.apply(lambda s: (s.abs() <= 7).mean()),
            "within7_%_of_results": 100 * g.apply(lambda s: (s.abs() <= 7).sum() / max(s.notna().sum(), 1))})
        print(f"\n{nm}:")
        print(t.round(1).to_string())
