"""Step 50: validate the WiKS implementation on synthetic seasons with known onset and withdrawal.

A daily rainfall generator with a two-state (wet/dry) Markov chain and gamma intensities whose
wet-day probability and mean intensity switch between a pre-/post-monsoon regime and a monsoon
regime. The true onset (withdrawal) is the first day of the monsoon (post-monsoon) regime.
Scenarios: three station types (humid, dry-west, high-mountain), transition widths of 0 (step),
10 and 25 days (logistic 10-90% width), and 0% and 15% randomly missing days (blocks of 1-10 days).
Reported: bias, median absolute error, share of years within 3 and 7 days, and failure rate.
Outputs: outputs/wiks_synthetic.csv (one row per synthetic year), printed summary.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]       # project root (shared inputs live in ROOT/outputs)
WORK = ROOT / "monsoon_timing"
OUT = WORK / "outputs"

TYPES = {  # wet-day probability and mean wet-day intensity (mm) outside / inside the monsoon
    "humid": dict(p0=0.20, m0=6.0, p1=0.80, m1=20.0),
    "dry_west": dict(p0=0.10, m0=4.0, p1=0.55, m1=14.0),
    "high_mountain": dict(p0=0.15, m0=3.0, p1=0.45, m1=8.0),
}
WIDTHS = (0, 10, 25)
MISSING = (0.0, 0.15)
N_YEARS = 150
RHO = 0.3


def simulate(rng, O, W, width, par, miss):
    t = np.arange(365)
    if width == 0:
        s = ((t >= O) & (t < W)).astype(float)
    else:
        sc = width / 4.39
        s = 1 / (1 + np.exp(-(t - O) / sc)) * (1 - 1 / (1 + np.exp(-(t - W) / sc)))
    p = par["p0"] + (par["p1"] - par["p0"]) * s
    mu = par["m0"] + (par["m1"] - par["m0"]) * s
    x = np.zeros(365)
    wet = rng.random() < p[0]
    for i in range(365):
        pc = p[i] + RHO * (1 - p[i]) if wet else p[i] * (1 - RHO)
        wet = rng.random() < pc
        if wet:
            x[i] = max(round(rng.gamma(0.7, mu[i] / 0.7), 1), 0.1)
    if miss > 0:
        target = int(miss * 365)
        removed = 0
        while removed < target:
            start, ln = rng.integers(0, 365), rng.integers(1, 11)
            x[start:start + ln] = np.nan
            removed = int(np.isnan(x).sum())
    return x


def one(args):
    seed, tname, width, miss = args
    rng = np.random.default_rng(seed)
    O = int(round(rng.normal(165, 8)))
    W = int(round(rng.normal(268, 7)))
    x = simulate(rng, O, W, width, TYPES[tname], miss)
    r = wiks.detect(x)
    return dict(type=tname, width=width, missing=miss, true_onset=O + 1, true_withdrawal=W + 1,
                onset=r["onset"], withdrawal=r["withdrawal"])


if __name__ == "__main__":
    jobs = []
    seed = 1000
    for tname in TYPES:
        for width in WIDTHS:
            for miss in MISSING:
                for _ in range(N_YEARS):
                    seed += 1
                    jobs.append((seed, tname, width, miss))
    with Pool(10) as pool:
        rows = pool.map(one, jobs, chunksize=20)
    df = pd.DataFrame(rows)
    df["err_onset"] = df.onset - df.true_onset
    df["err_withdrawal"] = df.withdrawal - df.true_withdrawal
    df.to_csv(OUT / "wiks_synthetic.csv", index=False)

    def summ(g, col, ok):
        e = g[col].dropna()
        return pd.Series({
            "n": len(g), "fail_%": 100 * g[col].isna().mean(),
            "bias_d": e.median(), "mae_d": e.abs().median(),
            "within3_%": 100 * (e.abs() <= 3).sum() / len(g),
            "within7_%": 100 * (e.abs() <= 7).sum() / len(g),
            "p90_abs_d": e.abs().quantile(0.9)})

    for col, name in (("err_onset", "ONSET"), ("err_withdrawal", "WITHDRAWAL")):
        t = df.groupby(["type", "width", "missing"]).apply(lambda g: summ(g, col, None))
        print(f"\n{name} error (days; detected minus true):")
        print(t.round(1).to_string())
    print("\nOverall:")
    for col in ("err_onset", "err_withdrawal"):
        e = df[col].dropna()
        print(col, "bias %.1f, median abs %.1f, within 7 d %.0f%%, failures %.1f%%" % (
            e.median(), e.abs().median(), 100 * (e.abs() <= 7).sum() / len(df), 100 * df[col].isna().mean()))
