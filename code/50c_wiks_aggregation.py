"""Step 50c: does averaging the window p-values geometrically (instead of arithmetically) help?

The published WiKS averages the p-values of the window half-widths arithmetically, which is dominated
by the largest p-value (the window that straddles a gradual transition). A geometric mean uses all
windows. This is judged on synthetic data only: accuracy (share of seasons within 7 days and median
absolute error) and the power of the null-based acceptance (a threshold from block-shuffled synthetic
seasons without a transition at the 5% false-alarm rate, computed separately for each aggregation).
The published arithmetic version stays the primary method unless the geometric version is clearly
better on every station type; the other is then reported as a sensitivity analysis.
Outputs: outputs/wiks_aggregation.csv, printed table.
"""
import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wiks  # noqa: E402
from config import OUT, RULE  # noqa: E402

spec = importlib.util.spec_from_file_location("syn", str(Path(__file__).resolve().parent / "50_wiks_synthetic.py"))
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)


def flat_series(rng, par, miss):
    """A season with no transition: a constant mid-level wet-day probability and intensity, with the
    same persistence and missing-day structure as the generator."""
    p = 0.5 * (par["p0"] + par["p1"])
    m = 0.5 * (par["m0"] + par["m1"])
    par2 = dict(p0=p, m0=m, p1=p, m1=m)
    return syn.simulate(rng, 165, 268, 0, par2, miss)


def one(args):
    seed, tname, width, miss, kind = args
    rng = np.random.default_rng(seed)
    O = int(round(rng.normal(165, 8)))
    W = int(round(rng.normal(268, 7)))
    x = syn.simulate(rng, O, W, width, syn.TYPES[tname], miss) if kind == "signal" else \
        flat_series(rng, syn.TYPES[tname], miss)
    rows = []
    for agg in ("arith", "geo"):
        r = wiks.detect(x, agg=agg, **RULE)
        rows.append(dict(agg=agg, kind=kind, type=tname, width=width, missing=miss, true_onset=O + 1,
                         true_withdrawal=W + 1, **{k: r[k] for k in (
                             "onset", "withdrawal", "p_ks_onset", "p_wmw_onset", "p_ks_withdrawal", "p_wmw_withdrawal")}))
    return rows


if __name__ == "__main__":
    jobs, seed = [], 20000
    for tname in syn.TYPES:
        for miss in (0.0, 0.05):
            for _ in range(150):
                seed += 1
                jobs.append((seed, tname, 0, miss, "null"))
        for width in (0, 10, 25):
            for miss in (0.0, 0.05):
                for _ in range(100):
                    seed += 1
                    jobs.append((seed, tname, width, miss, "signal"))
    with Pool(10) as p:
        res = p.map(one, jobs, chunksize=10)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / "wiks_aggregation.csv", index=False)
    out = []
    for agg in ("arith", "geo"):
        nl = df[(df["agg"] == agg) & (df.kind == "null")]
        # station-type-specific null thresholds (5% false alarm) pooled over missing levels
        for tname in syn.TYPES:
            nt = nl[nl.type == tname]
            thr = {c: nt[c].quantile(0.05) for c in ("p_ks_onset", "p_wmw_onset", "p_ks_withdrawal", "p_wmw_withdrawal")}
            sg = df[(df["agg"] == agg) & (df.kind == "signal") & (df.type == tname)]
            for width, g in sg.groupby("width"):
                eo = (g.onset - g.true_onset).abs()
                ew = (g.withdrawal - g.true_withdrawal).abs()
                pass_on = (g.p_ks_onset <= thr["p_ks_onset"]) & (g.p_wmw_onset <= thr["p_wmw_onset"])
                pass_wd = (g.p_ks_withdrawal <= thr["p_ks_withdrawal"]) & (g.p_wmw_withdrawal <= thr["p_wmw_withdrawal"])
                out.append(dict(agg=agg, type=tname, width=width, within7_onset=(eo <= 7).mean(), mae_onset=eo.median(),
                                within7_wd=(ew <= 7).mean(), mae_wd=ew.median(), power_onset=pass_on.mean(),
                                power_wd=pass_wd.mean(),
                                accurate_among_accepted_onset=(eo[pass_on] <= 7).mean() if pass_on.any() else np.nan))
    t = pd.DataFrame(out)
    print(t.round(2).to_string(index=False))
    print("\nmean over scenarios:")
    print(t.groupby("agg")[["within7_onset", "within7_wd", "power_onset", "power_wd", "accurate_among_accepted_onset"]].mean().round(3).to_string())
