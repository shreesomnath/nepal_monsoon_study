"""Second objective definition of monsoon onset and withdrawal: three-segment piecewise-linear fit to the
cumulative rainfall curve (after Cook and Buckley 2009, JGR 114, D23109, who fitted two-phase regressions
to cumulative station rainfall).

The cumulative curve from 1 March to 10 December is fitted with a continuous piecewise-linear function with
two knots, a (onset) and b (withdrawal), by least squares over a grid of knots (step 2 days; onset from
11 April to 15 August, withdrawal from 15 August to 16 November, at least 30 days after the onset). Missing
days are filled with the mean of the observed days within 7 days so that the cumulative curve is continuous
(an assumption that slightly smooths the curve; tested on synthetic data in code/61). Input as in wiks.py:
365 daily values with NaN for unobserved days, index 0 = 1 January.
"""
import numpy as np

T0, T1 = 59, 344           # 1 March to 10 December (0-based, end exclusive)
A_RANGE = range(100, 227, 2)
B_RANGE = range(226, 321, 2)
MIN_GAP = 30


def _fill(x):
    y = x.copy()
    bad = np.where(~np.isfinite(y))[0]
    for i in bad:
        w = x[max(0, i - 7):i + 8]
        w = w[np.isfinite(w)]
        y[i] = w.mean() if len(w) else 0.0
    return y


def detect(x):
    y = _fill(x[T0:T1])
    c = np.cumsum(y)
    t = np.arange(T0, T1).astype(float)
    best = (np.inf, np.nan, np.nan)
    for a in A_RANGE:
        ta = np.maximum(t - a, 0.0)
        for b in B_RANGE:
            if b - a < MIN_GAP:
                continue
            tb = np.maximum(t - b, 0.0)
            M = np.column_stack([np.ones_like(t), t, ta, tb])
            coef, res, rank, _ = np.linalg.lstsq(M, c, rcond=None)
            sse = float(((M @ coef - c) ** 2).sum())
            if sse < best[0] and coef[2] > 0 and coef[3] < 0:       # slope increases at onset, decreases at withdrawal
                best = (sse, a + 1, b + 1)
    return dict(onset=best[1], withdrawal=best[2])
