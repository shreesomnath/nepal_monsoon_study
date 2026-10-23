import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv(r'F:\nepal_research\monsoon_timing\outputs\recent_years_withdrawal_by_year.csv', index_col=0)
o = pd.read_csv(r'F:\nepal_research\monsoon_timing\outputs\recent_years_olr_withdrawal_anomaly.csv', index_col=0)
d['OLR 4 cells'] = o.iloc[:, 0]
rng = np.random.default_rng(5)
for c in d.columns:
    s = d[c].dropna()
    a, b = s[s.index <= 2017].to_numpy(), s[s.index >= 2018].to_numpy()
    diff = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t = stats.ttest_ind(b, a, equal_var=False)
    u = stats.mannwhitneyu(b, a, alternative='greater')
    # permutation of year labels (whole years) for the mean difference
    allv = np.concatenate([a, b]); nb = len(b)
    perm = np.array([rng.permutation(allv)[:nb].mean() - rng.permutation(allv)[nb:].mean() for _ in range(5000)])
    # better: one permutation per draw
    pm = []
    for _ in range(5000):
        p = rng.permutation(allv); pm.append(p[:nb].mean() - p[nb:].mean())
    print(f"{c[:55]:55s} diff {diff:+6.1f}  year-based SE {se:4.1f}  95% CI {diff - 2.03 * se:+6.1f} to {diff + 2.03 * se:+6.1f}  perm p(one-sided) {(np.array(pm) >= diff).mean():.3f}  n_after {len(b)}  sd_before {a.std(ddof=1):.1f}")
    print('    2018-2023 values:', np.round(b, 1), ' ranks among all years of 2018-23 values:', [int((allv >= v).sum()) for v in b])
