"""WiKS monsoon onset and withdrawal detection (Brunello et al. 2020, GRL 47, e2020GL088420).

Method as described in the paper (section 2.1):
  * for every day t between 1 May and 31 October, two samples of daily precipitation, the
    h days before t and the h days from t on, are compared with
      - the two-sample Kolmogorov-Smirnov test (KS, sensitive to any distribution change)
      - the Wilcoxon-Mann-Whitney test (WMW, one-sided: an increase in the median for onset,
        a decrease for withdrawal);
  * the p-values are averaged over half-widths h = 20, 30 and 40 days;
  * WiKS = log10(pKS)/min(log10(pKS)) + log10(pWMW)/min(log10(pWMW)), the minimum taken over
    the days of the series, so each term runs from 0 to 1;
  * onset (withdrawal) is the day at which WiKS is highest with pWMW testing for an increase
    (decrease).

Implementation choices the paper does not specify, all documented and tested in code/50:
  * the search range for onset is 1 May to 15 August, and for withdrawal 15 August to 31
    October and at least 30 days after the onset;
  * the Kolmogorov-Smirnov p-value uses the asymptotic distribution;
  * the Wilcoxon-Mann-Whitney p-value uses the normal approximation with tie and continuity
    corrections (daily rainfall has many tied zeros);
  * a window is used only if at least MIN_VALID of its days are observed in each half.
Input: a 365-element array of daily precipitation (mm) with NaN for unobserved days
(29 February removed), index 0 = 1 January.
"""
import numpy as np
from scipy import stats

H_LIST = (20, 30, 40)
START, END = 120, 304          # 0-based indices of 1 May and 31 October
ONSET_RANGE = (120, 226)       # 1 May to 15 August
WITHDRAW_END = 304
MIN_VALID = 0.8
MIN_GAP = 30
P_FLOOR = 1e-300


def _wmw_z(before, after):
    """Normal-approximation Z for 'after' larger than 'before' (tie and continuity corrected)."""
    n1, n2 = len(after), len(before)
    allv = np.concatenate([after, before])
    ranks = stats.rankdata(allv)
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    n = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    var = n1 * n2 / 12.0 * ((n + 1) - tie / (n * (n - 1)))
    if var <= 0:
        return 0.0
    return (u - n1 * n2 / 2.0 - 0.5 * np.sign(u - n1 * n2 / 2.0)) / np.sqrt(var)


def _agg(vals, how):
    """Combine the p-values of the window half-widths: 'arith' = arithmetic mean (Brunello et al.),
    'geo' = geometric mean (mean of log10 p), tested on synthetic data in code/50c."""
    if how == "geo":
        return 10 ** np.mean(np.log10(np.clip(vals, P_FLOOR, 1.0)))
    return float(np.mean(vals))


def p_series(x, h_list=H_LIST, min_valid=MIN_VALID, ks_method="asymp", min_windows=None, agg="arith"):
    """Daily mean p-values (KS, WMW-increase, WMW-decrease) for t in [START, END].
    A day is used when at least min_windows of the window half-widths pass the validity rule."""
    need = len(h_list) if min_windows is None else min_windows
    days = np.arange(START, END + 1)
    pks = np.full(len(days), np.nan)
    pinc = np.full(len(days), np.nan)
    pdec = np.full(len(days), np.nan)
    for i, t in enumerate(days):
        ks_l, inc_l, dec_l = [], [], []
        for h in h_list:
            b = x[t - h:t]
            a = x[t:t + h]
            b = b[np.isfinite(b)]
            a = a[np.isfinite(a)]
            if len(b) < min_valid * h or len(a) < min_valid * h:
                continue
            ks_l.append(stats.ks_2samp(b, a, method=ks_method).pvalue)
            z = _wmw_z(b, a)
            inc_l.append(stats.norm.sf(z))
            dec_l.append(stats.norm.cdf(z))
        if len(ks_l) >= need:
            pks[i], pinc[i], pdec[i] = _agg(ks_l, agg), _agg(inc_l, agg), _agg(dec_l, agg)
    return days, pks, pinc, pdec


def _wiks(pks, pw):
    a = np.log10(np.clip(pks, P_FLOOR, 1.0))
    b = np.log10(np.clip(pw, P_FLOOR, 1.0))
    with np.errstate(all="ignore"):
        w = a / np.nanmin(a) + b / np.nanmin(b)
    return w


def detect(x, min_days=60, **kw):
    """Return dict with onset and withdrawal day-of-year (1-based) and the WiKS peak values.
    Missing result (NaN) when the series does not support a two-sided change, or when fewer
    than min_days of the 185 test days have a valid window."""
    days, pks, pinc, pdec = p_series(x, **kw)
    out = dict(onset=np.nan, withdrawal=np.nan, wiks_onset=np.nan, wiks_withdrawal=np.nan,
               p_ks_onset=np.nan, p_wmw_onset=np.nan, p_ks_withdrawal=np.nan, p_wmw_withdrawal=np.nan)
    if np.isfinite(pks).sum() < min_days:
        return out
    w_on, w_off = _wiks(pks, pinc), _wiks(pks, pdec)
    on_mask = (days >= ONSET_RANGE[0]) & (days <= ONSET_RANGE[1]) & np.isfinite(w_on)
    if not on_mask.any():
        return out
    k = np.where(on_mask)[0][np.nanargmax(w_on[on_mask])]
    out.update(onset=days[k] + 1, wiks_onset=w_on[k], p_ks_onset=pks[k], p_wmw_onset=pinc[k])
    off_mask = (days >= max(ONSET_RANGE[1], days[k] + MIN_GAP)) & (days <= WITHDRAW_END) & np.isfinite(w_off)
    if off_mask.any():
        j = np.where(off_mask)[0][np.nanargmax(w_off[off_mask])]
        out.update(withdrawal=days[j] + 1, wiks_withdrawal=w_off[j], p_ks_withdrawal=pks[j],
                   p_wmw_withdrawal=pdec[j])
    return out
