"""Step 55: trends in monsoon timing, seasonal rainfall budgets, and the pre-monsoon test.

Inputs: outputs/wiks_station_dates.csv (accepted dates only, after the null-based QC in code/53,
column 'ok_onset', 'ok_withdrawal' written by code/56 if present), the observed daily series, and
DHM-declared dates.
Part A  Station dates -> regional composites (median over stations present, at least 60% of the
        regional stations) -> Sen slope and Mann-Kendall, with a station-level field-significance
        test (block permutation of whole years). Compared with DHM-declared dates over the same years.
Part B  Seasonal totals with WiKS-based bounds (pre-monsoon: 1 March to onset, monsoon: onset to
        withdrawal, post-monsoon: withdrawal to 30 November) and with calendar bounds (MAM, JJAS, ON),
        using observed days only and requiring 90% of the days of the season to be observed (no
        infilling). Trends per station and region; fixed-network (stations with at least 30 of 37
        seasons) versus all-station composites; frequency-intensity decomposition.
Outputs: outputs/trend_timing.csv, outputs/trend_budget.csv, outputs/declared_vs_gauge.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT, YEARS  # noqa: E402

rng = np.random.default_rng(55)


def sen_mk(y, t):
    ok = np.isfinite(y)
    y, t = y[ok], t[ok]
    n = len(y)
    if n < 15:
        return np.nan, np.nan, n
    s = np.sum(np.sign(y[None, :] - y[:, None])[np.triu_indices(n, 1)])
    var = n * (n - 1) * (2 * n + 5) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return stats.theilslopes(y, t)[0] * 10, 2 * (1 - stats.norm.cdf(abs(z))), n


def field_test(M, t, nperm=1000, block=3):
    """M: stations x years. Count of locally significant (p<0.05) negative and positive trends and
    the permutation (whole-year block) p-values for these counts."""
    def counts(A):
        neg = pos = 0
        for r in A:
            sl, p, n = sen_mk(r, t)
            if np.isfinite(p) and p < 0.05:
                neg += sl < 0
                pos += sl > 0
        return neg, pos
    o_neg, o_pos = counts(M)
    T = M.shape[1]
    blocks = [np.arange(s, min(s + block, T)) for s in range(0, T, block)]
    nn, npos = [], []
    for _ in range(nperm):
        order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
        a, b = counts(M[:, order])
        nn.append(a)
        npos.append(b)
    return o_neg, o_pos, (np.array(nn) >= o_neg).mean(), (np.array(npos) >= o_pos).mean()


TIER = sys.argv[1] if len(sys.argv) > 1 else "strict"   # 'strict': dates passing the null threshold; 'all': every detected date
SUF = "" if TIER == "strict" else "_all"

if __name__ == "__main__":
    dates = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
    if TIER == "all":
        dates["ok_onset"] = dates.onset.notna()
        dates["ok_withdrawal"] = dates.withdrawal.notna()
    print("TIER:", TIER, "| accepted onset %.1f%%, withdrawal %.1f%%" % (100 * dates.ok_onset.mean(), 100 * dates.ok_withdrawal.mean()))
    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                            np.where(sp.province == "Gandaki", "Central", "East"))
    dates = dates.merge(sp[["station_id", "region", "lon", "elev"]], on="station_id")
    yrs = np.array(list(YEARS))

    # ---------------- Part A: timing trends ----------------
    rows = []
    comp = {}
    for ev in ("onset", "withdrawal", "length"):
        col = ev
        if ev == "length":
            dates["length"] = np.where(dates.ok_onset & dates.ok_withdrawal, dates.withdrawal - dates.onset, np.nan)
        else:
            dates[ev] = np.where(dates[f"ok_{ev}"], dates[ev], np.nan)
        for reg, sub in list(dates.groupby("region")) + [("All", dates)]:
            P = sub.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
            frac = P.notna().mean(axis=0)
            med = P.median(axis=0).where(frac >= 0.6)
            sl, p, n = sen_mk(med.to_numpy(float), yrs.astype(float))
            comp[(ev, reg)] = med
            # station-level trends and field significance (stations with >= 25 dates)
            keep = P[P.notna().sum(axis=1) >= 25]
            sts = [sen_mk(r.to_numpy(float), yrs.astype(float)) for _, r in keep.iterrows()]
            o_neg, o_pos, pn, pp = field_test(keep.to_numpy(float), yrs.astype(float), nperm=500) if len(keep) >= 10 else (np.nan,) * 4
            rows.append(dict(event=ev, region=reg, composite_slope_d_per_decade=sl, composite_p=p, n_years=n,
                             n_stations=len(keep), station_median_slope=np.nanmedian([s[0] for s in sts]) if sts else np.nan,
                             local_sig_neg=o_neg, local_sig_pos=o_pos, field_p_neg=pn, field_p_pos=pp))
    tt = pd.DataFrame(rows)
    tt.to_csv(OUT / f"trend_timing{SUF}.csv", index=False)
    print("TIMING TRENDS 1981-2017 (days per decade):")
    print(tt.round(3).to_string(index=False))

    # ---------------- declared vs gauge ----------------
    dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")
    rows = []
    for ev, dcol in (("onset", "onset_doy"), ("withdrawal", "withdrawal_doy"), ("length", "length_d")):
        for reg in ("East", "All"):
            med = comp[(ev, reg)].dropna()
            d = dec.reindex(med.index)[dcol].dropna()
            both = pd.concat([med, d], axis=1, keys=["gauge", "declared"]).dropna()
            r, pr = stats.pearsonr(both.gauge, both.declared) if len(both) > 5 else (np.nan, np.nan)
            rs, ps = stats.spearmanr(both.gauge, both.declared) if len(both) > 5 else (np.nan, np.nan)
            sg = stats.theilslopes(both.gauge, both.index.to_numpy(float))[0] * 10
            sd = stats.theilslopes(both.declared, both.index.to_numpy(float))[0] * 10
            rows.append(dict(event=ev, gauge_region=reg, n=len(both), pearson_r=r, p=pr, spearman=rs,
                             mean_diff_d=(both.gauge - both.declared).mean(), slope_gauge=sg, slope_declared=sd))
    dv = pd.DataFrame(rows)
    dv.to_csv(OUT / f"declared_vs_gauge{SUF}.csv", index=False)
    print("\nGAUGE COMPOSITE VS DHM-DECLARED (1981-2017):")
    print(dv.round(3).to_string(index=False))

    # ---------------- Part B: seasonal budgets ----------------
    d = np.load(INP / "daily_qc.npz", allow_pickle=True)
    X, status = d["X"], d["status"]
    sid = list(d["station_id"])
    yr = pd.to_datetime(d["dates"]).year.to_numpy()
    V = np.where(status == 0, X, np.nan)

    def seasons(v, on, wd):
        """Return totals, wet-day counts and valid-day shares for WiKS and calendar seasons (0-based idx)."""
        bounds = {"WiKS_pre": (59, on - 1), "WiKS_mon": (on - 1, wd - 1), "WiKS_post": (wd - 1, 334),
                  "cal_MAM": (59, 151), "cal_JJAS": (151, 273), "cal_ON": (273, 304)}
        res = {}
        for k, (a, b) in bounds.items():
            seg = v[int(a):int(b)]
            if len(seg) < 10:
                continue
            valid = np.isfinite(seg)
            if valid.mean() < 0.9:
                continue
            wet = seg[valid] >= 1.0
            res[k] = dict(total=np.nansum(seg), n=len(seg), wetdays=wet.sum(), valid=valid.sum(),
                          intensity=seg[valid][wet].mean() if wet.any() else np.nan)
        return res
    recs = []
    for _, r in dates.iterrows():
        if not (r.ok_onset and r.ok_withdrawal):
            continue
        i = sid.index(r.station_id)
        v = V[i, yr == int(r.year)]
        if v.size == 366:
            v = np.delete(v, 59)
        for k, s in seasons(v, int(r.onset), int(r.withdrawal)).items():
            recs.append(dict(station_id=r.station_id, year=int(r.year), season=k, region=r.region, **s))
    B = pd.DataFrame(recs)
    B["freq"] = B.wetdays / B.valid
    B.to_csv(OUT / f"budget_station_seasons{SUF}.csv", index=False)
    rows = []
    for (season, reg), sub in list(B.groupby(["season", "region"])) + [((s, "All"), q) for s, q in B.groupby("season")]:
        for var in ("total", "freq", "intensity", "n"):
            P = sub.pivot(index="station_id", columns="year", values=var).reindex(columns=yrs)
            keep = P[P.notna().sum(axis=1) >= 25]
            if len(keep) < 5:
                continue
            sts = [sen_mk(rr.to_numpy(float), yrs.astype(float)) for _, rr in keep.iterrows()]
            slopes = np.array([s[0] for s in sts])
            medv = keep.median(axis=1).to_numpy()
            pct = 100 * slopes / np.where(medv > 0, medv, np.nan)
            o_neg, o_pos, pn, pp = field_test(keep.to_numpy(float), yrs.astype(float), nperm=300)
            rows.append(dict(season=season, region=reg, variable=var, n_stations=len(keep),
                             median_slope_per_decade=np.nanmedian(slopes), median_pct_per_decade=np.nanmedian(pct),
                             local_sig_neg=o_neg, local_sig_pos=o_pos, field_p_neg=pn, field_p_pos=pp))
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / f"trend_budget{SUF}.csv", index=False)
    print("\nSEASONAL BUDGET TRENDS (per decade; totals in mm, freq as fraction, intensity mm/wet day, n = season length d):")
    print(bt[bt.region == "All"].round(3).to_string(index=False))
    print(bt[(bt.season.isin(["WiKS_pre", "cal_MAM"])) & (bt.region != "All")].round(3).to_string(index=False))
