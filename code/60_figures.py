"""Step 60: figures for the monsoon-timing study (600 dpi PNG and vector PDF, 180 mm wide).

Every plotted number comes from a CSV in monsoon_timing/outputs; figure-source tables are written
next to each figure (figures/*_source.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.ops import unary_union
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, INP, OUT, ROOT, YEARS  # noqa: E402

FIG.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.edgecolor": INK,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
yrs = np.array(list(YEARS))
TYPE_LAB = {"humid": "Humid", "dry_west": "Dry west", "high_mountain": "High mountain"}
TYPE_COL = {"humid": BLUE, "dry_west": ORANGE, "high_mountain": VERM}


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


def label(ax, t):
    ax.text(-0.02, 1.04, t, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    plt.close(fig)


sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
dates = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str}).merge(
    sp[["station_id", "region", "lon", "lat", "elev"]], on="station_id")
dec = pd.read_csv(OUT / "dhm_declared_dates.csv")

# ---------------------------------------------------------------- Figure A1: method validation
mr = pd.read_csv(OUT / "wiks_missing_rule.csv")
mr = mr[(mr.min_valid == 0.5) & (mr.min_windows == 2)]
pw = pd.read_csv(OUT / "wiks_qc_power_synthetic.csv")
fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), layout="constrained")
src = []
for k, (ev, col) in enumerate((("Onset", "err_onset"), ("Withdrawal", "err_withdrawal"))):
    ax = axs[k]
    for t in TYPE_LAB:
        for miss, ls in ((0.0, "-"), (0.05, "--"), (0.15, ":")):
            g = mr[(mr.type == t) & (mr.missing == miss)].groupby("width")[col].apply(lambda s: (s.abs() <= 7).mean())
            ax.plot(g.index, 100 * g.values, ls, color=TYPE_COL[t], marker="o", ms=3.5, lw=1.3,
                    label=f"{TYPE_LAB[t]}" if miss == 0 else None)
            src += [(ev, t, miss, w, v) for w, v in g.items()]
    ax.set_xlabel("Transition width (days)"); ax.set_xticks([0, 10, 25]); ax.set_ylim(0, 100)
    ax.set_title(ev, loc="left", fontsize=9); clean(ax)
    if k == 0:
        ax.set_ylabel("Seasons within 7 days of truth (%)")
        ax.legend(frameon=False, loc="lower left")
        ax.text(0.98, 0.98, "solid 0%, dashed 5%,\ndotted 15% missing days", transform=ax.transAxes, ha="right", va="top", fontsize=7)
ax = axs[2]
for t in TYPE_LAB:
    g = pw[pw.type == t].groupby("width").pass_onset.mean()
    ax.plot(g.index, 100 * g.values, "-", color=TYPE_COL[t], marker="o", ms=3.5, lw=1.3, label=TYPE_LAB[t])
ax.set_xlabel("Transition width (days)"); ax.set_xticks([0, 10, 25]); ax.set_ylim(0, 100)
ax.set_ylabel("True onsets passing the null test (%)"); ax.set_title("Power of the 5% null filter", loc="left", fontsize=9); clean(ax)
for a, t in zip(axs, "abc"):
    label(a, t)
pd.DataFrame(src, columns=["event", "type", "missing", "width", "within7_pct"]).to_csv(FIG / "figA1_source.csv", index=False)
save(fig, "figA1_method_validation")

# ---------------------------------------------------------------- Figure A2: station climatology maps
summ = pd.read_csv(OUT / "wiks_station_summary.csv", dtype={"station_id": str})
summ = summ[summ.n_years >= 15]
r = shapefile.Reader(str(ROOT / "gis" / "Local Unit" / "local_unit"))
prov = {}
for rec, s in zip(r.records(), r.shapes()):
    prov.setdefault(rec["Province"], []).append(shape(s.__geo_interface__))
geom = {k: unary_union([g.buffer(0) for g in v]).simplify(0.01, preserve_topology=True) for k, v in prov.items()}
fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 52 * MM), layout="constrained")
panels = (("onset_median", "Onset (day of year)", "viridis_r", (150, 185)),
          ("withdrawal_median", "Withdrawal (day of year)", "viridis", (245, 285)),
          ("length_median", "Season length (days)", "magma_r", (75, 125)))
for ax, (col, ttl, cmap, (lo, hi)) in zip(axs, panels):
    for g in geom.values():
        for poly in (g.geoms if hasattr(g, "geoms") else [g]):
            x, y = poly.exterior.xy
            ax.fill(x, y, fc="#F2F2F2", ec=GREY, lw=0.3)
    sc = ax.scatter(summ.lon, summ.lat, c=summ[col], s=7, cmap=cmap, vmin=lo, vmax=hi, lw=0.2, ec="white")
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
    cb.ax.tick_params(labelsize=7)
    ax.set_title(ttl, loc="left", fontsize=8.5)
    ax.set_aspect(1 / np.cos(np.radians(28))); ax.set_xlim(80, 88.3); ax.set_ylim(26.3, 30.6)
    ax.set_xticks([81, 83, 85, 87]); ax.set_yticks([27, 28, 29, 30])
for a, t in zip(axs, "abc"):
    label(a, t)
summ[["station_id", "lon", "lat", "elev", "region", "onset_median", "withdrawal_median", "length_median", "n_years"]].to_csv(
    FIG / "figA2_source.csv", index=False)
save(fig, "figA2_climatology_maps")

# ---------------------------------------------------------------- Figure A3: composites vs DHM
fig, axs = plt.subplots(2, 2, figsize=(180 * MM, 100 * MM), layout="constrained", sharex=True)
rng = np.random.default_rng(3)
src = []
for r_i, ev in enumerate(("onset", "withdrawal")):
    for c_i, (reg, sub) in enumerate((("All stations", dates), ("East of 86°E", dates[dates.lon >= 86.0]))):
        ax = axs[r_i, c_i]
        P = sub.pivot(index="station_id", columns="year", values=ev).reindex(columns=yrs)
        P = P[P.notna().sum(axis=1) >= 30]
        comp = P.median(axis=0)
        boots = np.array([P.iloc[rng.integers(len(P), size=len(P))].median(axis=0).to_numpy() for _ in range(300)])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        ax.fill_between(yrs, lo, hi, color=SKY, alpha=0.35, lw=0)
        ax.plot(yrs, comp, color=BLUE, lw=1.4, label=f"Gauge composite (n = {len(P)})")
        d = dec[(dec.year >= 1981) & (dec.year <= 2017)]
        dcol = "onset_doy" if ev == "onset" else "withdrawal_doy"
        ax.plot(d.year, d[dcol], color=VERM, lw=1.2, marker="o", ms=2.5, label="DHM declared")
        for y_, col, c in ((comp, "g", BLUE), (d.set_index("year")[dcol], "d", VERM)):
            z = y_.dropna()
            sl, ic, _, _ = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
            ax.plot(z.index, ic + sl * z.index, "--", color=c, lw=1.0)
            src.append((ev, reg, col, sl * 10))
        ax.set_title(f"{ev.capitalize()}, {reg}", loc="left", fontsize=8.5); clean(ax)
        if c_i == 0:
            ax.set_ylabel("Day of year")
for a, t in zip(axs.ravel(), "abcd"):
    label(a, t)
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend([h[0], h[1]], ["Gauge composite (median of stations, shading: 95% bootstrap over stations)",
                          "DHM declared (eastern Nepal); dashed lines: Sen trends"],
           loc="outside lower center", ncol=2, frameon=False, fontsize=7)
axs[1, 0].set_xlabel("Year"); axs[1, 1].set_xlabel("Year")
pd.DataFrame(src, columns=["event", "set", "series(g=gauge,d=declared)", "sen_slope_d_per_decade"]).to_csv(FIG / "figA3_source.csv", index=False)
save(fig, "figA3_gauge_vs_declared")

# ---------------------------------------------------------------- Figure A4: agreement and noise ceiling
ceil = pd.read_csv(OUT / "robust_noise_ceiling.csv")
cmp_ = pd.read_csv(OUT / "wiks_era5_compare.csv")
cmp_ = cmp_[cmp_.region == "All"]
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 60 * MM), layout="constrained")
for ax, ev in zip(axs, ("onset", "withdrawal")):
    c = ceil[ceil.event == ev]
    labs = [f"gauge pairs\n{d} km" for d in c.distance_km] + ["ERA5\n≥1 mm", "ERA5\nraw", "ERA5\n3×3"]
    vals = list(100 * c.within7) + [100 * cmp_[(cmp_.event == ev) & (cmp_.version == v)].within7.iloc[0] for v in ("ge1mm", "raw", "mean3x3")]
    cols = [BLUE] * len(c) + [ORANGE] * 3
    ax.bar(range(len(vals)), vals, color=cols, ec="white", lw=0.5, width=0.7)
    for i_, v_ in enumerate(vals):
        ax.text(i_, v_ + 1, f"{v_:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=6.5)
    ax.set_ylim(0, 70); ax.set_title(ev.capitalize(), loc="left", fontsize=9); clean(ax)
axs[0].set_ylabel("Dates within 7 days\nof the gauge date (%)")
label(axs[0], "a"); label(axs[1], "b")
save(fig, "figA4_agreement_ceiling")

# ---------------------------------------------------------------- Figure A5: lengthening test
ci = pd.read_csv(OUT / "composite_slopes_block_bootstrap.csv")
ci = ci[ci.event == "withdrawal"]
lt = pd.read_csv(OUT / "robust_late_season_trends.csv")
fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 62 * MM), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
ax = axs[0]
names = ["All", "East provinces", "East of 86E"]
ypos = np.arange(len(names))[::-1] + 1
for off, (dname, colr, lab_) in zip((0.15, -0.15), (("WiKS", BLUE, "WiKS"), ("two-knot", ORANGE, "Two-knot"))):
    for y_, nm in zip(ypos, names):
        r_ = ci[(ci.definition == dname) & (ci.set == nm)].iloc[0]
        ax.errorbar(r_.slope, y_ + off, xerr=[[r_.slope - r_.boot_lo], [r_.boot_hi - r_.slope]], fmt="o", color=colr, ms=4.5, capsize=2.2, lw=1.2,
                    label=lab_ if nm == "All" else None)
ax.axvline(6.2, color=VERM, lw=1.6)
ax.text(6.0, len(names) + 0.6, "DHM declared\n+6.2 d per decade", color=VERM, fontsize=7.5, va="top", ha="right")
ax.legend(loc="upper left", frameon=False, fontsize=7)
w = pd.DataFrame({"set": ["All stations", "East provinces", "East of 86\u00b0E"]})
ax.axvline(0, color=INK, lw=0.7)
ax.set_yticks(ypos); ax.set_yticklabels(list(w.set)); ax.set_ylim(0.4, len(w) + 0.9); ax.set_xlim(-8, 7)
ax.set_xlabel("Withdrawal trend 1981–2017 (days per decade)\ngauge composites, 95% block-bootstrap interval"); clean(ax, "x")
label(ax, "a")
ax = axs[1]
sub = lt[(lt.variable == "late_share") & (lt.region.isin(["East", "West", "Central", "All"]))]
regs = ["West", "Central", "East", "All"]
for k, (srcn, col) in enumerate((("gauge", BLUE), ("era5", ORANGE))):
    vals = [sub[(sub.source == srcn) & (sub.region == r_)].composite_slope_per_decade.iloc[0] for r_ in regs]
    ax.bar(np.arange(4) + (k - 0.5) * 0.36, vals, 0.34, color=col, ec="white", label="Gauges" if srcn == "gauge" else "ERA5")
ax.axhline(0, color=INK, lw=0.7)
ax.set_xticks(range(4)); ax.set_xticklabels(regs)
ax.set_ylabel("Share of Jun–Oct rain after 20 Sep\n(% points per decade)"); ax.legend(frameon=False); clean(ax)
label(ax, "b")
save(fig, "figA5_lengthening_test")

# ---------------------------------------------------------------- Figure A6: seasonal budget trends
bt = pd.read_csv(OUT / "trend_budget_all.csv")
bt = bt[bt.region == "All"]
seasons = [("WiKS_pre", "Pre\nWiKS"), ("cal_MAM", "MAM"), ("WiKS_mon", "Mon\nWiKS"),
           ("cal_JJAS", "JJAS"), ("WiKS_post", "Post\nWiKS"), ("cal_ON", "Oct")]   # cal_ON in the CSV is 1-31 October
fig, axs = plt.subplots(1, 3, figsize=(180 * MM, 62 * MM), layout="constrained")
for ax, (var, ttl) in zip(axs, (("total", "Seasonal total"), ("freq", "Rain-day frequency"), ("intensity", "Intensity"))):
    vals, sig = [], []
    for s, _ in seasons:
        r_ = bt[(bt.season == s) & (bt.variable == var)]
        vals.append(r_.median_pct_per_decade.iloc[0] if len(r_) else np.nan)
        sig.append((len(r_) and (min(r_.field_p_neg.iloc[0], r_.field_p_pos.iloc[0]) < 0.05)))
    cols = [BLUE if "WiKS" in s else GREY for s, _ in seasons]
    ax.bar(range(len(seasons)), vals, color=cols, ec="white", width=0.7)
    for i_, (v_, s_) in enumerate(zip(vals, sig)):
        if s_ and np.isfinite(v_):
            ax.text(i_, v_ + (0.15 if v_ >= 0 else -0.15), "*", ha="center", va="bottom" if v_ >= 0 else "top", fontsize=11)
    ax.axhline(0, color=INK, lw=0.7)
    ax.margins(y=0.18)
    ax.set_xticks(range(len(seasons))); ax.set_xticklabels([l for _, l in seasons], fontsize=7)
    ax.set_title(ttl, loc="left", fontsize=9); clean(ax)
axs[0].set_ylabel("Median station trend (% per decade)")
fig.supxlabel("blue: WiKS-based seasons; grey: calendar seasons; * field significant (p < 0.05, whole-year block permutation); no bar: median station trend of zero", fontsize=6.5)
for a, t in zip(axs, "abc"):
    label(a, t)
bt.to_csv(FIG / "figA6_source.csv", index=False)
save(fig, "figA6_seasonal_budget_trends")
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
