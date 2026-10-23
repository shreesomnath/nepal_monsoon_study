"""Step 66: Figure A7, the OLR check (onset replication of Adhikari et al. 2022 and an OLR withdrawal)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, OUT, YEARS  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
yrs = np.array(list(YEARS))


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


cmp_ = pd.read_csv(OUT / "olr_adhikari_compare.csv", index_col=0)
cmp_.index.name = "year"
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")
olr = pd.read_csv(OUT / "olr_dates.csv")
wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
P = wk.pivot(index="station_id", columns="year", values="withdrawal").reindex(columns=yrs)
P = P[P.notna().sum(axis=1) >= 30]
gauge_wd = P.median(axis=0)
ow = olr[olr.series == "east_4cells"].set_index("year")["wd_250_15"].reindex(yrs)
ow_all = olr[olr.series == "all_15cells"].set_index("year")["wd_250_15"].reindex(yrs)

fig, axs_ = plt.subplots(2, 2, figsize=(180 * MM, 128 * MM), layout="constrained")
axs = axs_.ravel()
ax = axs[0]
a = cmp_.dropna(subset=["ad_fall", "ad_mfd"])
ra = np.corrcoef(a.ad_fall, a.ad_mfd)[0, 1]
ax.scatter(a.ad_fall, a.ad_mfd, s=14, color=BLUE)
b = cmp_.dropna(subset=["2cells_25-27.5N_87.5E", "ad_mfd"])
rb = np.corrcoef(b['2cells_25-27.5N_87.5E'], b.ad_mfd)[0, 1]
ax.scatter(b["2cells_25-27.5N_87.5E"], b.ad_mfd, s=14, marker="^", color=VERM)
ax.set_xlabel("OLR fall date (day of year)"); ax.set_ylabel("DHM declared onset (day of year)")
ax.set_title("Onset", loc="left", fontsize=9)
ax.text(0.02, 0.98, f"published 1° OLR (circles): r = {ra:.2f}\nthis study, 2.5° OLR (triangles): r = {rb:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
ax.set_ylim(150, 188); clean(ax)
ax = axs[1]
d = dec.reindex(yrs)
ax.plot(yrs, d.withdrawal_doy, color=VERM, lw=1.2, marker="o", ms=2.2, label="DHM declared")
ax.plot(yrs, gauge_wd, color=BLUE, lw=1.4, label="Gauge composite (WiKS)")
ax.plot(yrs, ow, color=PURPLE, lw=1.4, label="OLR (4-cell mean, ≥250 W m$^{-2}$ for 15 d)")
for s_, c in ((d.withdrawal_doy, VERM), (gauge_wd, BLUE), (ow, PURPLE)):
    z = s_.dropna()
    sl, ic, _, _ = stats.theilslopes(z.to_numpy(float), z.index.to_numpy(float))
    ax.plot(z.index, ic + sl * z.index, "--", color=c, lw=1.0)
ax.set_xlabel("Year"); ax.set_ylabel("Withdrawal (day of year)"); ax.set_title("Withdrawal", loc="left", fontsize=9)
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
clean(ax)
ax = axs[2]
chk = pd.read_csv(OUT / "olr_checks.csv")
w = chk[(chk.event == "withdrawal") & chk.trend_d_per_decade.notna() & (chk.n_years >= 24)].copy()
w["lab"] = w.olr_series.str.replace("east_4cells", "4 cells").str.replace("all_15cells", "15 cells").str.replace("cell_27.5N_87.5E", "1 cell") + " " + w.definition.str.replace("wd_", "").str.replace("_", "/")
w = w.sort_values(["olr_series", "definition"]).reset_index(drop=True)
y = np.arange(len(w))[::-1]
ax.errorbar(w.trend_d_per_decade, y, xerr=[w.trend_d_per_decade - w.ci_lo, w.ci_hi - w.trend_d_per_decade], fmt="o", color=PURPLE, ms=3.5, lw=1.0, capsize=1.8)
ax.axvline(6.2, color=VERM, lw=1.6); ax.axvline(0, color=INK, lw=0.7)
ax.set_yticks(y); ax.set_yticklabels(w.lab, fontsize=5.4)
ax.set_xlabel("Trend (days per decade)"); ax.set_title("Withdrawal definitions (n = %d)" % len(w), loc="left", fontsize=9)
ax.text(6.0, len(w) - 0.3, "DHM\n+6.2", color=VERM, fontsize=6.5, ha="right", va="top")
ax.set_xlim(-10, 10); clean(ax)
ax = axs[3]
sens = pd.read_csv(OUT / "olr_onset_sensitivity.csv")
ax.hist(sens.r_declared, bins=np.arange(-0.4, 0.85, 0.05), color=PURPLE, ec="white", lw=0.4)
ax.axvline(sens.r_declared.median(), color=INK, lw=0.9)
ax.axvline(0.53, color=VERM, lw=1.6)
ax.text(0.55, 0.97, "published\n(1\u00b0 OLR) 0.53", color=VERM, fontsize=6.5, transform=ax.get_xaxis_transform(), va="top", ha="left")
ax.text(sens.r_declared.median() - 0.02, 0.97, f"median\n{sens.r_declared.median():.2f}", fontsize=6.5, transform=ax.get_xaxis_transform(), va="top", ha="right")
ax.set_xlabel("Correlation of OLR onset with declared onset"); ax.set_ylabel("Combinations (n = %d)" % len(sens))
ax.set_title("Onset, 900 choices", loc="left", fontsize=9); clean(ax)
for a_, t in zip(axs, "abcd"):
    a_.text(-0.02, 1.04, t, transform=a_.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"figA7_olr_check.{ext}", **kw)
print("saved figA7_olr_check")
