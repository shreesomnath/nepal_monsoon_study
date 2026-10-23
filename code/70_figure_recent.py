"""Step 70: Figure A8, the withdrawal anomaly 1981-2023 in the declared series, gauges, ERA5 and OLR, with decade means."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, OUT  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
Y = pd.read_csv(OUT / "recent_years_withdrawal_by_year.csv", index_col=0)
Y.index.name = "year"
olr = pd.read_csv(OUT / "recent_years_olr_withdrawal_anomaly.csv", index_col=0).iloc[:, 0]
R = pd.read_csv(OUT / "recent_years_summary.csv")
gcol = [c for c in Y.columns if c.startswith("Gauge WiKS | fixed")][0]
ecol = [c for c in Y.columns if c.startswith("ERA5 WiKS | fixed")][0]
dcol = [c for c in Y.columns if c.startswith("DHM")][0]

fig, axs = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), layout="constrained", gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
ax.axvspan(2017.5, 2023.5, color=LIGHT, alpha=0.5, lw=0)
ax.axhline(0, color=INK, lw=0.6)
ax.plot(Y.index, Y[dcol], color=VERM, lw=1.3, marker="o", ms=2.5, label="DHM declared")
ax.plot(Y.index, Y[gcol], color=BLUE, lw=1.4, label="Gauges (98 fixed stations)")
ax.plot(Y.index, Y[ecol], color=ORANGE, lw=1.1, ls="--", label="ERA5 (same station-years)")
ax.plot(olr.index, olr.values, color=PURPLE, lw=1.1, label="OLR (4 cells; to 2022)")
ax.set_xlabel("Year"); ax.set_ylabel("Withdrawal anomaly (days)\nrelative to each series' 1981–2017 mean")
ax.set_title("Withdrawal, anomaly by year", loc="left", fontsize=9)
ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=6.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=LIGHT, lw=0.5)
ax.set_ylim(-30, 42)
ax2 = axs[1]
per = ["1981-1990", "1991-2000", "2001-2010", "2011-2017", "2018-2023"]
rows = {"DHM declared": ("DHM declared", VERM), "Gauge WiKS, fixed (>=30 before, >=4 after)": ("Gauges", BLUE),
        "ERA5 WiKS, fixed (>=30 before, >=4 after)": ("ERA5", ORANGE)}
w = 0.26
from scipy import stats  # noqa: E402
spans = [(1981, 1990), (1991, 2000), (2001, 2010), (2011, 2017), (2018, 2023)]
ycol = {"DHM declared": dcol, "Gauge WiKS, fixed (>=30 before, >=4 after)": gcol, "ERA5 WiKS, fixed (>=30 before, >=4 after)": ecol}
for k, (key, (lab, col)) in enumerate(rows.items()):
    r = R[(R.series == key) & (R.event == "withdrawal")].iloc[0]
    means, errs = [], []
    for (a_, b_), p in zip(spans, per):
        v = Y.loc[a_:b_, ycol[key]].dropna()
        assert abs(v.mean() - r[p]) < 0.05, (key, p, v.mean(), r[p])
        means.append(v.mean())
        errs.append(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))     # year-based 95% interval of the mean
    ax2.bar(np.arange(5) + (k - 1) * w, means, w, color=col, ec="white", label=lab)
    ax2.errorbar(np.arange(5) + (k - 1) * w, means, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
ax2.axhline(0, color=INK, lw=0.6)
ax2.set_xticks(range(5)); ax2.set_xticklabels(["81–90", "91–00", "01–10", "11–17", "18–23"])
ax2.set_ylabel("Decade mean anomaly (days)\nbars: year-based 95% interval"); ax2.set_title("Decade means", loc="left", fontsize=9)
ax2.set_ylim(-17, 31)
ax2.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=3, columnspacing=0.8, handlelength=1.2)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.grid(axis="y", color=LIGHT, lw=0.5)
for a_, t in zip(axs, "ab"):
    a_.text(-0.02, 1.04, t, transform=a_.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"figA8_recent_years.{ext}", **kw)
print("saved figA8")
