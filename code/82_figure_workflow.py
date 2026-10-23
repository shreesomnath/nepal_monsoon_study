"""Step 82: methodological diagram (workflow) of the study. Output: figures/figW_workflow.pdf, .png (600 dpi)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK = "#222222"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 7, "text.color": INK, "pdf.fonttype": 42})
W, H = 180, 192
fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def light(c, a=0.16):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def band(y0, y1, color, label):
    ax.add_patch(FancyBboxPatch((0.5, y0), W - 1, y1 - y0, boxstyle="round,pad=0,rounding_size=2", fc=light(color, 0.07), ec="none"))
    ax.text(3.2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=7, fontweight="bold", color=color)


def box(x, y, w, h, text, color, size=6.6, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=light(color, 0.2), ec=color, lw=0.9))
    lines = text.split("\n")
    n = len(lines)
    ys = [y + h / 2 + (i - (n - 1) / 2) * 3.1 for i in range(n)]
    for i, (ln, yy) in enumerate(zip(lines, ys)):
        ax.text(x + w / 2, yy, ln, ha="center", va="center", fontsize=size, fontweight="bold" if (i == 0 and bold_first) else "normal")


def arrow(x0, y0, x1, y1, color=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, ls=ls, shrinkA=0, shrinkB=0, mutation_scale=7))


# ---- bands ----
band(2, 32, SKY, "Data")
band(37, 67, BLUE, "Dates")
band(72, 102, ORANGE, "Checks")
band(107, 127, GREEN, "Composites")
band(132, 162, PURPLE, "Tests")
band(176, 190, VERM, "Answers")
# ---- data ----
box(8, 5, 41, 24, "DHM gauges\n277 stations, daily, 1980 to 2023\nquality-control codes:\nobserved, missing, suspect", SKY)
box(53, 5, 41, 24, "ERA5\nhourly rainfall in each\nstation's 0.25\u00b0 cell\n24 h ending 03 UTC", SKY)
box(98, 5, 41, 24, "NOAA interpolated OLR\n2.5\u00b0 cells, 25 to 30\u00b0N,\n80 to 90\u00b0E\n15 May to 30 November", SKY)
box(143, 5, 35, 24, "DHM declared dates\nonset and withdrawal,\neastern Nepal,\n1968 to 2024", SKY)
# ---- dates ----
box(8, 40, 86, 24, "Objective onset and withdrawal per station-year\nstation-years with at least 85% of days observed (8,835)\nWiKS: Kolmogorov-Smirnov and Wilcoxon tests, windows of 20, 30, 40 days\ntwo-knot fit to cumulative rainfall", BLUE)
box(98, 40, 41, 24, "OLR onset and withdrawal\n7-day mean, thresholds of\n240 to 260 W m$^{-2}$\n18 withdrawal definitions", BLUE)
box(143, 40, 35, 24, "Declared series\ntranscribed from DHM\nbulletins and checked\nagainst three sources", BLUE)
for x in (28.5, 73.5, 118.5, 160.5):
    arrow(x, 29, x, 40)
# ---- checks ----
box(8, 75, 41, 24, "Synthetic seasons\nknown dates; 3 station types,\n3 transition widths,\n0 to 15% missing days", ORANGE)
box(53, 75, 41, 24, "Block-shuffled real seasons\nno transition: false-alarm\nthreshold of the change point", ORANGE)
box(98, 75, 41, 24, "Neighbouring gauges\nagreement within 7 days;\nERA5 against gauges\n(agreement ceiling)", ORANGE)
box(143, 75, 35, 24, "Split-half reliability\nrandom halves of the\nstations, Spearman-Brown\ncorrection", ORANGE)
for x in (28.5, 73.5):
    arrow(x, 64, x, 75, ORANGE)
arrow(118.5, 64, 118.5, 75, ORANGE)
# ---- composites ----
box(8, 110, 170, 14, "Regional composites of a fixed station set: all (213), East province group (109), east of 86\u00b0E (52)\nmedian of station dates per year; station-anomaly composites; ERA5 and OLR composites for comparison", GREEN)
for x in (28.5, 73.5, 118.5, 160.5):
    arrow(x, 99, x, 110, GREEN)
# ---- tests (five boxes) ----
xs = [8, 42.75, 77.5, 112.25, 147]
labels = ["Trend and step\nTheil-Sen slopes,\nblock bootstrap,\nPettitt test",
          "Declared minus\ncomposite\ndifference-series\nslope and step",
          "Power of the test\nsynthetic networks,\nreal composite noise,\n2 to 6.2 d per decade",
          "Robustness\nstricter list,\nleave-one-province-out,\nelevation, 76 homogeneous",
          "Recent years\n2018 to 2023 anomalies,\nfixed 98 stations,\nyear-based interval"]
for x, lab in zip(xs, labels):
    box(x, 135, 31, 24, lab, PURPLE, size=6.2)
    arrow(x + 15.5, 124, x + 15.5, 135, PURPLE)
# ---- collector and answers ----
ax.plot([xs[0] + 15.5, xs[-1] + 15.5], [166, 166], color=VERM, lw=0.9)
for x in xs:
    ax.plot([x + 15.5, x + 15.5], [159, 166], color=VERM, lw=0.9)
qs = [(8, "Does rainfall show the declared\nlengthening, 1981 to 2017?"), (66, "Are the 2018 to 2023\nwithdrawals late?"), (124, "Seasonal budgets:\nfrequency and intensity")]
for x, q in qs:
    box(x, 179, 54, 9, q, VERM, size=6.4, bold_first=False)
    arrow(x + 27, 166, x + 27, 179, VERM)
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"figW_workflow.{ext}", **kw)
print("saved figW_workflow")
