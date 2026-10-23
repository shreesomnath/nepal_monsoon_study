"""Step 78: station network figure (province groups and coverage by year).

a  Map of the 268 stations with at least 15 analysed years in 1981-2017, by province group, with the 213 stations of the
   composite set marked and the 86 E line.
b  Stations with an analysed station-year and, of the 213 composite stations, those with a withdrawal date, per year, 1981-2023.
c  Station elevation with the bands used in the robustness test.
Outputs: figures/fig1_network.pdf, .png (600 dpi), fig1_source.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, INP, OUT, ROOT  # noqa: E402

BLUE, ORANGE, GREEN, PURPLE, VERM, GREY, SKY = "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#8C8C8C", "#56B4E9"
INK, LIGHT = "#222222", "#D9D9D9"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.linewidth": 0.6, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42})
COL = {"East": BLUE, "Central": ORANGE, "West": GREEN}


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
sp["region"] = np.where(sp.province.isin(["Sudur Pashchim", "Karnali", "Lumbini"]), "West",
                        np.where(sp.province == "Gandaki", "Central", "East"))
d1 = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
d2 = pd.read_csv(OUT / "wiks_station_dates_2018_2023.csv", dtype={"station_id": str})
alld = pd.concat([d1[["station_id", "year", "withdrawal"]], d2[["station_id", "year", "withdrawal"]]])
n_years = d1.groupby("station_id").size()
net = sp[sp.station_id.isin(n_years[n_years >= 15].index)].copy()
P = d1.pivot(index="station_id", columns="year", values="withdrawal").reindex(columns=range(1981, 2018))
comp_ids = set(P.index[P.notna().sum(axis=1) >= 30])
net["composite"] = net.station_id.isin(comp_ids)

r = shapefile.Reader(str(ROOT / "gis" / "Local Unit" / "local_unit"))
prov = {}
for rec, s in zip(r.records(), r.shapes()):
    prov.setdefault(rec["Province"], []).append(shape(s.__geo_interface__))
geom = {k: unary_union([g.buffer(0) for g in v]).simplify(0.01, preserve_topology=True) for k, v in prov.items()}

fig = plt.figure(figsize=(180 * MM, 70 * MM), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.2, 0.9])
ax = fig.add_subplot(gs[0])
for g in geom.values():
    for poly in (g.geoms if hasattr(g, "geoms") else [g]):
        x, y = poly.exterior.xy
        ax.fill(x, y, fc="#F2F2F2", ec=GREY, lw=0.3)
for reg, c in COL.items():
    s_ = net[(net.region == reg) & ~net.composite]
    ax.scatter(s_.lon, s_.lat, s=8, facecolors="none", edgecolors=c, lw=0.6)
    s_ = net[(net.region == reg) & net.composite]
    ax.scatter(s_.lon, s_.lat, s=9, color=c, lw=0.2, ec="white", label=f"{reg} ({len(s_)})")
ax.axvline(86.0, color=INK, lw=0.7, ls="--")
ax.text(86.1, 30.45, "86°E", fontsize=6.5, va="top")
ax.set_aspect(1 / np.cos(np.radians(28))); ax.set_xlim(80, 88.3); ax.set_ylim(26.3, 30.6)
ax.set_xticks([81, 83, 85, 87]); ax.set_yticks([27, 28, 29, 30])
ax.legend(loc="lower left", frameon=False, title="Composite stations", title_fontsize=6.5, fontsize=6.5, handletextpad=0.1,
          borderpad=0.1, labelspacing=0.2, markerscale=1.1)
ax.set_title("Network (open: not in composite set)", loc="left", fontsize=8.5)

ax = fig.add_subplot(gs[1])
yrs = np.arange(1981, 2024)
n_all = alld.groupby("year").station_id.nunique().reindex(yrs)
n_comp = alld[alld.station_id.isin(comp_ids) & alld.withdrawal.notna()].groupby("year").station_id.nunique().reindex(yrs)
ax.bar(yrs, n_all, color=SKY, width=0.8, label="Stations with an analysed year")
ax.plot(yrs, n_comp, color=BLUE, lw=1.4, label="Composite stations with a date")
ax.axvspan(2017.5, 2023.5, color=LIGHT, alpha=0.5, lw=0)
ax.set_xlabel("Year"); ax.set_ylabel("Stations"); ax.set_title("Coverage by year", loc="left", fontsize=8.5)
ax.set_ylim(0, 340)
ax.legend(loc="upper center", frameon=False, fontsize=6.5); clean(ax)

ax = fig.add_subplot(gs[2])
bands = [(0, 1000), (1000, 2000), (2000, 9000)]
ax.hist(net.elev.clip(upper=4500), bins=np.arange(0, 4600, 250), color=GREY, ec="white", lw=0.4)
for x in (1000, 2000):
    ax.axvline(x, color=VERM, lw=0.9, ls="--")
for (lo, hi), name in zip(bands, ("below 1000 m", "1000 to 2000 m", "2000 m and above")):
    pass
ax.set_xlabel("Elevation (m)"); ax.set_ylabel("Stations"); ax.set_title("Elevation bands", loc="left", fontsize=8.5); clean(ax)
for a_, t in zip(fig.axes, "abc"):
    a_.text(-0.02, 1.04, t, transform=a_.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"fig1_network.{ext}", **kw)
pd.DataFrame({"year": yrs, "stations_with_analysed_year": n_all.values, "composite_stations_with_date": n_comp.values}).to_csv(
    FIG / "fig1_source.csv", index=False)
print(len(net), "stations;", int(net.composite.sum()), "composite")
print(pd.DataFrame({"all": n_all, "comp": n_comp}).T.to_string())
