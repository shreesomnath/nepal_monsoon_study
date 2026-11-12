"""Step 85: animation of the daily rainfall climatology of Nepal through the year (1981-2017 mean of the gauges),
in the style of the Himalaya rainfall climatology video of Brunello et al. (2020, supporting information).

Map: Gaussian-kernel smoothing of the station climatology (bandwidth 0.30 degrees), masked to Nepal, with the stations
drawn as dots. Top strip: mean daily rainfall of the stations through the year with the seasons delimited by the
median objective onset and withdrawal (WiKS, station medians) and the mean DHM-declared dates for 1981-2017.
Usage: python 85_animation_climatology.py [preview]
Outputs: figures/animation/nepal_rainfall_climatology.mp4 and .gif (or preview PNGs)
"""
import sys
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.colors import Normalize
from matplotlib.path import Path as MPath
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, INP, OUT, ROOT  # noqa: E402

PREVIEW = len(sys.argv) > 1 and sys.argv[1] == "preview"
ANIM = FIG / "animation"
ANIM.mkdir(exist_ok=True)
INK, LIGHT = "#222222", "#D9D9D9"
CMAP = plt.get_cmap("YlGnBu")
VMAX = 25.0                      # mm per day
H = 0.30                         # kernel bandwidth, degrees
SMOOTH = 11                      # days, circular running mean of the climatology

# ---------------- climatology of the stations ----------------
d = np.load(INP / "daily_qc.npz", allow_pickle=True)
X, status = d["X"].astype(float), d["status"]
ids = [str(s) for s in d["station_id"]]
dt = pd.to_datetime(d["dates"])
V = np.where(status == 0, X, np.nan)
keep = (dt.year >= 1981) & (dt.year <= 2017)
dt, V = dt[keep], V[:, keep]
leap_feb29 = (dt.month == 2) & (dt.day == 29)
dt, V = dt[~leap_feb29], V[:, ~leap_feb29]
doy = np.array([(pd.Timestamp(year=2001, month=m, day=dd).dayofyear) for m, dd in zip(dt.month, dt.day)])     # 1..365 on a non-leap calendar
clim = np.full((len(ids), 365), np.nan)
for k in range(365):
    col = V[:, doy == k + 1]
    n = np.isfinite(col).sum(axis=1)
    m = np.nanmean(np.where(np.isfinite(col), col, np.nan), axis=1)
    clim[:, k] = np.where(n >= 15, m, np.nan)
# circular running mean over SMOOTH days
pad = SMOOTH // 2
ext = np.concatenate([clim[:, -pad:], clim, clim[:, :pad]], axis=1)
sm = np.full_like(clim, np.nan)
for k in range(365):
    w = ext[:, k:k + SMOOTH]
    sm[:, k] = np.nanmean(w, axis=1)

sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
lon = np.array([sp.loc[s, "lon"] if s in sp.index else np.nan for s in ids])
lat = np.array([sp.loc[s, "lat"] if s in sp.index else np.nan for s in ids])
ok = np.isfinite(lon) & np.isfinite(lat) & (np.isfinite(sm).sum(axis=1) > 300)
lon, lat, sm = lon[ok], lat[ok], sm[ok]
print("stations in the animation:", len(lon))
national = np.nanmean(sm, axis=0)

dates_all = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
on, wd = float(dates_all.onset.median()), float(dates_all.withdrawal.median())
dec = pd.read_csv(OUT / "dhm_declared_dates.csv")
dec = dec[(dec.year >= 1981) & (dec.year <= 2017)]
don, dwd = dec.onset_doy.mean(), dec.withdrawal_doy.mean()
print(f"objective onset {on:.0f}, withdrawal {wd:.0f}; declared mean {don:.1f}, {dwd:.1f}")

# ---------------- map geometry and kernel weights ----------------
r = shapefile.Reader(str(ROOT / "gis" / "Local Unit" / "local_unit"))
geoms = [shape(s.__geo_interface__).buffer(0) for s in r.shapes()]
nepal = unary_union(geoms).simplify(0.01, preserve_topology=True)
polys = list(nepal.geoms) if hasattr(nepal, "geoms") else [nepal]
gx = np.arange(80.0, 88.35, 0.025)
gy = np.arange(26.3, 30.55, 0.025)
GX, GY = np.meshgrid(gx, gy)
pts = np.c_[GX.ravel(), GY.ravel()]
dist2 = (pts[:, None, 0] - lon[None, :]) ** 2 + ((pts[:, None, 1] - lat[None, :])) ** 2
W = np.exp(-dist2 / (2 * H ** 2)).astype(np.float32)
del dist2


def field(day):
    v = sm[:, day]
    m = np.isfinite(v)
    num = W[:, m] @ v[m]
    den = W[:, m].sum(axis=1)
    f = (num / np.maximum(den, 1e-9)).reshape(GX.shape)
    return f


# ---------------- figure ----------------
MONTHS = [("Jan", 1), ("Feb", 32), ("Mar", 60), ("Apr", 91), ("May", 121), ("Jun", 152), ("Jul", 182), ("Aug", 213), ("Sep", 244), ("Oct", 274), ("Nov", 305), ("Dec", 335)]
SEASONS = [("Winter", 1, 59), ("Pre-monsoon", 60, on), ("Monsoon", on, wd), ("Post-monsoon", wd, 334), ("Winter", 335, 365)]
fig = plt.figure(figsize=(12.8, 8.0), dpi=100, facecolor="white")
ax_s = fig.add_axes([0.07, 0.80, 0.80, 0.07])
ax_m = fig.add_axes([0.07, 0.07, 0.72, 0.68])
ax_c = fig.add_axes([0.83, 0.17, 0.022, 0.50])
ax_t = fig.add_axes([0.07, 0.935, 0.80, 0.045]); ax_t.axis("off")

ax_s.imshow(national[None, :], aspect="auto", cmap=CMAP, norm=Normalize(0, VMAX), extent=[1, 366, 0, 1])
for lab, a, b in SEASONS:
    ax_s.text((a + b) / 2, 1.18, lab, ha="center", va="bottom", fontsize=11, fontweight="bold", color=INK, transform=ax_s.get_xaxis_transform() if False else ax_s.transData)
for x in (on, wd):
    ax_s.axvline(x, color=INK, lw=1.2)
ax_s.axvline(60, color=INK, lw=1.2); ax_s.axvline(335, color=INK, lw=1.2)
ax_s.plot([don, dwd], [-0.2, -0.2], "v", color="#D55E00", ms=7, clip_on=False)
ax_s.text(don, -0.62, "declared", color="#D55E00", fontsize=8, ha="center", va="top")
ax_s.text(dwd, -0.62, "declared", color="#D55E00", fontsize=8, ha="center", va="top")
ax_s.set_xlim(1, 366); ax_s.set_ylim(0, 1)
ax_s.set_yticks([]); ax_s.set_xticks([m[1] for m in MONTHS]); ax_s.set_xticklabels([m[0] for m in MONTHS], fontsize=9)
ax_s.tick_params(length=2, pad=1)
for s in ax_s.spines.values():
    s.set_visible(False)
marker = ax_s.axvspan(1, 3, ymin=-0.2, ymax=1.2, fc="none", ec="#D00000", lw=2.2, clip_on=False)

from matplotlib.patches import PathPatch
verts, codes = [], []
for pg in polys:
    c = np.asarray(pg.exterior.coords)
    verts += list(c)
    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
outline = PathPatch(MPath(verts, codes), fc="none", ec="#555555", lw=0.9, zorder=5)
ax_m.add_patch(outline)
im = ax_m.imshow(field(0), extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower", cmap=CMAP, norm=Normalize(0, VMAX), aspect=1 / np.cos(np.radians(28.4)), zorder=2, interpolation="bilinear")
im.set_clip_path(outline)
sc = ax_m.scatter(lon, lat, c=sm[:, 0], cmap=CMAP, norm=Normalize(0, VMAX), s=16, ec="#444444", lw=0.35, zorder=4)
ax_m.set_xlim(80.0, 88.35); ax_m.set_ylim(26.3, 30.55)
ax_m.set_xticks([81, 83, 85, 87]); ax_m.set_yticks([27, 28, 29, 30])
ax_m.set_xticklabels(["81°E", "83°E", "85°E", "87°E"], fontsize=10); ax_m.set_yticklabels(["27°N", "28°N", "29°N", "30°N"], fontsize=10)
for s in ax_m.spines.values():
    s.set_color("#444444")
date_txt = ax_m.text(0.02, 0.06, "", transform=ax_m.transAxes, fontsize=30, fontweight="bold", color=INK, zorder=6,
                     bbox=dict(fc="white", ec="none", alpha=0.75, pad=4))
ax_m.text(0.98, 0.02, "Gauge values smoothed with a 0.3° Gaussian kernel; dots are the gauges.\nMean of 1981–2017, %d-day running mean." % SMOOTH,
          transform=ax_m.transAxes, fontsize=8.5, ha="right", va="bottom", color="#333333", zorder=6)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, VMAX), cmap=CMAP), cax=ax_c, extend="max")
cb.set_label("Mean daily rainfall (mm per day)", fontsize=10)
cb.ax.tick_params(labelsize=9)
ax_t.text(0, 0.5, "Nepal rainfall through the year: gauges, 1981–2017", fontsize=15, fontweight="bold", va="center", color=INK)
fig.text(0.07, 0.025, "Season limits: median WiKS onset and withdrawal of the gauges (solid lines); triangles: mean DHM-declared dates, 1981–2017.", fontsize=9, color="#333333")


def render(day):
    im.set_data(field(day))
    sc.set_array(sm[:, day])
    t = pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=day)
    date_txt.set_text(t.strftime("%d %b"))
    marker.set_x(day)
    marker.set_width(3.0)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


if PREVIEW:
    for dd in (100, 175, 215, 290):
        imageio.imwrite(ANIM / f"preview_{dd}.png", render(dd))
    print("preview frames saved")
else:
    frames = []
    for dd in range(365):
        frames.append(render(dd))
    mp4 = ANIM / "nepal_rainfall_climatology.mp4"
    imageio.mimwrite(mp4, frames, fps=15, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
    small = [f[::2, ::2] for f in frames[::2]]
    imageio.mimwrite(ANIM / "nepal_rainfall_climatology.gif", small, duration=0.07, loop=0)
    print("saved", mp4, round(mp4.stat().st_size / 1e6, 1), "MB")
