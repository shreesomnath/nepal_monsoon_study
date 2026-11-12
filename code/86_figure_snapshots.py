"""Step 86: snapshot figure of the daily rainfall climatology of Nepal (six dates through the season), from the same
climatology as the animation (code/85): gauge means for 1981-2017, 11-day running mean, 0.3 degree Gaussian kernel.
Outputs: figures/figS_snapshots.pdf, .png (600 dpi), figS_snapshots_source.csv (regional mean rainfall on each date)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.colors import Normalize
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FIG, INP, OUT, ROOT  # noqa: E402

CMAP, VMAX, H, SMOOTH = plt.get_cmap("YlGnBu"), 25.0, 0.30, 11
INK = "#222222"
MM = 1 / 25.4
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "text.color": INK, "pdf.fonttype": 42})

# ---------- climatology (as in code/85) ----------
d = np.load(INP / "daily_qc.npz", allow_pickle=True)
X, status = d["X"].astype(float), d["status"]
ids = [str(s) for s in d["station_id"]]
dt = pd.to_datetime(d["dates"])
V = np.where(status == 0, X, np.nan)
keep = (dt.year >= 1981) & (dt.year <= 2017)
dt, V = dt[keep], V[:, keep]
feb29 = (dt.month == 2) & (dt.day == 29)
dt, V = dt[~feb29], V[:, ~feb29]
doy = np.array([pd.Timestamp(year=2001, month=m, day=dd).dayofyear for m, dd in zip(dt.month, dt.day)])
clim = np.full((len(ids), 365), np.nan)
for k in range(365):
    col = V[:, doy == k + 1]
    n = np.isfinite(col).sum(axis=1)
    m = np.nanmean(np.where(np.isfinite(col), col, np.nan), axis=1)
    clim[:, k] = np.where(n >= 15, m, np.nan)
pad = SMOOTH // 2
ext = np.concatenate([clim[:, -pad:], clim, clim[:, :pad]], axis=1)
sm = np.array([np.nanmean(ext[:, k:k + SMOOTH], axis=1) for k in range(365)]).T

sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str}).set_index("station_id")
lon = np.array([sp.loc[s, "lon"] if s in sp.index else np.nan for s in ids])
lat = np.array([sp.loc[s, "lat"] if s in sp.index else np.nan for s in ids])
region = np.array([("West" if sp.loc[s, "province"] in ("Sudur Pashchim", "Karnali", "Lumbini") else "Central" if sp.loc[s, "province"] == "Gandaki" else "East") if s in sp.index else "" for s in ids])
ok = np.isfinite(lon) & np.isfinite(lat) & (np.isfinite(sm).sum(axis=1) > 300)
lon, lat, sm, region = lon[ok], lat[ok], sm[ok], region[ok]

dates_all = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
on, wd = int(dates_all.onset.median()), int(dates_all.withdrawal.median())
dec = pd.read_csv(OUT / "dhm_declared_dates.csv")
dec = dec[(dec.year >= 1981) & (dec.year <= 2017)]
don, dwd = round(dec.onset_doy.mean()), round(dec.withdrawal_doy.mean())


def label(day):          # day of year on the non-leap calendar
    return (pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=day - 1)).strftime("%-d %b") if False else (pd.Timestamp(year=2001, month=1, day=1) + pd.Timedelta(days=day - 1)).strftime("%d %b").lstrip("0")


SNAP = [(105, "Pre-monsoon"), (on, "Median onset"), (196, "Monsoon"), (227, "Monsoon"), (wd, "Median withdrawal"), (288, "Post-monsoon")]
print("onset", on, label(on), "| withdrawal", wd, label(wd), "| declared mean", don, label(don), dwd, label(dwd))

# ---------- map ----------
r = shapefile.Reader(str(ROOT / "gis" / "Local Unit" / "local_unit"))
nepal = unary_union([shape(s.__geo_interface__).buffer(0) for s in r.shapes()]).simplify(0.01, preserve_topology=True)
polys = list(nepal.geoms) if hasattr(nepal, "geoms") else [nepal]
gx, gy = np.arange(80.0, 88.35, 0.03), np.arange(26.3, 30.55, 0.03)
GX, GY = np.meshgrid(gx, gy)
pts = np.c_[GX.ravel(), GY.ravel()]
W = np.exp(-((pts[:, None, 0] - lon[None, :]) ** 2 + (pts[:, None, 1] - lat[None, :]) ** 2) / (2 * H ** 2)).astype(np.float32)


def field(day):
    v = sm[:, day - 1]
    m = np.isfinite(v)
    return ((W[:, m] @ v[m]) / np.maximum(W[:, m].sum(axis=1), 1e-9)).reshape(GX.shape)


verts, codes = [], []
for pg in polys:
    c = np.asarray(pg.exterior.coords)
    verts += list(c)
    codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
outline_path = MPath(verts, codes)

fig, axs = plt.subplots(2, 3, figsize=(180 * MM, 100 * MM), layout="constrained")
rows = []
for ax, (day, tag), letter in zip(axs.ravel(), SNAP, "abcdef"):
    patch = PathPatch(outline_path, fc="none", ec="#555555", lw=0.6, zorder=5)
    ax.add_patch(patch)
    im = ax.imshow(field(day), extent=[gx[0], gx[-1], gy[0], gy[-1]], origin="lower", cmap=CMAP, norm=Normalize(0, VMAX),
                   aspect=1 / np.cos(np.radians(28.4)), interpolation="bilinear", zorder=2)
    im.set_clip_path(patch)
    ax.scatter(lon, lat, c=sm[:, day - 1], cmap=CMAP, norm=Normalize(0, VMAX), s=3.0, ec="#444444", lw=0.15, zorder=4)
    ax.set_xlim(80.0, 88.35); ax.set_ylim(26.3, 30.55)
    ax.set_xticks([82, 84, 86, 88]); ax.set_yticks([27, 28, 29, 30])
    ax.set_xticklabels(["82", "84", "86", "88"], fontsize=6.5); ax.set_yticklabels(["27", "28", "29", "30"], fontsize=6.5)
    ax.tick_params(length=2, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.set_title(f"{label(day)}: {tag}", loc="left", fontsize=8, pad=2)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")
    v = sm[:, day - 1]
    row = dict(panel=letter, day=day, date=label(day), tag=tag, all_mean=np.nanmean(v))
    for rg in ("East", "Central", "West"):
        row[f"{rg}_mean"] = np.nanmean(v[region == rg])
    rows.append(row)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, VMAX), cmap=CMAP), ax=axs, orientation="horizontal", shrink=0.5, pad=0.02, aspect=40, extend="max")
cb.set_label("Mean daily rainfall, 1981 to 2017 (mm per day)", fontsize=8)
cb.ax.tick_params(labelsize=7)
S = pd.DataFrame(rows)
S.to_csv(FIG / "figS_snapshots_source.csv", index=False)
for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(FIG / f"figS_snapshots.{ext}", **kw)
print(S.round(2).to_string(index=False))
