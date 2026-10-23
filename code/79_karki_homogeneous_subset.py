"""Step 79: sensitivity of the withdrawal result to the homogeneity-tested network of Karki et al. (2017).

Karki et al. (2017, Climate 5, 4, Table 1) list 76 DHM stations with continuous records for 1970-2012 that passed a
relative homogeneity test (RHtest) on monthly series. This step restricts the composites to those stations that are also in
our network, matched by DHM index number, and repeats the withdrawal trend, the difference-series slope and the
correlation with the declared series. Outputs: outputs/karki_subset_match.csv, outputs/karki_subset_results.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

KARKI = {  # index number: (name, lat, lon, height m), Table 1 of Karki et al. (2017)
    106: ("Belauri santipur", 28.683, 80.35, 159), 209: ("Dhangadhi (atariya)", 28.8, 80.55, 187), 416: ("Nepalgunj Reg. off.", 28.052, 81.523, 144),
    510: ("Koilabas", 27.7, 82.533, 320), 101: ("Kakerpakha", 29.65, 80.5, 842), 103: ("Patan (west)", 29.467, 80.533, 1266),
    104: ("Dadeldhura", 29.3, 80.583, 1848), 201: ("Pipalkot", 29.617, 80.867, 1456), 202: ("Chainpur (west)", 29.55, 81.217, 1304),
    203: ("Silgadhi doti", 29.267, 80.983, 1360), 206: ("Asara ghat", 28.953, 81.442, 650), 303: ("Jumla", 29.275, 82.18, 2366),
    308: ("Nagma", 29.2, 81.9, 1905), 402: ("Dailekh", 28.85, 81.717, 1402), 404: ("Jajarkot", 28.7, 82.2, 1231),
    406: ("Surkhet", 28.587, 81.635, 720), 504: ("Libang gaun", 28.3, 82.633, 1270), 511: ("Salyan bazar", 28.383, 82.167, 1457),
    703: ("Butwal", 27.694, 83.466, 205), 902: ("Rampur", 27.654, 84.351, 169), 903: ("Jhawani", 27.591, 84.522, 177),
    907: ("Amlekhganj", 27.281, 84.992, 310), 909: ("Simara airport", 27.164, 84.98, 137), 910: ("Nijgadh", 27.183, 85.167, 244),
    911: ("Parwanipur", 27.079, 84.933, 115), 912: ("Ramoli bairiya", 27.017, 85.383, 152), 1109: ("Pattharkot (east)", 27.1, 85.66, 162),
    1110: ("Tulsi", 27.013, 85.921, 251), 1111: ("Janakpur airport", 26.711, 85.924, 78), 701: ("Ridi bazar", 27.95, 83.433, 442),
    722: ("Musikot", 28.167, 83.267, 1280), 802: ("Khudi bazar", 28.283, 84.367, 823), 804: ("Pokhara airport", 28.2, 83.979, 827),
    807: ("Kunchha", 28.133, 84.35, 855), 808: ("Bandipur", 27.942, 84.406, 995), 810: ("Chapkot", 27.883, 83.817, 460),
    814: ("Lumle", 28.297, 83.818, 1740), 904: ("Chisapani gadhi", 27.56, 85.139, 1729), 1008: ("Nawalpur", 27.813, 85.625, 1457),
    1015: ("Thankot", 27.688, 85.221, 1457), 1022: ("Godavari", 27.593, 85.379, 1527), 1023: ("Dolal ghat", 27.639, 85.705, 659),
    1029: ("Khumaltar", 27.652, 85.326, 1334), 1030: ("Kathmandu airport", 27.704, 85.373, 1337), 1115: ("Nepalthok", 27.42, 85.849, 698),
    601: ("Jomsom", 28.78, 83.72, 2744), 604: ("Thakmarpha", 28.739, 83.681, 2655), 607: ("Lete", 28.633, 83.609, 2490),
    1112: ("Chisapani bazar", 26.93, 86.145, 107), 1213: ("Udayapur gadhi", 26.933, 86.517, 1175), 1216: ("Siraha", 26.656, 86.212, 102),
    1311: ("Dharan bazar", 26.792, 87.285, 310), 1316: ("Chatara", 26.82, 87.159, 105), 1319: ("Biratnagar airport", 26.481, 87.264, 72),
    1320: ("Tarahara", 26.699, 87.279, 121), 1408: ("Damak", 26.671, 87.703, 119), 1409: ("Anarmani birta", 26.625, 87.989, 122),
    1102: ("Charikot", 27.667, 86.05, 1940), 1103: ("Jiri", 27.633, 86.233, 2003), 1202: ("Chaurikhark", 27.7, 86.717, 2619),
    1203: ("Pakarnas", 27.443, 86.569, 1944), 1204: ("Aisealukhark", 27.36, 86.749, 2063), 1206: ("Okhaldhunga", 27.308, 86.504, 1731),
    1207: ("Mane bhanjyang", 27.215, 86.444, 1528), 1210: ("Kurule ghat", 27.136, 86.43, 341), 1211: ("Khotang bazar", 27.029, 86.843, 1305),
    1303: ("Chainpur (east)", 27.292, 87.317, 1262), 1305: ("Leguwa ghat", 27.154, 87.289, 444), 1306: ("Munga", 27.05, 87.244, 1457),
    1307: ("Dhankuta", 26.983, 87.346, 1192), 1308: ("Mul ghat", 26.932, 87.32, 286), 1309: ("Tribeni", 26.914, 87.16, 146),
    1322: ("Machuwaghat", 26.938, 87.155, 168), 1325: ("Dingla", 27.353, 87.146, 1169), 1403: ("Lungthung", 27.55, 87.783, 1780),
    1410: ("Himali gaun", 26.887, 88.027, 1654),
}
assert len(KARKI) == 76

sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
print("station table columns:", sp.columns.tolist())
# match by index number where the id parses to an integer, then confirm by distance to the listed coordinates
sp["num"] = pd.to_numeric(sp.station_id.str.extract(r"(\d+)")[0], errors="coerce")
rows = []
for k, (name, lat, lon, h) in KARKI.items():
    cand = sp[sp.num == k]
    if len(cand):
        r = cand.iloc[0]
        d = np.hypot((r.lat - lat) * 111, (r.lon - lon) * 98)
        rows.append(dict(karki_id=k, name=name, station_id=r.station_id, dist_km=d))
    else:
        rows.append(dict(karki_id=k, name=name, station_id=None, dist_km=np.nan))
M = pd.DataFrame(rows)
M.to_csv(OUT / "karki_subset_match.csv", index=False)
ok = M[(M.station_id.notna()) & (M.dist_km < 10)]
print(f"matched by index and within 10 km: {len(ok)} of 76; matched but far: {int(((M.dist_km >= 10)).sum())}; no index match: {int(M.station_id.isna().sum())}")

yrs = np.arange(1981, 2018)
dec = pd.read_csv(OUT / "dhm_declared_dates.csv").set_index("year")["withdrawal_doy"].reindex(yrs).to_numpy(float)
wk = pd.read_csv(OUT / "wiks_station_dates_qc.csv", dtype={"station_id": str})
cb = pd.read_csv(OUT / "cb_station_dates.csv", dtype={"station_id": str})
rng = np.random.default_rng(79)


def bidx(n, L=4):
    nb = int(np.ceil(n / L))
    s = rng.integers(0, n - L + 1, nb)
    return np.concatenate([np.arange(a, a + L) for a in s])[:n]


def slope(y, idx):
    idx = idx[np.isfinite(y[idx])]
    if len(np.unique(idx)) < 12:
        return np.nan
    return stats.theilslopes(y[idx], yrs[idx].astype(float))[0] * 10


def block_ci(y, nb=4000):
    b = np.array([slope(y, bidx(len(y))) for _ in range(nb)])
    return slope(y, np.arange(len(y))), np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)


res = []
ids = set(ok.station_id)
for name, df, col in (("WiKS", wk, "withdrawal"), ("two-knot", cb, "cb_withdrawal")):
    sub = df[df.station_id.isin(ids)]
    P = sub.pivot(index="station_id", columns="year", values=col).reindex(columns=yrs)
    for lab, minn in (("at least 30 dates", 30), ("at least 20 dates", 20)):
        Q = P[P.notna().sum(axis=1) >= minn]
        if len(Q) < 8:
            continue
        y = Q.median(axis=0).to_numpy(float)
        s, lo, hi = block_ci(y)
        d = np.where(np.isfinite(y), dec - y, np.nan)
        ds, dlo, dhi = block_ci(d)
        okk = np.isfinite(y)
        res.append(dict(definition=name, rule=lab, n_stations=len(Q), slope=s, lo=lo, hi=hi, diff_slope=ds, diff_lo=dlo, diff_hi=dhi,
                        r_declared=np.corrcoef(y[okk], dec[okk])[0, 1]))
R = pd.DataFrame(res)
R.to_csv(OUT / "karki_subset_results.csv", index=False)
print(R.round(2).to_string(index=False))
