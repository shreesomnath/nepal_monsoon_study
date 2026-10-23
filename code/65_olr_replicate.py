"""Step 65: replicate the OLR onset of Adhikari et al. (2022) and compare with their published Table 1.

Adhikari et al. (2022, Jalawaayu 2(1):19-32, doi:10.3126/jalawaayu.v2i1.45391) used NCEI 1 x 1 degree daily OLR
averaged over 25-28 N, 87-89.5 E, a 7-day moving mean, and defined the 'characteristic fall' as the first date
after 21 May on which OLR falls below 250 W m-2 for at least 3 consecutive days with OLR1 >= OLR2 >= OLR3
and OLR1 - OLR3 >= 5 W m-2. Their Table 1 (transcribed below from the paper, 1979-2019, 1985 missing in OLR)
gives the fall date and the MFD (DHM) onset. Their reported r = 0.6 is the correlation between the predicted onset
(fall date plus a constant 15 days) and the MFD onset, which equals the correlation of the fall dates with MFD onset.

Here: (1) the correlation implied by their table; (2) our fall dates from the NOAA PSL interpolated 2.5 degree OLR
with the same criteria for cell sets approximating their region; (3) the agreement of our fall dates with theirs.
Output: outputs/olr_adhikari_compare.csv, printed tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUT, WORK  # noqa: E402

TABLE = """1979 9-Jun 24-Jun|1980 27-May 12-Jun|1981 12-Jun 23-Jun|1982 3-Jun 27-Jun|1983 31-May 24-Jun|1984 22-May 11-Jun|1985 NA 17-Jun
1986 8-Jun 12-Jun|1987 27-May 14-Jun|1988 23-May 11-Jun|1989 22-May 11-Jun|1990 28-May 12-Jun|1991 24-May 9-Jun|1992 7-Jun 21-Jun
1993 24-May 8-Jun|1994 23-May 10-Jun|1995 30-May 4-Jun|1996 25-May 31-May|1997 23-May 14-Jun|1998 30-May 13-Jun|1999 23-May 11-Jun
2000 22-May 6-Jun|2001 25-May 3-Jun|2002 4-Jun 15-Jun|2003 2-Jun 16-Jun|2004 30-May 11-Jun|2005 9-Jun 20-Jun|2006 23-May 1-Jun
2007 4-Jun 7-Jun|2008 29-May 10-Jun|2009 10-Jun 23-Jun|2010 3-Jun 17-Jun|2011 12-Jun 15-Jun|2012 24-May 16-Jun|2013 23-May 14-Jun
2014 23-May 20-Jun|2015 26-May 13-Jun|2016 7-Jun 15-Jun|2017 23-May 12-Jun|2018 22-May 8-Jun|2019 11-Jun 20-Jun"""
rows = []
for item in TABLE.replace("\n", "|").split("|"):
    y, fall, mfd = item.split()
    y = int(y)
    d = lambda s: np.nan if s == "NA" else pd.to_datetime(f"{s}-{y}", format="%d-%b-%Y").dayofyear
    rows.append(dict(year=y, ad_fall=d(fall), ad_mfd=d(mfd)))
A = pd.DataFrame(rows).set_index("year")
m = A.dropna()
r_all = stats.pearsonr(m.ad_fall, m.ad_mfd)[0]
m2 = m.loc[1981:2017]
r_81 = stats.pearsonr(m2.ad_fall, m2.ad_mfd)[0]
print(f"Table 1 as printed: n={len(m)}, mean fall {m.ad_fall.mean():.1f} (29 May=149), r(fall, MFD onset) = {r_all:.2f}; "
      f"1981-2017 only: n={len(m2)}, r = {r_81:.2f}")

o = pd.read_csv(WORK / "inputs" / "olr_nepal_1981_2017.csv", parse_dates=["time"]).rename(columns={"latitude": "lat", "longitude": "lon"})
SETS = {"2cells_25-27.5N_87.5E": [(25.0, 87.5), (27.5, 87.5)],
        "1cell_27.5N_87.5E": [(27.5, 87.5)],
        "4cells_25-27.5N_85-87.5E": [(25.0, 85.0), (25.0, 87.5), (27.5, 85.0), (27.5, 87.5)],
        "all_15cells": [(la, lo) for la in (25.0, 27.5, 30.0) for lo in (80.0, 82.5, 85.0, 87.5, 90.0)]}


def fall_date(ma, y, thr=250.0):
    s = ma[(ma.index >= f"{y}-05-22") & (ma.index <= f"{y}-08-31")]
    v = s.to_numpy()
    for i in range(len(v) - 2):
        if v[i] < thr and v[i + 1] < thr and v[i + 2] < thr and v[i] >= v[i + 1] >= v[i + 2] and (v[i] - v[i + 2]) >= 5.0:
            return s.index[i].dayofyear
    return np.nan


res = {}
for name, cells in SETS.items():
    ser = pd.concat([o[(o.lat == la) & (o.lon == lo)].set_index("time").olr for la, lo in cells], axis=1).mean(axis=1).sort_index()
    ma = ser.rolling(7).mean()
    res[name] = pd.Series({y: fall_date(ma, y) for y in range(1981, 2018)})
    # climatological 7-day mean OLR on 21 May and on 15 June, for the threshold choice
    cl = ma[ma.index.strftime("%m-%d") == "05-21"].mean(), ma[ma.index.strftime("%m-%d") == "06-15"].mean()
    print(f"{name}: climatological 7-day mean OLR on 21 May {cl[0]:.1f}, on 15 June {cl[1]:.1f} W m-2")
R = pd.DataFrame(res)
R = R.join(A.loc[1981:2017])
R.to_csv(OUT / "olr_adhikari_compare.csv")
print("\nFall dates (day of year): mean, sd, share of years found, correlation with Adhikari's fall dates and with MFD onset (1981-2017):")
for name in SETS:
    z = R[[name, "ad_fall", "ad_mfd"]].dropna()
    print(f"{name:28s} mean {R[name].mean():6.1f} sd {R[name].std():5.1f} found {R[name].notna().mean():.2f} | "
          f"r(ours, their fall) {np.corrcoef(z[name], z.ad_fall)[0, 1]:+.2f} (n={len(z)}) | r(ours, MFD onset) {np.corrcoef(z[name], z.ad_mfd)[0, 1]:+.2f}")
print("Their own fall dates vs MFD onset, 1981-2017: r = %.2f" % r_81)
