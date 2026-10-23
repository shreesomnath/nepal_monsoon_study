"""Step 52: DHM-declared monsoon onset and withdrawal dates for eastern Nepal, 1968-2024.

Source: DHM Climate Analysis Section, 'Monsoon Onset and Withdrawal date information' (2024),
Table 1, saved locally by an earlier web fetch (dhm.gov.np/uploads/dhm/climateService/
monsoon_onset_and_withdrawal_dates_english3.pdf). The values were transcribed by hand from the
rendered table and must be spot-checked against the PDF. 1980 withdrawal is not available there.
The document gives no objective definition or station list; the dates are declared by DHM from
synoptic criteria. Output: outputs/dhm_declared_dates.csv (day of year in the actual year).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TXT = """
1968 17-Jun 21-Sep|1969 2-Jun 28-Sep|1970 16-Jun 30-Sep|1971 10-Jun 13-Sep|1972 18-Jun 26-Sep
1973 14-Jun 27-Sep|1974 19-Jun 21-Sep|1975 16-Jun 18-Sep|1976 9-Jun 21-Sep|1977 18-Jun 18-Sep
1978 8-Jun 16-Sep|1979 24-Jun 4-Sep|1980 12-Jun --|1981 23-Jun 11-Sep|1982 27-Jun 20-Sep
1983 24-Jun 29-Sep|1984 11-Jun 20-Sep|1985 17-Jun 26-Sep|1986 12-Jun 5-Oct|1987 14-Jun 26-Sep
1988 11-Jun 9-Sep|1989 11-Jun 2-Oct|1990 12-Jun 30-Sep|1991 9-Jun 19-Sep|1992 21-Jun 19-Sep
1993 8-Jun 28-Sep|1994 10-Jun 23-Sep|1995 4-Jun 2-Sep|1996 31-May 18-Sep|1997 14-Jun 23-Sep
1998 13-Jun 3-Oct|1999 11-Jun 8-Oct|2000 6-Jun 25-Sep|2001 3-Jun 25-Sep|2002 15-Jun 19-Sep
2003 16-Jun 30-Sep|2004 9-Jun 1-Oct|2005 20-Jun 2-Oct|2006 1-Jun 29-Sep|2007 7-Jun 9-Oct
2008 10-Jun 17-Oct|2009 23-Jun 15-Oct|2010 17-Jun 1-Oct|2011 15-Jun 7-Oct|2012 16-Jun 28-Sep
2013 14-Jun 19-Oct|2014 20-Jun 7-Oct|2015 13-Jun 3-Oct|2016 15-Jun 12-Oct|2017 12-Jun 16-Oct
2018 8-Jun 5-Oct|2019 20-Jun 12-Oct|2020 12-Jun 16-Oct|2021 11-Jun 11-Oct|2022 5-Jun 16-Oct
2023 14-Jun 15-Oct|2024 10-Jun 12-Oct
"""


def doy(y, s):
    if s == "--":
        return None
    return pd.to_datetime(f"{s}-{y}", format="%d-%b-%Y").dayofyear


rows = []
for item in TXT.replace("\n", "|").split("|"):
    item = item.strip()
    if not item:
        continue
    y, on, off = item.split()
    y = int(y)
    rows.append(dict(year=y, onset_text=on, withdrawal_text=off, onset_doy=doy(y, on),
                     withdrawal_doy=doy(y, off)))
df = pd.DataFrame(rows)
df["length_d"] = df.withdrawal_doy - df.onset_doy
df.to_csv(ROOT / "monsoon_timing" / "outputs" / "dhm_declared_dates.csv", index=False)
print(len(df), "years", df.year.min(), "-", df.year.max())
# the document quotes: normal onset 13 June, withdrawal 2 October, earliest onset 31 May 1996,
# latest onset 27 June 1982, earliest withdrawal 2 September 1995 and latest 19 October 2013,
# shortest season 73 days (1979), longest 134 days (2022)
print("earliest onset", df.loc[df.onset_doy.idxmin(), ["year", "onset_text"]].tolist(),
      "latest onset", df.loc[df.onset_doy.idxmax(), ["year", "onset_text"]].tolist())
print("earliest withdrawal", df.loc[df.withdrawal_doy.idxmin(), ["year", "withdrawal_text"]].tolist(),
      "latest withdrawal", df.loc[df.withdrawal_doy.idxmax(), ["year", "withdrawal_text"]].tolist())
print("shortest season", df.loc[df.length_d.idxmin(), ["year", "length_d"]].tolist(),
      "longest", df.loc[df.length_d.idxmax(), ["year", "length_d"]].tolist())
from scipy import stats
for col in ("onset_doy", "withdrawal_doy", "length_d"):
    d = df.dropna(subset=[col])
    sl = stats.theilslopes(d[col], d.year)[0] * 10
    tau, p = stats.kendalltau(d.year, d[col])
    print(f"{col}: Sen slope {sl:+.1f} d/decade, Kendall tau {tau:+.2f}, p = {p:.3g}, 1968-2024")
