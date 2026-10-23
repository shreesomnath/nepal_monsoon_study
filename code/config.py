"""Shared settings for the monsoon-timing study (frozen from the synthetic sweep, code/50b)."""
from pathlib import Path

import os

WORK = Path(__file__).resolve().parents[1]        # this repository
ROOT = WORK.parent
OUT = WORK / "outputs"          # results of this study
FIG = WORK / "figures"
INP = Path(os.environ.get("NEPAL_INPUTS", ROOT / "outputs"))   # restricted DHM inputs: daily_qc.npz, station_province.csv, ...

# FROZEN 22 Oct 2026 from the synthetic sweep (code/50b, outputs/wiks_missing_rule.csv): at 5% missing days
# the best rules (0.40/1, 0.50/2, 0.50/3) are within sampling error of each other (about 61-62% of seasons
# within 7 days); 0.50/2 is taken because it requires two window half-widths and still returns a date for
# 100% of synthetic seasons at 15% missing days (the strict 0.80/3 rule failed on 34-48%).
RULE = dict(min_valid=0.5, min_windows=2, min_days=30)
MIN_SHARE = 0.85                # share of days observed (22 Mar - 10 Dec) for a station-year to be analysed
YEARS = range(1981, 2018)
SPAN = slice(80, 344)           # 22 March (index 80) to 10 December (index 343)
