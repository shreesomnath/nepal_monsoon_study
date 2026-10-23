"""Step 56: apply the null-based quality control to the station dates.

A station-year's onset (withdrawal) is accepted when both its KS and WMW p-values are at or below
the Q-quantile of the null distribution of the best change point in block-shuffled seasons
(code/53; Q = 5% by default, i.e. at most about 5% of seasons without any real transition are
accepted). Station-level acceptance rates are written so that weakly seasonal stations (typically
high-mountain stations) are visible and can be excluded from regional composites.
Outputs: outputs/wiks_station_dates_qc.csv and outputs/wiks_station_acceptance.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import INP, OUT  # noqa: E402

Q = "q05"

if __name__ == "__main__":
    dates = pd.read_csv(OUT / "wiks_station_dates.csv", dtype={"station_id": str})
    thr = pd.read_csv(OUT / "wiks_null_thresholds.csv").set_index(["event", "test"])[Q]
    for ev in ("onset", "withdrawal"):
        ok = dates[ev].notna()
        for test in ("ks", "wmw"):
            ok &= dates[f"p_{test}_{ev}"] <= thr[(ev, test)]
        dates[f"ok_{ev}"] = ok
    dates.to_csv(OUT / "wiks_station_dates_qc.csv", index=False)
    g = dates.groupby("station_id")
    acc = pd.DataFrame({"n_seasons": g.size(), "onset_found": g.onset.apply(lambda s: s.notna().mean()),
                        "onset_accept": g.ok_onset.mean(), "withdrawal_accept": g.ok_withdrawal.mean()})
    sp = pd.read_csv(INP / "station_province.csv", dtype={"station_id": str})
    acc = acc.reset_index().merge(sp[["station_id", "lon", "lat", "elev", "province"]], on="station_id")
    acc.to_csv(OUT / "wiks_station_acceptance.csv", index=False)
    print("thresholds (%s):\n%s" % (Q, thr.to_string()))
    print("seasons: %d; onset accepted %.1f%%, withdrawal accepted %.1f%%" % (
        len(dates), 100 * dates.ok_onset.mean(), 100 * dates.ok_withdrawal.mean()))
    acc["elev_band"] = pd.cut(acc.elev, [0, 500, 1000, 1500, 2000, 4000])
    print(acc.groupby("elev_band", observed=True)[["onset_accept", "withdrawal_accept"]].agg(["mean", "count"]).round(2).to_string())
