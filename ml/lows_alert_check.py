"""Alert-level check for lows: at the same number of alerts per patient-day, how many low EVENTS does each
method warn about in time? An event = a run of consecutive moments whose label low60 is True (a low starts
within 60 min). It counts as caught if any alert falls inside that window. Out-of-fold, dev patients only.

Run:  python lows_alert_check.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from baseline import prepare
from calibration import oof

data, feats = prepare(sys.argv[1])
d = oof(data, feats, "low60", "XGB constrained").sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
d["y"] = d.low60.astype(bool); days = len(d) / 96
newrun = d.y & ~(d.y.shift(1, fill_value=False) & (d.recording_id == d.recording_id.shift(1)))
d["ev"] = np.where(d.y, newrun.cumsum(), 0); nev = d.ev[d.ev > 0].nunique()
print("events:", nev, " patient-days:", round(days))


def stats(a):
    s = pd.Series(a, index=d.index); runs = 0
    for rid, g in d.groupby("recording_id"):
        x = s[g.index].values; runs += int((x & ~np.r_[False, x[:-1]]).sum())
    caught = d[d.ev > 0].assign(a=s[d.ev > 0]).groupby("ev").a.any().sum()
    return runs / days, 100 * caught / nev


for T in (80, 90, 100, 110):
    r = stats((d.cgm_now <= T).values); lo, hi = 0, 0.2
    for _ in range(25):                      # model threshold that gives the same alerts per day as the rule
        mid = (lo + hi) / 2
        if stats((d.raw >= mid).values)[0] > r[0]: lo = mid
        else: hi = mid
    m = stats((d.raw >= hi).values)
    print(f"rule sugar<={T}: {r[0]:.2f} alerts/day catches {r[1]:.0f}% of low events | model p>={hi:.4f}: {m[0]:.2f}/day catches {m[1]:.0f}%")
