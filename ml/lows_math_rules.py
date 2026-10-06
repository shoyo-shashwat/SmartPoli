"""Mathematical (no-training) scores for lows, compared at the same alert budget as lows_alert_check.py.
score = current sugar + a * (change over the last 30 min): a linear extrapolation of where sugar is heading
(a = 0 is the plain sugar rule). Lower score = more worried. Dev patients, low events as in lows_alert_check.

Run:  python lows_math_rules.py data/shanghai_t2dm_clean
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
rid = d.recording_id.values


def stats(a):
    runs = int((a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])).sum())
    caught = pd.Series(a[d.ev.values > 0]).groupby(d.ev.values[d.ev.values > 0]).any().sum()
    return runs / days, 100 * caught / nev


def at_budget(score, budget):
    """Alert on the lowest-scoring fraction of moments. Alerts per day first rise as more moments alert and then fall
    again once alerts merge into long runs, so only the rising side (up to the peak) is used."""
    s = np.nan_to_num(score, nan=1e9)
    res = [stats(s <= np.quantile(s, q)) for q in np.linspace(0.001, 0.25, 250)]
    peak = int(np.argmax([r[0] for r in res]))
    ok = [r for r in res[:peak + 1] if r[0] <= budget]
    return ok[-1] if ok else res[0]


budgets = (0.92, 1.58, 2.12)
print("share of the", nev, "low events warned in time, at the same alerts per patient-day")
print(f"{'score':36s}" + "".join(f"{b:>8.2f}/day" for b in budgets))
rows = [(f"sugar + {a} x 30-min change", d.cgm_now + a * d.cgm_change_30) for a in (0, 0.5, 1, 2, 3)]
rows += [("sugar + 1 x 60-min change", d.cgm_now + d.cgm_change_60), ("1 h minimum + 1 x 30-min change", d.cgm_min_1h + d.cgm_change_30)]
for name, sc in rows:
    print(f"{name:36s}" + "".join(f"{at_budget(sc.values, b)[1]:>12.0f}%" for b in budgets))
print(f"{'constrained XGBoost (model)':36s}" + "".join(f"{at_budget(-d.raw.values, b)[1]:>12.0f}%" for b in budgets))
