"""Use the forecast model as a lows alerter: alert when the predicted sugar 30 or 60 minutes ahead is low.
Compared at the same alert budget as lows_math_rules.py (share of the 66 low events warned in time).
Run:  python lows_forecast_alert.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
exec(open("lows_math_rules.py").read().split("budgets =")[0])          # loads data, out-of-fold lows table d, at_budget
from forecast import add_targets, xgb

dd = add_targets(data, sys.argv[1]); dd = dd[(dd.group != "test") & dd.low60.notna()].copy()
for h in (30, 60):
    dd["delta"] = dd[f"y{h}"] - dd.cgm_now; dd[f"p{h}"] = np.nan
    for k in range(5):
        tr, es, va = dd[~dd.group.isin([k, (k + 1) % 5])], dd[dd.group == (k + 1) % 5], dd[dd.group == k]
        trn, esn = tr[tr.delta.notna()], es[es.delta.notna()]
        m = xgb().fit(trn[feats], trn.delta, eval_set=[(esn[feats], esn.delta)], verbose=False)
        dd.loc[va.index, f"p{h}"] = va.cgm_now + m.predict(va[feats])
d2 = d.merge(dd[["recording_id", "timestamp", "p30", "p60"]], on=["recording_id", "timestamp"], how="left"); assert len(d2) == len(d)
budgets = (0.92, 1.58, 2.12)
print("share of the", nev, "low events warned in time at the same alerts per day")
print(f"{'score':38s}" + "".join(f"{b:>8.2f}/day" for b in budgets))
for name, sc in [("sugar + 0.5 x 30-min change (rule)", d.cgm_now + 0.5 * d.cgm_change_30), ("forecast: predicted sugar at +60 min", d2.p60),
                 ("forecast: lower of +30 and +60 min", np.minimum(d2.p30, d2.p60)), ("constrained XGBoost warning model", -d.raw)]:
    print(f"{name:38s}" + "".join(f"{at_budget(np.asarray(sc, float), b)[1]:>12.0f}%" for b in budgets))
