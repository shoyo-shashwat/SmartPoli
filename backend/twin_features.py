"""
Digital Twin features for ONE moment, in plain Python (no pandas: the app runs on a small server).

This must produce exactly what ml/features.py produces in training; tests/test_twin_features.py checks the
two against each other. If you change a feature, change both and rerun that test.

Inputs
  grid    {datetime: mg/dL} readings on the 15-minute grid the models were trained on (gaps are simply absent)
  events  [(datetime, kind, iu, text)] kind is 'meal' | 'insulin' | 'oral'; iu is the insulin units when known
  profile {name: value} static facts (age, bmi, diabetes_years, hba1c, egfr, sex, hypertension, on_*)
  now     a grid time
"""
import math
from datetime import timedelta

STEP = timedelta(minutes=15)
NO_EVENT_MIN = 1440
NAN = float("nan")
STATIC = ["age", "bmi", "diabetes_years", "hba1c", "egfr", "sex", "hypertension", "on_insulin", "on_sulfonylurea", "on_metformin"]


def _num(x):
    return NAN if x is None else float(x)


def _since(events, kind, now):
    """minutes since the latest event of this kind at or before now (capped; none -> 1440)."""
    best = None
    for e in events:
        ts, k = e[0], e[1]
        if k == kind and ts <= now and (best is None or ts > best):
            best = ts
    if best is None:
        return float(NO_EVENT_MIN)
    return min(float(NO_EVENT_MIN), (now - best).total_seconds() / 60)


def build_features(feature_names, grid, events, profile, now):
    v = [grid.get(now - i * STEP, NAN) for i in range(5)]        # now, -15, -30, -45, -60 minutes
    win = [x for x in v if not math.isnan(x)]
    f = {"cgm_now": v[0], "cgm_change_30": v[0] - v[2], "cgm_change_60": v[0] - v[4]}
    if len(win) >= 4:
        mean = sum(win) / len(win)
        f["cgm_std_1h"] = math.sqrt(sum((x - mean) ** 2 for x in win) / (len(win) - 1))
        f["cgm_min_1h"], f["cgm_max_1h"] = min(win), max(win)
    else:
        f["cgm_std_1h"] = f["cgm_min_1h"] = f["cgm_max_1h"] = NAN
    f["hour_sin"] = math.sin(2 * math.pi * now.hour / 24)
    f["hour_cos"] = math.cos(2 * math.pi * now.hour / 24)
    f["min_since_meal"] = _since(events, "meal", now)
    f["min_since_insulin"] = _since(events, "insulin", now)
    f["min_since_oral"] = _since(events, "oral", now)
    dose = None
    for e in events:                                             # latest insulin row with a known dose size
        ts, k, iu = e[0], e[1], e[2]
        if k == "insulin" and iu is not None and ts <= now and (dose is None or ts > dose[0]):
            dose = (ts, iu)
    if f["min_since_insulin"] < NO_EVENT_MIN:
        f["last_insulin_iu"] = NAN if dose is None else float(dose[1])
    else:
        f["last_insulin_iu"] = 0.0
    for name in STATIC:
        f[name] = _num(profile.get(name))
    return f, [f[n] for n in feature_names]
