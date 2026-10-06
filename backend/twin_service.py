"""
Digital Twin inference: loads the saved XGBoost models (backend/twin_models/) and turns a patient's readings,
meals/doses and facts into what the doctor screen shows. Uses xgboost + the standard library only.

The twin only informs. It never diagnoses and never suggests a treatment change.
"""
import json
import math
import os
from datetime import datetime, timedelta

from twin_features import build_features, STEP

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin_models")
HORIZONS = [15, 30, 60, 120]
MODEL_VERSION = "twin-1 (XGBoost, trained on ShanghaiT2DM)"
TRAINING_DATA = "ShanghaiT2DM (Zhao et al., Scientific Data 2023), 79 patients; checked on 21 patients it had never seen"
DISCLAIMER = "Risk flag for the doctor, not a diagnosis or a treatment change."

_models = None
_meta = None


def _load():
    global _models, _meta
    if _models is None:
        import xgboost as xgb
        _meta = json.load(open(os.path.join(MODEL_DIR, "meta.json")))
        names = ["spike120", "large_rise120"] + [f"forecast_{h}_{k}" for h in HORIZONS for k in ("mean", "q10", "q90")]
        loaded = {}
        for n in names:
            b = xgb.Booster(); b.load_model(os.path.join(MODEL_DIR, n + ".json")); loaded[n] = b
        _models = loaded
    return _models, _meta


def feature_names():
    return _load()[1]["features"]


def _predict(booster, row, names, contribs=False):
    import numpy as np
    import xgboost as xgb
    dm = xgb.DMatrix(np.array([row], dtype=float), feature_names=names)
    return booster.predict(dm, pred_contribs=contribs, validate_features=False)[0]


def _interp(known, h):
    """linear interpolation between the horizons we have models for."""
    xs = sorted(known)
    if h <= xs[0]:
        return known[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if a <= h <= b:
            return known[a] + (known[b] - known[a]) * (h - a) / (b - a)
    return known[xs[-1]]


def snap(grid, at):
    """latest grid time at or before `at` that has a reading."""
    ts = [t for t in grid if t <= at]
    return max(ts) if ts else None


def _fmt_since(m):
    if m >= 1440:
        return None
    if m < 90:
        return f"{int(round(m))} minutes ago"
    return f"{m / 60:.1f} hours ago".replace(".0 ", " ")


def _reason_text(name, f):
    v = f.get(name)
    if name == "cgm_now":
        return f"Sugar is {round(v)} mg/dL"
    if name == "cgm_change_30" and v == v:
        return f"Sugar {'fell' if v < 0 else 'rose'} {abs(round(v))} mg/dL in the last 30 minutes"
    if name == "cgm_change_60" and v == v:
        return f"Sugar {'fell' if v < 0 else 'rose'} {abs(round(v))} mg/dL in the last hour"
    if name == "cgm_min_1h" and v == v:
        return f"The lowest reading in the last hour was {round(v)} mg/dL"
    if name == "cgm_max_1h" and v == v:
        return f"The highest reading in the last hour was {round(v)} mg/dL"
    if name == "cgm_std_1h" and v == v:
        return "Sugar has been swinging over the last hour"
    if name == "min_since_meal":
        s = _fmt_since(v)
        return f"A meal was logged {s}" if s else "No meal logged in the last 24 hours"
    if name == "min_since_insulin":
        s = _fmt_since(v)
        return f"Insulin was given {s}" if s else "No insulin logged in the last 24 hours"
    if name == "min_since_oral":
        s = _fmt_since(v)
        return f"A diabetes tablet was taken {s}" if s else "No diabetes tablet logged in the last 24 hours"
    if name == "last_insulin_iu":
        return f"The last insulin dose was {v:g} units" if v else "No recent insulin dose"
    if name in ("hour_sin", "hour_cos"):
        return "Time of day"
    return {"age": f"Age {round(v)}" if v == v else "Age", "bmi": "Body size (BMI)", "diabetes_years": "Years with diabetes", "hba1c": "Long-term sugar control (HbA1c)",
            "egfr": "Kidney function (eGFR)", "sex": "Sex", "hypertension": "High blood pressure", "on_insulin": "Takes insulin",
            "on_sulfonylurea": "Takes a sulfonylurea tablet", "on_metformin": "Takes metformin"}.get(name, name)


def _reasons(contribs, names, f, raises_when_negative, top=4):
    """top contributing features as {text, up, weight}; up = raises the risk the card is about."""
    pairs = []
    hour = 0.0
    for n, c in zip(names, contribs[:-1]):
        if n in ("hour_sin", "hour_cos"):
            hour += c
            continue
        pairs.append((n, c))
    pairs.append(("hour_sin", hour))
    pairs = [p for p in pairs if abs(p[1]) > 1e-6]
    pairs.sort(key=lambda p: -abs(p[1]))
    pairs = pairs[:top]
    top_abs = max([abs(p[1]) for p in pairs] or [1])
    return [{"text": _reason_text(n, f), "up": bool((c < 0) == raises_when_negative), "weight": round(100 * abs(c) / top_abs)} for n, c in pairs]


def _low_episodes(grid):
    n, prev = 0, False
    ts = sorted(grid)
    flags = [grid[t] < 70 for t in ts]
    for i in range(len(flags) - 1):
        if flags[i] and flags[i + 1] and not prev:
            n += 1
        prev = flags[i]
    return n


def predict(profile, grid, events, at, history_minutes=360):
    """Everything the Twin tab needs for one patient at one moment. `grid` is {datetime: mg/dL}."""
    models, meta = _load(); names = meta["features"]
    now = snap(grid, at)
    if now is None:
        raise ValueError("no readings at or before this time")
    f, row = build_features(names, grid, events, profile, now)
    v = f["cgm_now"]
    delta = {}
    for h in HORIZONS:
        delta[h] = {k: float(_predict(models[f"forecast_{h}_{k}"], row, names)) for k in ("mean", "q10", "q90")}
    pts = []
    for h in range(15, 121, 15):
        m = v + _interp({k: delta[k]["mean"] for k in HORIZONS}, h)
        a = v + _interp({k: delta[k]["q10"] for k in HORIZONS}, h)
        b = v + _interp({k: delta[k]["q90"] for k in HORIZONS}, h)
        pts.append({"minutes_ahead": h, "time": (now + timedelta(minutes=h)).isoformat(), "mg_dl": round(m, 1),
                    "low": round(min(a, b, m), 1), "high": round(max(a, b, m), 1)})
    p30, p60 = v + delta[30]["mean"], v + delta[60]["mean"]
    lowest, lowest_at = (p30, 30) if p30 <= p60 else (p60, 60)
    alert_t = 100 if profile.get("heart_disease") else 90
    watch_t = alert_t + 20
    already_low, already_high = v < 70, v > 180
    if already_low:
        tier, low_text = "alert", f"Sugar is already low at {round(v)} mg/dL."
    elif lowest <= alert_t:
        tier, low_text = "alert", f"Sugar is {round(v)} mg/dL and the forecast falls to about {round(lowest)} mg/dL within {lowest_at} minutes."
    elif lowest <= watch_t:
        tier, low_text = "watch", f"Sugar is {round(v)} mg/dL and the forecast reaches about {round(lowest)} mg/dL within the hour."
    else:
        tier, low_text = "calm", f"Sugar is {round(v)} mg/dL and no low is expected in the next hour."
    headline = {"alert": ("Sugar is already low" if already_low else f"Possible low in about {lowest_at} minutes"),
                "watch": "Sugar is drifting down. Keep an eye on the next hour", "calm": "No low expected in the next hour"}[tier]

    sc = _predict(models["spike120"], row, names, contribs=True)
    p_spike = float(1 / (1 + math.exp(-sc.sum())))                 # contributions sum to the margin
    rc = _predict(models["large_rise120"], row, names, contribs=True)
    p_rise = float(1 / (1 + math.exp(-rc.sum())))
    pct = round(100 * p_spike)
    high_tier = "alert" if pct >= 40 else ("watch" if pct >= 15 else "calm")
    high_text = "Sugar is already above 180 mg/dL." if already_high else f"About {pct} in 100 chance of going above 180 mg/dL in the next 2 hours."

    mean60 = _predict(models["forecast_60_mean"], row, names, contribs=True)
    low_reasons = _reasons(mean60, names, f, raises_when_negative=True)           # pushes sugar down -> raises low risk
    high_reasons = _reasons(sc, names, f, raises_when_negative=False)

    checks = []
    if f["min_since_insulin"] < 1440:
        last_ins = max([e for e in events if e[1] == "insulin" and e[0] <= now], key=lambda e: e[0])
        checks.append(f"Last insulin {_fmt_since(f['min_since_insulin'])}" + (f" ({last_ins[3]})" if last_ins[3] else ""))
    if f["min_since_meal"] < 1440:
        checks.append(f"Last meal logged {_fmt_since(f['min_since_meal'])}")
    days = (max(grid) - min(grid)).total_seconds() / 86400
    n_low = _low_episodes({t: x for t, x in grid.items() if t <= now})
    if n_low:
        checks.append(f"This patient had {n_low} {'low' if n_low == 1 else 'lows'} in the readings so far ({days:.0f} days on record)")
    if profile.get("sensor_note"):
        checks.append(profile["sensor_note"])

    start = now - timedelta(minutes=history_minutes)
    hist = [[t.isoformat(), x] for t, x in sorted(grid.items()) if start <= t <= now]
    ev = [{"time": e[0].isoformat(), "kind": e[1], "text": e[3]} for e in events if start <= e[0] <= now + timedelta(minutes=120)]
    truth = [[t.isoformat(), x] for t, x in sorted(grid.items()) if now < t <= now + timedelta(minutes=120)]
    return {
        "as_of": now.isoformat(), "current": round(v, 1), "range": {"start": min(grid).isoformat(), "end": max(grid).isoformat()},
        "headline": headline, "low": {"tier": tier, "text": low_text, "lowest": round(lowest, 1), "lowest_in_minutes": lowest_at,
                                      "already_low": already_low, "alert_at_or_under": alert_t, "watch_at_or_under": watch_t, "reasons": low_reasons},
        "high": {"tier": high_tier, "chance_percent": pct, "text": high_text, "already_high": already_high, "reasons": high_reasons},
        "large_rise_percent": round(100 * p_rise),
        "forecast": pts, "history": hist, "events": ev, "what_really_happened": truth, "checks": checks,
        "heart_note": bool(profile.get("heart_disease")) and tier != "calm",
        "confidence": profile.get("confidence") or "ok", "confidence_reason": profile.get("confidence_reason"),
        "model_version": MODEL_VERSION, "training_data": TRAINING_DATA, "disclaimer": DISCLAIMER,
    }
