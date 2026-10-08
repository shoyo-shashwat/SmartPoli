"""Repeat-alert suppression for the low alert (forecast: lowest of +30 and +60 min <= 90, or 100 with heart disease).
After an alert starts, a NEW alert is held back for `cooldown` minutes (an alert already running keeps going).

Plan fixed before looking: cooldown chosen on the 80 development patients only (out-of-fold forecasts), as the
longest cooldown that loses at most 1 development low event compared with no cooldown. The chosen value is then
applied ONCE to the locked test group (forecast models already saved by final_test.py, no retraining).
Run:  python lows_cooldown.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from baseline import prepare
from forecast import add_targets, xgb

folder = sys.argv[1]
data, feats = prepare(folder); data = add_targets(data, folder)
st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)


def episodes(df):
    y = df["y"]; newrun = y & ~(y.shift(1, fill_value=False) & (df.recording_id == df.recording_id.shift(1)))
    return np.where(y, newrun.cumsum(), 0)


def prep(df):
    L = df[df["low60"].notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    L["y"] = L["low60"].astype(bool); L["ev"] = episodes(L); L["heart"] = L.patient_id.map(heart).fillna(False)
    return L


def suppress(L, raw, minutes):
    """keep a raw alert moment if it continues an effective alert, or starts >= `minutes` after the last effective start."""
    out = np.zeros(len(L), bool); rid = L.recording_id.values; t = L.timestamp.values
    last_start = None; prev = False
    for i in range(len(L)):
        if i == 0 or rid[i] != rid[i - 1]: last_start, prev = None, False
        if raw[i]:
            if prev: out[i] = True
            elif last_start is None or (t[i] - last_start) / np.timedelta64(1, "m") >= minutes: out[i] = True; last_start = t[i]
        prev = out[i]
    return out


def score(L, a):
    rid = L.recording_id.values; a = np.asarray(a, bool)
    start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])
    run = np.cumsum(start) * a                               # alert-run id on alert moments
    days = len(L) / 96; n_alerts = int(start.sum())
    hit_runs = len(set(run[a & L.y.values])) if n_alerts else 0
    ev = L.ev.values; n_ev = len(set(ev[ev > 0])); caught = int(pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum()) if n_ev else 0
    return n_alerts / days, caught, n_ev, (hit_runs / n_alerts if n_alerts else 0.0), n_alerts


def show(name, L, raw, cooldowns):
    print(name)
    res = {}
    for c in cooldowns:
        a = raw if c == 0 else suppress(L, raw, c); r, ca, ne, pr, na = score(L, a); res[c] = (r, ca, ne, pr)
        print(f"  cooldown {c:3d} min: {r:.2f} alerts/patient-day ({na} alerts), warned {ca} of {ne} events ({100 * ca / ne:.0f}%), alerts followed by a low {100 * pr:.0f}%")
    return res


COOL = (0, 30, 60, 90, 120, 180, 240)
# ---- development: out-of-fold forecasts
dev = data[data.group != "test"].copy(); dd = dev[dev.low60.notna()].copy()
for h in (30, 60):
    dd["delta"] = dd[f"y{h}"] - dd.cgm_now; dd[f"p{h}"] = np.nan
    for k in range(5):
        tr, es, va = dd[~dd.group.isin([k, (k + 1) % 5])], dd[dd.group == (k + 1) % 5], dd[dd.group == k]
        trn, esn = tr[tr.delta.notna()], es[es.delta.notna()]
        m = xgb().fit(trn[feats], trn.delta, eval_set=[(esn[feats], esn.delta)], verbose=False)
        ok = va.delta.notna() & va.cgm_now.notna()
        dd.loc[va.index[ok.values], f"p{h}"] = va.cgm_now[ok] + m.predict(va[ok][feats])
D = prep(dd)
thr = np.where(D.heart, 100, 90); raw = np.nan_to_num(np.minimum(D.p30, D.p60).values, nan=1e9) <= thr
r = show(f"=== development patients ({D.patient_id.nunique()}), out-of-fold ===", D, raw, COOL)
chosen = max(c for c in COOL if r[c][1] >= r[0][1] - 1)
print(f"chosen on development: cooldown {chosen} min (longest losing at most 1 event)\n")

# ---- locked test group, once
test = data[data.group == "test"].copy()
for h in (30, 60):
    m = XGBRegressor(); m.load_model(f"models/forecast_{h}_mean.json")
    ok = test[f"y{h}"].notna() & test.cgm_now.notna(); test[f"p{h}"] = np.nan
    test.loc[ok, f"p{h}"] = test.cgm_now[ok] + m.predict(test[ok][feats])
T = prep(test); thr = np.where(T.heart, 100, 90); raw = np.nan_to_num(np.minimum(T.p30, T.p60).values, nan=1e9) <= thr
rt = show(f"=== locked test group ({T.patient_id.nunique()} patients), scored once ===", T, raw, (0, chosen))
