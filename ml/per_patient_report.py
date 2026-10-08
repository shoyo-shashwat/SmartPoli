"""Per-patient breakdown on the 21 locked test patients, from the saved models (inference only, nothing trained or tuned).
For each patient: days of data, spike and large-rise AUC (when both outcomes occur), forecast error against 'sugar stays put',
and the low alert (events, warned, alerts per day).
Run from ml/:  python per_patient_report.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor
from baseline import prepare
from forecast import add_targets

folder = sys.argv[1]
data, feats = prepare(folder); data = add_targets(data, folder)
test = data[data.group == "test"].copy()
st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
test["p_spike"] = np.nan; test["p_rise"] = np.nan
for t, col in (("spike120", "p_spike"), ("large_rise120", "p_rise")):
    m = XGBClassifier(); m.load_model(f"models/{t}.json"); ok = test[t].notna(); test.loc[ok, col] = m.predict_proba(test[ok][feats])[:, 1]
for h in (30, 60):
    m = XGBRegressor(); m.load_model(f"models/forecast_{h}_mean.json"); ok = test[f"y{h}"].notna() & test.cgm_now.notna()
    test[f"p{h}"] = np.nan; test.loc[ok, f"p{h}"] = test.cgm_now[ok] + m.predict(test[ok][feats])

rows = []
for pid, g in test.groupby("patient_id"):
    g = g.sort_values(["recording_id", "timestamp"]); r = dict(patient=int(pid), days=round(len(g) / 96, 1), heart=bool(heart.get(pid, False)))
    for t, col, name in (("spike120", "p_spike", "spike_auc"), ("large_rise120", "p_rise", "rise_auc")):
        e = g[g[t].notna()]; r[name] = round(roc_auc_score(e[t].astype(int), e[col]), 2) if e[t].astype(int).nunique() == 2 else np.nan
    for h in (30, 60):
        e = g[g[f"y{h}"].notna() & g[f"p{h}"].notna()]
        r[f"mae{h}"] = round(float((e[f"p{h}"] - e[f"y{h}"]).abs().mean()), 1); r[f"stay{h}"] = round(float((e.cgm_now - e[f"y{h}"]).abs().mean()), 1)
    L = g[g.low60.notna()].reset_index(drop=True); y = L.low60.astype(bool)
    nr = y & ~(y.shift(1, fill_value=False) & (L.recording_id == L.recording_id.shift(1))); ev = np.where(y, nr.cumsum(), 0)
    a = np.nan_to_num(np.minimum(L.p30, L.p60).values, nan=1e9) <= (100 if r["heart"] else 90)
    rid = L.recording_id.values; start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])
    n_ev = len(set(ev[ev > 0])); caught = int(pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum()) if n_ev else 0
    r["low_events"], r["warned"], r["alerts_day"] = n_ev, caught, round(start.sum() / (len(L) / 96), 2)
    rows.append(r)
T = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
print("Per patient, locked test group (saved models, no retraining)\n")
print(T.to_string(index=False))
print("\nSummary over the 21 patients")
for c, n in (("spike_auc", "spike AUC"), ("rise_auc", "large-rise AUC")):
    v = T[c].dropna(); print(f"  {n}: {len(v)} patients with both outcomes; min {v.min():.2f}, median {v.median():.2f}, max {v.max():.2f}")
for h in (30, 60):
    better = int((T[f"mae{h}"] < T[f"stay{h}"]).sum()); print(f"  +{h} min forecast beats 'stays put' for {better} of {len(T)} patients (typical error {T[f'mae{h}'].median():.1f} vs {T[f'stay{h}'].median():.1f} mg/dL; worst patient {T[f'mae{h}'].max():.1f})")
w = T[T.low_events > 0]; print(f"  lows: {int(w.low_events.sum())} events in {len(w)} patients; warned {int(w.warned.sum())}; alerts/day median {T.alerts_day.median():.2f}, max {T.alerts_day.max():.2f}")
T.to_csv("per_patient_results.csv", index=False)
