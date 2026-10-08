"""Experiment 5: does adapting to one patient's own recent history help? (per-patient recalibration)
Each recording is split in time: the first 3 days are the 'adaptation window'; everything after is scored. Only the patient's own
EARLIER data is ever used (no future values).
  Methods for the sugar forecast (+30 / +60 min):
    M0  global model as is
    M1  fixed bias: add the median error seen in the first 3 days
    M2  rolling bias: add the median of the errors already known over the last 2 days (updates all the time)
    M3  linear recalibration on the first 3 days, shrunk halfway toward 'no change'
  Spike probability: M0 as is, M5 shift the average level toward this patient's own spike rate in the first 3 days (half-way shrink).
Plan fixed before looking: methods are compared on the 79 development patients (out-of-fold models). Only if one method clearly beats M0
there (pooled error lower by at least 0.3 mg/dL at +60 min, and not worse at +30) is it shown on the 21 locked test patients, once.
Run from ml/:  python personalize.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from baseline import prepare
from forecast import add_targets, xgb

folder = sys.argv[1]
OUT = open("personalize_results.txt", "w", encoding="utf8")


def say(s=""):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


data, feats = prepare(folder); data = add_targets(data, folder)
st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
WIN = 288          # 3 days of 15-minute rows
CLS = dict(learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=4)


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-z))


def dev_oof(dev):
    d = dev.copy()
    for h in (30, 60):
        d["delta"] = d[f"y{h}"] - d.cgm_now; d[f"p{h}"] = np.nan
        for k in range(5):
            tr, es, va = d[~d.group.isin([k, (k + 1) % 5])], d[d.group == (k + 1) % 5], d[d.group == k]
            trn, esn = tr[tr.delta.notna()], es[es.delta.notna()]
            m = xgb().fit(trn[feats], trn.delta, eval_set=[(esn[feats], esn.delta)], verbose=False)
            ok = (va.delta.notna() & va.cgm_now.notna()).values; d.loc[va.index[ok], f"p{h}"] = va.cgm_now[ok] + m.predict(va[ok][feats])
    d["p_spike"] = np.nan
    for k in range(5):
        tr, va = d[(d.group != k) & d.spike120.notna()], d[(d.group == k) & d.spike120.notna()]
        m = XGBClassifier(n_estimators=100, **CLS).fit(tr[feats], tr.spike120.astype(int)); d.loc[va.index, "p_spike"] = m.predict_proba(va[feats])[:, 1]
    return d


def test_preds(test):
    d = test.copy()
    for h in (30, 60):
        m = XGBRegressor(); m.load_model(f"models/forecast_{h}_mean.json"); ok = d[f"y{h}"].notna() & d.cgm_now.notna()
        d[f"p{h}"] = np.nan; d.loc[ok, f"p{h}"] = d.cgm_now[ok] + m.predict(d[ok][feats])
    m = XGBClassifier(); m.load_model("models/spike120.json"); ok = d.spike120.notna(); d["p_spike"] = np.nan; d.loc[ok, "p_spike"] = m.predict_proba(d[ok][feats])[:, 1]
    return d


def adapt(d):
    """adds corrected forecasts and recalibrated spike probability; marks rows after the adaptation window."""
    d = d.sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    d["step"] = d.groupby("recording_id").cumcount(); d["late"] = d.step >= WIN
    keep = d.groupby("recording_id").step.transform("max") >= WIN + 96          # recordings long enough to score
    d = d[keep].reset_index(drop=True)
    for h in (30, 60):
        s = h // 15; y, p = d[f"y{h}"], d[f"p{h}"]; res = y - p
        d[f"m1_{h}"] = p + d.groupby("recording_id").apply(lambda g: pd.Series(g.loc[g.step < WIN, f"y{h}"].sub(g.loc[g.step < WIN, f"p{h}"]).median(), index=g.index)).reset_index(level=0, drop=True).fillna(0)
        known = res.groupby(d.recording_id).shift(s)
        d[f"m2_{h}"] = p + known.groupby(d.recording_id).transform(lambda x: x.rolling(192, min_periods=24).median()).fillna(0)
        m3 = np.full(len(d), np.nan)
        for rid, g in d.groupby("recording_id"):
            w = g[(g.step < WIN) & y.loc[g.index].notna() & p.loc[g.index].notna()]
            if len(w) > 50:
                b, a = np.polyfit(w[f"p{h}"], w[f"y{h}"], 1); b, a = 0.5 + 0.5 * b, 0.5 * a          # halfway to identity
                m3[g.index] = a + b * g[f"p{h}"]
        d[f"m3_{h}"] = np.where(np.isnan(m3), p, m3)
    sp = np.full(len(d), np.nan)
    for rid, g in d.groupby("recording_id"):
        w = g[(g.step < WIN) & g.spike120.notna() & g.p_spike.notna()]
        if len(w) > 50:
            c = 0.5 * (logit(w.spike120.astype(float).mean()) - logit(w.p_spike.mean())); sp[g.index] = sigmoid(logit(g.p_spike) + c)
    d["m5_spike"] = np.where(np.isnan(sp), d.p_spike, sp)
    return d


def mae(e, col, h): ok = e[f"y{h}"].notna() & e[col].notna(); return float((e.loc[ok, col] - e.loc[ok, f"y{h}"]).abs().mean())
def brier(e, col): ok = e.spike120.notna() & e[col].notna(); return float(((e.loc[ok, col] - e.loc[ok, "spike120"].astype(float)) ** 2).mean())
def ece(e, col, bins=10):
    ok = e.spike120.notna() & e[col].notna(); x = e[ok]; q = pd.qcut(x[col], bins, duplicates="drop")
    g = pd.DataFrame({"y": x.spike120.astype(float), "p": x[col], "q": q}).groupby("q", observed=True); return float((g.size() / len(x) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def lows(e, c30, c60):
    L = e[e.low60.notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    y = L.low60.astype(bool); nr = y & ~(y.shift(1, fill_value=False) & (L.recording_id == L.recording_id.shift(1))); ev = np.where(y, nr.cumsum(), 0)
    a = np.nan_to_num(np.minimum(L[c30], L[c60]).values, nan=1e9) <= np.where(L.patient_id.map(heart).fillna(False), 100, 90)
    rid = L.recording_id.values; start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])
    ne = len(set(ev[ev > 0])); ca = int(pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum()) if ne else 0
    return start.sum() / max(1, len(L) / 96), ca, ne


def report(name, d):
    e = d[d.late]; npat = e.patient_id.nunique()
    say(f"{name}: {npat} patients, {len(e)} scored moments (after each recording's first 3 days)")
    say(f"  {'method':34s} {'MAE +30':>8} {'MAE +60':>8} {'spike Brier':>12} {'spike ECE':>10} {'alerts/day':>11} {'lows warned':>12}")
    out = {}
    for lab, c30, c60, cs in (("M0 global model as is", "p30", "p60", "p_spike"), ("M1 fixed bias (first 3 days)", "m1_30", "m1_60", None),
                              ("M2 rolling bias (last 2 days)", "m2_30", "m2_60", None), ("M3 linear recalibration, half-shrunk", "m3_30", "m3_60", None),
                              ("M5 spike level shift", None, None, "m5_spike")):
        r30 = mae(e, c30, 30) if c30 else np.nan; r60 = mae(e, c60, 60) if c60 else np.nan
        br = brier(e, cs) if cs else np.nan; ec = ece(e, cs) if cs else np.nan
        a, ca, ne = lows(e, c30, c60) if c30 else (np.nan, 0, 0)
        out[lab] = (r30, r60); say(f"  {lab:34s} {r30:8.2f} {r60:8.2f} {br:12.4f} {ec:10.3f} {a:11.2f} {f'{ca} of {ne}' if c30 else '-':>12}")
    return out


dev = data[data.group != "test"]; D = adapt(dev_oof(dev[dev.low60.notna() | dev.spike120.notna()]))
say("EXPERIMENT 5: per-patient tuning and recalibration (adapting to the patient's own earlier data)\n")
res = report("DEVELOPMENT (out-of-fold)", D)
base = res["M0 global model as is"]
best = min((k for k in res if k.startswith("M") and k[:2] in ("M1", "M2", "M3")), key=lambda k: res[k][1])
good = res[best][1] <= base[1] - 0.3 and res[best][0] <= base[0] + 0.0
say(f"\nbest on development at +60 min: {best}; clearly better than M0 by the pre-set bar (0.3 mg/dL at +60, not worse at +30): {good}")
if good:
    say("")
    report("LOCKED TEST (scored once)", adapt(test_preds(data[data.group == "test"])))
else:
    say("pre-set bar not met: not shown on the locked test group.")
