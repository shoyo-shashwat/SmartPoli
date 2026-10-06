"""Threshold-based metrics (accuracy, precision, recall, F1) for the final models. REPORT ONLY.

Nothing is retrained and nothing is tuned on the locked test patients:
 - spike120 / large_rise120: the probability cut-off is the one that maximises F1 on the DEVELOPMENT patients'
   out-of-fold predictions; it is then applied to the saved final models on the test patients.
 - lows: the alert thresholds (90 mg/dL, 100 for heart-disease patients) were fixed in advance.
Run:  python extra_metrics.py data/shanghai_t2dm_clean
"""
import sys, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import precision_recall_curve
from baseline import prepare
from forecast import add_targets

meta = json.load(open("models/meta.json"))
CLS = dict(learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=4)
OUT = open("extra_metrics_results.txt", "w", encoding="utf8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n")


def block(name, y, pred):
    y = np.asarray(y, bool); pred = np.asarray(pred, bool)
    tp, fp, fn, tn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum()), int((~y & ~pred).sum())
    pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn); f1 = 2 * pr * rc / max(1e-9, pr + rc)
    say(f"  {name:34s} accuracy {100 * (tp + tn) / len(y):5.1f}%  precision {100 * pr:5.1f}%  recall {100 * rc:5.1f}%  F1 {f1:.3f}  specificity {100 * tn / max(1, tn + fp):5.1f}%   [TP {tp}, FP {fp}, FN {fn}, TN {tn}]")


def main():
    folder = sys.argv[1]
    data, feats = prepare(folder); data = add_targets(data, folder)
    dev, test = data[data["group"] != "test"].copy(), data[data["group"] == "test"].copy()
    for t in ("spike120", "large_rise120"):
        d, e = dev[dev[t].notna()].copy(), test[test[t].notna()]
        n = meta["models"][t]["trees"]; oof = pd.Series(np.nan, index=d.index)
        for k in range(5):
            tr, va = d[d["group"] != k], d[d["group"] == k]
            m = XGBClassifier(n_estimators=n, **CLS).fit(tr[feats], tr[t].astype(int)); oof[va.index] = m.predict_proba(va[feats])[:, 1]
        pr, rc, th = precision_recall_curve(d[t].astype(int), oof); f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(1e-9, pr[:-1] + rc[:-1]); cut = float(th[int(np.argmax(f1))])
        m = XGBClassifier(); m.load_model(f"models/{t}.json"); p = m.predict_proba(e[feats])[:, 1]; y = e[t].astype(bool)
        say(f"\n{t}: test moments {len(e)}, positive rate {100 * y.mean():.1f}%, cut-off {cut:.3f} (chosen on development patients to maximise F1)")
        block("model at that cut-off", y, p >= cut)
        block("model at 0.5", y, p >= 0.5)
        block("always say 'no event'", y, np.zeros(len(y), bool))
        block("always say 'event'", y, np.ones(len(y), bool))
    # ---- lows: fixed thresholds, forecast alert
    st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
    heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
    L = test[test["low60"].notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    for h in (30, 60):
        r = XGBRegressor(); r.load_model(f"models/forecast_{h}_mean.json"); L[f"p{h}"] = L["cgm_now"] + r.predict(L[feats])
    thr = np.where(L.patient_id.map(heart).fillna(False), 100, 90); a = (np.minimum(L.p30, L.p60).values <= thr); y = L["low60"].astype(bool).values
    say(f"\nlows (forecast alert, thresholds 90 / 100 mg/dL fixed in advance): moments {len(L)}, positive moments {int(y.sum())} ({100 * y.mean():.2f}%)")
    block("moment by moment", y, a)
    block("always say 'no low'", y, np.zeros(len(y), bool))
    rid = L.recording_id.values; start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]]); run_id = np.cumsum(start) * a
    runs = pd.DataFrame({"run": run_id[a], "y": y[a]}).groupby("run").y.any()
    newrun = pd.Series(y) & ~(pd.Series(y).shift(1, fill_value=False) & (L.recording_id == L.recording_id.shift(1)))
    ev = np.where(y, newrun.cumsum(), 0); caught = pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any()
    pr_run, rc_ev = runs.mean(), caught.mean(); f1 = 2 * pr_run * rc_ev / (pr_run + rc_ev)
    say(f"  alert by alert: {len(runs)} alerts, {int(runs.sum())} were followed by a real low within an hour -> precision {100 * pr_run:.0f}%; events warned {int(caught.sum())} of {len(caught)} -> recall {100 * rc_ev:.0f}%; F1 {f1:.2f}")
    say("done")


main()
