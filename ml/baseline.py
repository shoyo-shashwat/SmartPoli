"""Step 5: baselines to beat (decision 11): a simple rule, then logistic regression.
Scored with 5-fold patient-wise cross-validation on the development patients only.
The locked test group (splits.csv, group 'test') is NOT used here.

Metrics: ROC AUC (can it rank risky moments above safe ones; 0.5 = coin flip) and
PR AUC / average precision (how well it finds the rare events; compare with the positive rate).

Run:  python baseline.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from labels import load, build_labels
from features import build_features
from split import make_split

TARGETS = ["low60", "spike120", "large_rise120"]
RULE_SCORE = {"low60": lambda X: -X["cgm_now"], "spike120": lambda X: X["cgm_now"],
              "large_rise120": lambda X: X["cgm_change_60"].fillna(0)}   # simplest "sugar already near/rising" rules


def prepare(folder):
    ts = load(f"{folder}/shanghai_t2dm_timeseries.csv")
    static = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv")
    labels = build_labels(ts)
    X = build_features(ts, static, labels)
    data = labels.merge(X, on=["recording_id", "timestamp"])
    sp = make_split(labels)
    return data.merge(sp[["patient_id", "group"]], on="patient_id"), [c for c in X.columns if c not in ("recording_id", "timestamp")]


def cross_validate(data, feats, target, model_fn):
    d = data[(data["group"] != "test") & data[target].notna()]
    oof = pd.Series(np.nan, index=d.index)
    for k in range(5):
        tr, va = d[d["group"] != k], d[d["group"] == k]
        m = model_fn(); m.fit(tr[feats], tr[target].astype(int))
        oof[va.index] = m.predict_proba(va[feats])[:, 1]
    y = d[target].astype(int)
    return y.mean(), roc_auc_score(y, oof), average_precision_score(y, oof)


if __name__ == "__main__":
    data, feats = prepare(sys.argv[1])
    logreg = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(max_iter=1000, class_weight="balanced"))
    print(f"{'label':14s} {'positive rate':>13s} | {'rule AUC':>8s} {'rule PR':>8s} | {'logreg AUC':>10s} {'logreg PR':>9s}")
    for t in TARGETS:
        d = data[(data["group"] != "test") & data[t].notna()]
        y, s = d[t].astype(int), RULE_SCORE[t](d)
        ok = s.notna()
        rate, auc, pr = cross_validate(data, feats, t, logreg)
        print(f"{t:14s} {rate:13.3f} | {roc_auc_score(y[ok], s[ok]):8.3f} {average_precision_score(y[ok], s[ok]):8.3f} | {auc:10.3f} {pr:9.3f}")
