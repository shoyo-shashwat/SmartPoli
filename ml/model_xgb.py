"""LightGBM vs XGBoost, same patient-wise folds and the same honest early stopping as model_lgbm.py.
Run:  python model_xgb.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from baseline import prepare, TARGETS


def cv_xgb(data, feats, target):
    d = data[(data["group"] != "test") & data[target].notna()]
    oof = pd.Series(np.nan, index=d.index); rounds = []
    for k in range(5):
        es_fold = (k + 1) % 5
        tr, es, va = d[~d["group"].isin([k, es_fold])], d[d["group"] == es_fold], d[d["group"] == k]
        m = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8,
                          colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", eval_metric="logloss",
                          early_stopping_rounds=50, random_state=42, n_jobs=4)
        m.fit(tr[feats], tr[target].astype(int), eval_set=[(es[feats], es[target].astype(int))], verbose=False)
        oof[va.index] = m.predict_proba(va[feats])[:, 1]; rounds.append(m.best_iteration)
    y = d[target].astype(int)
    pf = [roc_auc_score(y[d["group"] == k], oof[d["group"] == k]) for k in range(5)]
    pr = [average_precision_score(y[d["group"] == k], oof[d["group"] == k]) for k in range(5)]
    return np.mean(pf), np.mean(pr), rounds


if __name__ == "__main__":
    data, feats = prepare(sys.argv[1])
    for t in TARGETS:
        auc, pr, rounds = cv_xgb(data, feats, t)
        print(f"{t:14s} xgboost mean AUC {auc:.3f}  mean PR AUC {pr:.3f}  trees per fold {rounds}")
