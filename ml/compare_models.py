"""Model-selection analysis: the same patient-wise folds, one table, several algorithm families.

Families: simple rule | logistic regression (linear) | random forest and extra trees (bagged trees) |
LightGBM and XGBoost (boosted trees) | small neural network (MLP).
Everything is scored per fold on patients the model never saw; we report the mean and the spread.
Boosters use early stopping on a separate fold (k+1); the MLP stops on a random 10% of its own training rows.

Run:  python compare_models.py data/shanghai_t2dm_clean      (takes several minutes)
"""
import sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from baseline import prepare, TARGETS, RULE_SCORE
from model_lgbm import PARAMS


def fit_predict(name, target, tr, es, va, feats):
    X, y = tr[feats], tr[target].astype(int)
    if name == "rule":
        return RULE_SCORE[target](va).fillna(0).values
    if name == "logistic regression":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    elif name == "random forest":
        m = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(150, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    elif name == "extra trees":
        m = make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(150, max_depth=12, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=42))
    elif name == "neural network (MLP)":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          MLPClassifier((32, 16), alpha=1e-3, early_stopping=True, n_iter_no_change=10, max_iter=300, random_state=42))
    elif name == "LightGBM":
        b = lgb.train(PARAMS, lgb.Dataset(X, y), 1000, valid_sets=[lgb.Dataset(es[feats], es[target].astype(int))],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        return b.predict(va[feats], num_iteration=b.best_iteration)
    elif name == "XGBoost":
        m = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          tree_method="hist", eval_metric="logloss", early_stopping_rounds=50, random_state=42, n_jobs=4)
        m.fit(X, y, eval_set=[(es[feats], es[target].astype(int))], verbose=False)
        return m.predict_proba(va[feats])[:, 1]
    m.fit(X, y)
    return m.predict_proba(va[feats])[:, 1]


MODELS = ["rule", "logistic regression", "random forest", "extra trees", "LightGBM", "XGBoost", "neural network (MLP)"]

if __name__ == "__main__":
    data, feats = prepare(sys.argv[1])
    rows = []
    for t in TARGETS:
        d = data[(data["group"] != "test") & data[t].notna()]
        for name in MODELS:
            aucs, prs, secs = [], [], 0.0
            for k in range(5):
                tr, es, va = d[~d["group"].isin([k, (k + 1) % 5])], d[d["group"] == (k + 1) % 5], d[d["group"] == k]
                t0 = time.time(); p = fit_predict(name, t, tr, es, va, feats); secs += time.time() - t0
                y = va[t].astype(int)
                aucs.append(roc_auc_score(y, p)); prs.append(average_precision_score(y, p))
            rows.append(dict(label=t, model=name, auc=np.mean(aucs), auc_std=np.std(aucs), pr_auc=np.mean(prs),
                             worst_fold_auc=min(aucs), seconds=secs / 5))
            print(f"{t:14s} {name:22s} AUC {np.mean(aucs):.3f} (sd {np.std(aucs):.3f}, worst {min(aucs):.3f})  PR {np.mean(prs):.3f}  {secs / 5:5.1f}s/fold", flush=True)
    pd.DataFrame(rows).round(4).to_csv("model_comparison.csv", index=False)
