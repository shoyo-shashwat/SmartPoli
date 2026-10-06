"""Step 6: LightGBM (decision 11). Same patient-wise 5-fold CV as baseline.py, dev patients only.

Early stopping needs a "practice exam" the model is not trained on. To keep the score honest,
each outer fold k is scored on patients the model never saw, and the early-stopping check uses a
DIFFERENT fold (k+1), so fold k never influences when training stops.

Run:  python model_lgbm.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from baseline import prepare, TARGETS, RULE_SCORE

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=42)


def cv_lgbm(data, feats, target):
    d = data[(data["group"] != "test") & data[target].notna()]
    oof = pd.Series(np.nan, index=d.index)
    rounds = []
    for k in range(5):
        stop_fold = (k + 1) % 5
        tr = d[~d["group"].isin([k, stop_fold])]
        es = d[d["group"] == stop_fold]
        va = d[d["group"] == k]
        m = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr[target].astype(int)), num_boost_round=1000,
                      valid_sets=[lgb.Dataset(es[feats], es[target].astype(int))],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va.index] = m.predict(va[feats], num_iteration=m.best_iteration)
        rounds.append(m.best_iteration)
    y = d[target].astype(int)
    per_fold = [roc_auc_score(y[d["group"] == k], oof[d["group"] == k]) for k in range(5)]
    return y.mean(), roc_auc_score(y, oof), average_precision_score(y, oof), rounds, per_fold


def per_fold_baselines(data, feats, target):
    """Mean AUC over the 5 folds for the rule and logistic regression, so all three models are
    compared the same way (pooling folds mixes score scales and can mislead for lows)."""
    d = data[(data["group"] != "test") & data[target].notna()]
    rule, lr = [], []
    for k in range(5):
        tr, va = d[d["group"] != k], d[d["group"] == k]
        y = va[target].astype(int)
        s = RULE_SCORE[target](va); ok = s.notna()
        rule.append(roc_auc_score(y[ok], s[ok]))
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        m.fit(tr[feats], tr[target].astype(int)); lr.append(roc_auc_score(y, m.predict_proba(va[feats])[:, 1]))
    return np.mean(rule), np.mean(lr)


if __name__ == "__main__":
    data, feats = prepare(sys.argv[1])
    print(f"{'label':14s} {'rate':>6s} | {'AUC':>6s} {'PR AUC':>7s} | trees per fold | AUC per fold")
    for t in TARGETS:
        rate, auc, pr, rounds, pf = cv_lgbm(data, feats, t)
        print(f"{t:14s} {rate:6.3f} | {auc:6.3f} {pr:7.3f} | {rounds} | {[round(x, 2) for x in pf]}")
        r, l = per_fold_baselines(data, feats, t)
        print(f"{'':14s} mean AUC over folds: rule {r:.3f}  logreg {l:.3f}  lightgbm {np.mean(pf):.3f}")
