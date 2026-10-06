"""Constrained boosters for lows. Medical common sense is built in: with everything else equal, risk of a
low can only go UP when current sugar is lower, the last hour's minimum is lower, or sugar is falling faster.
Smaller trees and bigger leaves stop the model chasing the few hundred low events.

Run:  python constrained_lows.py data/shanghai_t2dm_clean 10
"""
import sys
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from baseline import prepare
from compare_models import fit_predict
from model_lgbm import PARAMS

DOWN = {"cgm_now": -1, "cgm_min_1h": -1, "cgm_change_30": -1, "cgm_change_60": -1}   # higher value -> lower risk


def fit_constrained(kind, target, tr, es, va, feats):
    mono = [DOWN.get(f, 0) for f in feats]
    y, ye = tr[target].astype(int), es[target].astype(int)
    if kind == "XGB constrained":
        m = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                          monotone_constraints=tuple(mono), tree_method="hist", eval_metric="logloss", early_stopping_rounds=50,
                          random_state=42, n_jobs=4)
        m.fit(tr[feats], y, eval_set=[(es[feats], ye)], verbose=False)
        return m.predict_proba(va[feats])[:, 1]
    p = {**PARAMS, "num_leaves": 7, "min_child_samples": 100, "monotone_constraints": mono}
    b = lgb.train(p, lgb.Dataset(tr[feats], y), 1000, valid_sets=[lgb.Dataset(es[feats], ye)], callbacks=[lgb.early_stopping(50, verbose=False)])
    return b.predict(va[feats], num_iteration=b.best_iteration)


MODELS = ["rule", "XGBoost", "XGB constrained", "LGBM constrained"]
if __name__ == "__main__":
    data, feats = prepare(sys.argv[1]); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 10; t = "low60"
    dev = data[(data["group"] != "test") & data[t].notna()].copy()
    has = dev.groupby("patient_id")[t].apply(lambda s: bool(s.astype(bool).any()))
    auc = {m: [] for m in MODELS}; pr = {m: [] for m in MODELS}; worst = {m: [] for m in MODELS}
    for seed in range(seeds):
        rng = np.random.default_rng(seed); fold = {}
        for flag in (True, False):
            ids = list(has.index[has == flag]); rng.shuffle(ids)
            for i, p in enumerate(ids): fold[p] = i % 5
        dev["f"] = dev["patient_id"].map(fold)
        for m in MODELS:
            a, b = [], []
            for k in range(5):
                tr, es, va = dev[~dev["f"].isin([k, (k + 1) % 5])], dev[dev["f"] == (k + 1) % 5], dev[dev["f"] == k]
                p = fit_constrained(m, t, tr, es, va, feats) if "constrained" in m else fit_predict(m, t, tr, es, va, feats)
                y = va[t].astype(int); a.append(roc_auc_score(y, p)); b.append(average_precision_score(y, p))
            auc[m].append(np.mean(a)); pr[m].append(np.mean(b)); worst[m].append(min(a))
    print(f"low60, {seeds} fold assignments: mean AUC (sd across assignments), mean PR AUC, mean worst-fold AUC")
    for m in MODELS:
        print(f"{m:18s} {np.mean(auc[m]):.3f} ({np.std(auc[m]):.3f})  PR {np.mean(pr[m]):.3f}  worst fold {np.mean(worst[m]):.3f}")
