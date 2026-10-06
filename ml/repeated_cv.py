"""Repeated patient-wise CV for one label: re-deal the development patients into 5 folds with
different random seeds and average, so a single lucky/unlucky fold assignment cannot decide the
model choice. The locked test group is never used.

Run:  python repeated_cv.py data/shanghai_t2dm_clean low60
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from baseline import prepare
from compare_models import fit_predict

MODELS = ["rule", "logistic regression", "random forest", "LightGBM", "XGBoost"]

if __name__ == "__main__":
    data, feats = prepare(sys.argv[1]); target = sys.argv[2]; seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    dev = data[(data["group"] != "test") & data[target].notna()].copy()
    has = dev.groupby("patient_id")[target].apply(lambda s: bool(s.astype(bool).any()))
    auc = {m: [] for m in MODELS}; pr = {m: [] for m in MODELS}
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
                p = fit_predict(m, target, tr, es, va, feats); y = va[target].astype(int)
                a.append(roc_auc_score(y, p)); b.append(average_precision_score(y, p))
            auc[m].append(np.mean(a)); pr[m].append(np.mean(b))
        print("seed", seed, {m: round(auc[m][-1], 3) for m in MODELS}, flush=True)
    print(f"\n{target}: mean AUC over {seeds} fold-assignments (sd across assignments) / mean PR AUC")
    for m in MODELS:
        print(f"{m:22s} {np.mean(auc[m]):.3f} ({np.std(auc[m]):.3f})   PR {np.mean(pr[m]):.3f}")
