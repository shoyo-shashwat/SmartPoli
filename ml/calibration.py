"""Calibration check: when the model says 30%, does it happen about 30% of the time?

Out-of-fold predictions from the chosen model (XGBoost; monotone-constrained for lows) are corrected
two ways, Platt scaling (a 2-number logistic curve) and isotonic regression (a monotone staircase).
Each fold is corrected by a calibrator fitted only on the OTHER folds' predictions, so the check is honest.
Locked test patients are not used.

Run:  python calibration.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from baseline import prepare
from constrained_lows import fit_constrained
from compare_models import fit_predict

TARGETS = {"low60": "XGB constrained", "spike120": "XGBoost", "large_rise120": "XGBoost"}


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(y) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def oof(data, feats, t, kind):
    d = data[(data["group"] != "test") & data[t].notna()].copy(); d["raw"] = np.nan
    for k in range(5):
        tr, es, va = d[~d["group"].isin([k, (k + 1) % 5])], d[d["group"] == (k + 1) % 5], d[d["group"] == k]
        d.loc[va.index, "raw"] = fit_constrained(kind, t, tr, es, va, feats) if "constrained" in kind else fit_predict(kind, t, tr, es, va, feats)
    return d


if __name__ == "__main__":
    data, feats = prepare(sys.argv[1])
    print(f"{'label':14s} {'method':10s} {'Brier':>8s} {'log loss':>9s} {'ECE':>7s}   (rate {{}})")
    for t, kind in TARGETS.items():
        d = oof(data, feats, t, kind); y = d[t].astype(int).values; d["y"] = y
        out = {"raw": d["raw"].values.copy(), "platt": np.zeros(len(d)), "isotonic": np.zeros(len(d))}
        for k in range(5):
            fit, va = d[d["group"] != k], d["group"] == k
            z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
            pl = LogisticRegression(C=1e6).fit(z(fit["raw"].values), fit["y"]); out["platt"][va.values] = pl.predict_proba(z(d.loc[va, "raw"].values))[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(fit["raw"].values, fit["y"]); out["isotonic"][va.values] = iso.predict(d.loc[va, "raw"].values)
        base = np.full(len(y), y.mean())
        print(f"{t:14s} {'base rate':10s} {brier_score_loss(y, base):8.4f} {log_loss(y, base):9.4f} {'':>7s}   rate {y.mean():.4f}")
        for m, p in out.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            print(f"{'':14s} {m:10s} {brier_score_loss(y, p):8.4f} {log_loss(y, p):9.4f} {ece(y, p):7.4f}")
