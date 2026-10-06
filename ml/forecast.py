"""Step 7: forecast line for the dashboard. Predict the sensor value (mg/dL) 15, 30, 60 and 120 minutes ahead,
plus a 10th-90th percentile range for the shaded band. Same features as the warning models (no future info).

Baselines (decision 11): "sugar stays where it is" (persistence), a straight-line extrapolation of the last
30 minutes, and ridge regression. Main model: XGBoost predicting the CHANGE from now, one per horizon.
Scored with the same patient-wise folds: dev patients only; the locked test group is not used.

Run:  python forecast.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from baseline import prepare
from labels import load, STEP_MIN

HORIZONS = {15: 1, 30: 2, 60: 4, 120: 8}      # minutes -> readings ahead


def add_targets(data, folder):
    """Add column y{h}: the raw sensor value h minutes after each moment (NaN when missing)."""
    ts = load(f"{folder}/shanghai_t2dm_timeseries.csv"); out = []
    for rid, g in ts.groupby("recording_id"):
        s = g.set_index("timestamp")["cgm_mgdl"].resample(f"{STEP_MIN}min", origin="start").mean()
        f = pd.DataFrame({"recording_id": rid, "timestamp": s.index})
        for h, k in HORIZONS.items():
            f[f"y{h}"] = s.shift(-k).values
        out.append(f)
    return data.merge(pd.concat(out), on=["recording_id", "timestamp"], how="left")


def xgb(alpha=None):
    kw = dict(objective="reg:quantileerror", quantile_alpha=alpha) if alpha else {}
    return XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", early_stopping_rounds=50, random_state=42, n_jobs=4, **kw)


if __name__ == "__main__":
    folder = sys.argv[1]
    data, feats = prepare(folder); data = add_targets(data, folder)
    rows = []
    for h in HORIZONS:
        d = data[(data["group"] != "test") & data[f"y{h}"].notna() & data["cgm_now"].notna()].copy()
        d["delta"] = d[f"y{h}"] - d["cgm_now"]
        pred = {m: pd.Series(np.nan, index=d.index) for m in ("persistence", "straight line", "ridge", "xgboost", "q10", "q90")}
        for k in range(5):
            tr, es, va = d[~d["group"].isin([k, (k + 1) % 5])], d[d["group"] == (k + 1) % 5], d[d["group"] == k]
            i = va.index
            pred["persistence"][i] = va["cgm_now"]
            pred["straight line"][i] = va["cgm_now"] + (h / 30) * va["cgm_change_30"].fillna(0)
            r = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(tr[feats], tr["delta"])
            pred["ridge"][i] = va["cgm_now"] + r.predict(va[feats])
            for name, a in (("xgboost", None), ("q10", 0.1), ("q90", 0.9)):
                m = xgb(a).fit(tr[feats], tr["delta"], eval_set=[(es[feats], es["delta"])], verbose=False)
                pred[name][i] = va["cgm_now"] + m.predict(va[feats])
        y = d[f"y{h}"]
        for name in ("persistence", "straight line", "ridge", "xgboost"):
            e = (pred[name] - y).abs()
            rows.append(dict(horizon_min=h, model=name, MAE=e.mean(), RMSE=float(np.sqrt(((pred[name] - y) ** 2).mean())),
                             MAE_when_sugar_under_100=e[d["cgm_now"] < 100].mean()))
        lo, hi = np.minimum(pred["q10"], pred["q90"]), np.maximum(pred["q10"], pred["q90"])
        print(f"+{h:3d} min: 10-90% band covers {100 * ((y >= lo) & (y <= hi)).mean():.0f}% of real values (target 80%), mean width {(hi - lo).mean():.0f} mg/dL", flush=True)
    t = pd.DataFrame(rows).round(2)
    print(t.pivot(index="model", columns="horizon_min", values="MAE").loc[["persistence", "straight line", "ridge", "xgboost"]].to_string())
    print("\nMAE when sugar is under 100 mg/dL (the region that matters for lows):")
    print(t.pivot(index="model", columns="horizon_min", values="MAE_when_sugar_under_100").loc[["persistence", "straight line", "ridge", "xgboost"]].to_string())
    t.to_csv("forecast_results.csv", index=False)
