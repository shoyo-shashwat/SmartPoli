"""Step 3: features (the facts the model may look at) for every moment in the label table.

Rule: a feature at moment t may only use information known at or before t. The label looks
into the future; the features never do. Model INPUTS use the RAW sensor (decision 21); only
the labels use the offset-corrected sensor.

Run (self-check, includes a no-future-leak test):
    python features.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from labels import load, build_labels, STEP_MIN

NO_EVENT_MIN = 1440   # "no meal/dose in the last 24 h" is stored as 1440 minutes
STATIC_COLS = {       # model name -> column in shanghai_t2dm_static.csv
    "age": "Age (years)", "bmi": "BMI (kg/m2)", "diabetes_years": "Duration of diabetes (years)",
    "hba1c": "HbA1c (mmol/mol)", "egfr": "Estimated Glomerular Filtration Rate (ml/min/1.73m2)",
    "sex": "Gender (Female=1, Male=2)", "hypertension": "has_hypertension",
    "on_insulin": "on_insulin", "on_sulfonylurea": "on_sulfonylurea", "on_metformin": "on_metformin",
}


def _minutes_since(moments, events):
    """For each moment, minutes since the latest event at or before it (same recording).
    moments and events both have recording_id + timestamp. Returns a Series aligned to moments."""
    ev = events[["recording_id", "timestamp"]].copy()
    ev["event_time"] = ev["timestamp"]
    m = pd.merge_asof(moments[["recording_id", "timestamp"]].reset_index().sort_values("timestamp"),
                      ev.sort_values("timestamp"), on="timestamp", by="recording_id", direction="backward")
    mins = ((m["timestamp"] - m["event_time"]).dt.total_seconds() / 60).fillna(NO_EVENT_MIN)
    return pd.Series(mins.clip(upper=NO_EVENT_MIN).values, index=m["index"].values).sort_index()


def dynamic_features(ts, moments):
    """moments: DataFrame with recording_id, timestamp. Returns one feature row per moment."""
    out = []
    for rid, g in ts.groupby("recording_id"):
        s = g.set_index("timestamp")["cgm_mgdl"].resample(f"{STEP_MIN}min", origin="start").mean()
        f = pd.DataFrame({"recording_id": rid, "timestamp": s.index, "cgm_now": s.values})
        f["cgm_change_30"] = s.values - s.shift(2).values          # 30 min ago
        f["cgm_change_60"] = s.values - s.shift(4).values
        w = s.rolling(5, min_periods=4)                            # last hour including now
        f["cgm_std_1h"], f["cgm_min_1h"], f["cgm_max_1h"] = w.std().values, w.min().values, w.max().values
        f["hour_sin"] = np.sin(2 * np.pi * s.index.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * s.index.hour / 24)
        out.append(f)
    f = moments[["recording_id", "timestamp"]].merge(pd.concat(out), on=["recording_id", "timestamp"], how="left")

    kinds = {
        "min_since_meal": ts["diet_en"].notna(),
        "min_since_insulin": ts["insulin_sc"].notna() | ts["csii_bolus_iu"].notna() | ts["insulin_iv"].notna(),
        "min_since_oral": ts["oral_agents"].notna(),
    }
    for name, mask in kinds.items():
        f[name] = _minutes_since(f, ts[mask]).values
    # size of the latest insulin dose (units) from "6 IU" text or the pump bolus column
    dose = ts[["recording_id", "timestamp"]].copy()
    dose["iu"] = ts["insulin_sc"].astype(str).str.extract(r"(\d+(?:\.\d+)?)\s*IU")[0].astype(float)
    dose["iu"] = dose["iu"].fillna(pd.to_numeric(ts["csii_bolus_iu"], errors="coerce"))
    dose = dose.dropna(subset=["iu"]).sort_values("timestamp")
    f = pd.merge_asof(f.sort_values("timestamp"), dose.rename(columns={"timestamp": "dose_time"}), left_on="timestamp",
                      right_on="dose_time", by="recording_id", direction="backward")
    f["last_insulin_iu"] = np.where(f["min_since_insulin"] < NO_EVENT_MIN, f["iu"], 0.0)
    return f.drop(columns=["iu", "dose_time"]).sort_values(["recording_id", "timestamp"]).reset_index(drop=True)


def build_features(ts, static, labels):
    f = dynamic_features(ts, labels)
    s = static[["recording_id"] + list(STATIC_COLS.values())].rename(columns={v: k for k, v in STATIC_COLS.items()})
    return f.merge(s, on="recording_id", how="left")


if __name__ == "__main__":
    folder = sys.argv[1]
    ts = load(f"{folder}/shanghai_t2dm_timeseries.csv")
    static = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv")
    labels = build_labels(ts)
    X = build_features(ts, static, labels)
    assert len(X) == len(labels)
    print("moments", len(X), "features", X.shape[1] - 2)
    print(X.isna().mean().round(3).to_string())
    # leak test: cut one recording at its midpoint; features at earlier moments must not change
    rid = labels["recording_id"].value_counts().index[0]
    cut = labels.loc[labels.recording_id == rid, "timestamp"].iloc[len(labels[labels.recording_id == rid]) // 2]
    ts_cut = ts[~((ts.recording_id == rid) & (ts.timestamp > cut))]
    early = labels[(labels.recording_id == rid) & (labels.timestamp <= cut)]
    a = build_features(ts, static, early).drop(columns=["recording_id", "timestamp"]).reset_index(drop=True)
    b = build_features(ts_cut, static, early).drop(columns=["recording_id", "timestamp"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    print("leak test passed: features never use readings after the moment")
