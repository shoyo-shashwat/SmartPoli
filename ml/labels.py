"""Step 2: build the labels (the answers the model learns to predict) for ShanghaiT2DM.

One row = one moment (one 15-minute sensor reading). A label says what happens AFTER that
moment: a low, a spike, or a large rise. Rules: decisions 21, 33, 39, 40, 42 in the briefing.
Must reproduce scripts/label_reference.py exactly (see EXPECTED below).

Run:  python labels.py data/shanghai_t2dm_clean/shanghai_t2dm_timeseries.csv
"""
import sys
import numpy as np
import pandas as pd

STEP_MIN = 15                    # the sensor reads every 15 minutes
LOW, SPIKE = 70, 180             # mg/dL thresholds
LOW_AHEAD, SPIKE_AHEAD = 4, 8    # readings ahead: 60 min for lows, 120 min for spikes/rises
FIRST_HOURS_NO_LOW = 12          # decision 33: sensor is unreliable in its first 12 h
# (moments, positives, patients) for the three main labels, from label_reference.py
EXPECTED = {"low60": (106100, 378, 24), "spike120": (84716, 13795, 99), "large_rise120": (92877, 13977, 99)}


def load(path):
    ts = pd.read_csv(path, low_memory=False,
                     usecols=["patient_id", "recording_id", "timestamp", "cgm_mgdl", "cbg_mgdl",
                              "diet_en", "insulin_sc", "oral_agents", "csii_bolus_iu", "insulin_iv"])
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], format="mixed")
    ts.loc[ts["cbg_mgdl"] > 600, "cbg_mgdl"] = np.nan   # data fix: 2,044.8 in recording 2044 is the patient ID
    return ts.sort_values(["recording_id", "timestamp"])


def sensor_offsets(ts):
    """Per recording: how far the sensor reads from finger-pricks (median of sensor - finger-prick
    within 15 min). Fewer than 5 pairs -> use the dataset median. Labels use sensor - offset."""
    cgm = ts.dropna(subset=["cgm_mgdl"])[["recording_id", "timestamp", "cgm_mgdl"]]
    fp = ts.dropna(subset=["cbg_mgdl"])[["recording_id", "timestamp", "cbg_mgdl"]]
    pairs = pd.merge_asof(fp.sort_values("timestamp"), cgm.sort_values("timestamp"), on="timestamp",
                          by="recording_id", direction="nearest", tolerance=pd.Timedelta("15min")).dropna()
    pairs["d"] = pairs["cgm_mgdl"] - pairs["cbg_mgdl"]
    per = pairs.groupby("recording_id")["d"].agg(["size", "median"])
    default = pairs["d"].median()
    return {r: (per.loc[r, "median"] if r in per.index and per.loc[r, "size"] >= 5 else default)
            for r in ts["recording_id"].unique()}


def episode_starts(v, below, thr):
    """True at the FIRST reading of every run of 2+ readings below (or above) thr.
    Counting only the start means one long low is one event, not many."""
    x = (v < thr) if below else (v > thr)
    run2 = np.zeros(len(v), bool)
    run2[:-1] = x[:-1] & x[1:]                # this and the next reading both beyond thr
    return run2 & ~np.r_[False, x[:-1]]       # ...and the previous one was not (so it starts here)


def build_labels(ts):
    """Return one row per moment with columns low60, spike120, large_rise120 (True/False, or NaN
    when the label cannot be decided, e.g. readings missing ahead). Each recording is placed on its
    own 15-minute grid and gaps stay empty, never filled."""
    offset = sensor_offsets(ts)
    rows = []
    for rid, g in ts.groupby("recording_id"):
        s = g.set_index("timestamp")["cgm_mgdl"].resample(f"{STEP_MIN}min", origin="start").mean()
        v = s.values - offset[rid]            # corrected values: used for labels only
        n = len(v)
        lo, hi = episode_starts(v, True, LOW), episode_starts(v, False, SPIKE)
        past = pd.Series(v).shift(1)          # shift(1): a moment must not be part of its own bar
        usual = past.rolling(96, min_periods=72).median().values   # previous 24 h
        p90 = past.rolling(96, min_periods=72).quantile(0.9).values
        for i in range(n):
            if np.isnan(v[i]):
                continue
            r = dict(recording_id=rid, patient_id=int(rid[:4]), timestamp=s.index[i], hours=i * STEP_MIN / 60)
            if i + 5 < n and not np.isnan(v[i + 1:i + 6]).any() and v[i] >= LOW:
                r["low60"] = bool(lo[i + 1:i + 1 + LOW_AHEAD].any())
            if i + 9 < n and not np.isnan(v[i + 1:i + 10]).any() and v[i] <= SPIKE:
                r["spike120"] = bool(hi[i + 1:i + 1 + SPIKE_AHEAD].any())
            if i + 8 < n and not np.isnan(v[i + 1:i + 9]).any() and not np.isnan(usual[i]):
                bar = max(p90[i], usual[i] + 30)   # personal bar: own 90th percentile or usual + 30
                if v[i] < bar:
                    r["large_rise120"] = bool((v[i + 1:i + 1 + SPIKE_AHEAD] >= bar).any())
            rows.append(r)
    d = pd.DataFrame(rows)
    d.loc[d["hours"] < FIRST_HOURS_NO_LOW, "low60"] = np.nan   # decision 33
    return d


def summary(d):
    out = {}
    for col in EXPECTED:
        y = d[col].dropna().astype(bool)
        out[col] = (len(y), int(y.sum()), d.loc[y[y].index, "patient_id"].nunique())
    return out


if __name__ == "__main__":
    labels = build_labels(load(sys.argv[1]))
    got = summary(labels)
    for col, exp in EXPECTED.items():
        print(f"{col:14s} got {got[col]}  expected {exp}  {'OK' if got[col] == exp else 'MISMATCH'}")
    assert got == EXPECTED, "labels do not match the reference counts"
