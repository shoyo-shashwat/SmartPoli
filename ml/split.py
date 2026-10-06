"""Step 4: patient-wise split (decision 41). Same person never lands in both train and test.

About 20 patients are locked away as the FINAL test group (touched once, at the very end).
The other patients go into 5 folds for development (cross-validation). Patients with at least
one low are spread evenly, because lows are rare (24 of 100 patients).
Result is saved to splits.csv (patient ids only, no data) so everyone uses the same split.

Run:  python split.py data/shanghai_t2dm_clean/shanghai_t2dm_timeseries.csv
"""
import sys
import numpy as np
import pandas as pd
from labels import load, build_labels

SEED, N_TEST, N_FOLDS = 42, 20, 5
# Demo patients for the app (decision 41: locked away in addition to the random test group; never trained on).
DEMO_SHANGHAI = (2094,)     # recording 2094_0_20211109: coronary heart disease, insulin, 18 lows
DEMO_CGMACROS = (38,)       # CGMacros participant 38: type 2 diabetes, full heart-rate coverage (not in this split)


def make_split(labels, demo_patients=DEMO_SHANGHAI):
    """Return DataFrame patient_id, has_low, group ('test' or fold number 0-4).
    demo_patients (for the app demo) are always put in the test group."""
    low = labels.groupby("patient_id")["low60"].apply(lambda s: bool(s.dropna().astype(bool).any()))
    rng = np.random.default_rng(SEED)
    test = set(demo_patients)
    need = N_TEST                       # random test patients, on top of the demo patients
    # share of low-patients in the test group follows their share overall
    n_low_test = round(need * low.mean())
    for has_low, k in ((True, n_low_test), (False, need - n_low_test)):
        pool = [p for p in low.index[low == has_low] if p not in test]
        test |= set(rng.choice(pool, size=k, replace=False))
    group = pd.Series("test", index=low.index, dtype=object)
    for has_low in (True, False):                       # deal dev patients into folds, lows first
        dev = [p for p in low.index[low == has_low] if p not in test]
        rng.shuffle(dev)
        for i, p in enumerate(dev):
            group[p] = i % N_FOLDS
    group[list(test)] = "test"
    return pd.DataFrame({"patient_id": low.index, "has_low": low.values, "group": group.values})


if __name__ == "__main__":
    labels = build_labels(load(sys.argv[1]))
    sp = make_split(labels)
    sp.to_csv("splits.csv", index=False)
    print(sp.groupby("group")["has_low"].agg(patients="size", with_lows="sum").to_string())
    assert sp["patient_id"].is_unique and len(sp) == 100
