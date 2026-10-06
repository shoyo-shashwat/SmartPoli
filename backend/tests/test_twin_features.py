"""The app's plain-Python features must equal the training pipeline's (ml/features.py) on a real demo recording.
Skipped when the demo data / ml data are not on this machine (they are not in the public repo)."""
import json
import math
import os
import sys
from datetime import datetime

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
ML = os.path.join(os.path.dirname(os.path.dirname(HERE)), "ml")
DEMO = os.path.join(os.path.dirname(HERE), "twin_demo_data", "demo_a.json")
DATA = os.path.join(ML, "data", "shanghai_t2dm_clean")

pytestmark = pytest.mark.skipif(not (os.path.exists(DEMO) and os.path.exists(DATA)), reason="demo data / ml data not on this machine")


def _demo():
    d = json.load(open(DEMO))
    grid = {datetime.fromisoformat(t): v for t, v in d["readings"]}
    events = [(datetime.fromisoformat(t), k, iu, txt) for t, k, txt, iu in d["events"]]
    return d, grid, events


def test_features_match_training_pipeline():
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, ML)
    from baseline import prepare
    from twin_features import build_features
    d, grid, events = _demo()
    data, feats = prepare(DATA)
    sub = data[data.recording_id == "2094_0_20211109"]
    names = json.load(open(os.path.join(os.path.dirname(HERE), "twin_models", "meta.json")))["features"]
    assert names == feats
    sample = sub.iloc[::37]                                    # about 35 moments across the recording
    assert len(sample) > 20
    for _, r in sample.iterrows():
        _, row = build_features(names, grid, events, d["profile"], r["timestamp"].to_pydatetime())
        for n, mine in zip(names, row):
            theirs = float(r[n])
            assert (math.isnan(mine) and math.isnan(theirs)) or abs(mine - theirs) < 1e-6 * max(1, abs(theirs)), (r["timestamp"], n, mine, theirs)
