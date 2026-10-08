"""Experiment: weight the forecast training rows where sugar is already low (< 120 mg/dL) so the +30 / +60 min forecast
is more accurate in the range that triggers the low alert. Alert rule unchanged: lowest of +30 and +60 forecast
<= 90 mg/dL (100 with heart disease), shifted by an `offset` when stated.

Plan fixed before looking (see EXPERIMENT_LOG.md, experiment 3):
  weights tried: 1 (current), 3, 6, 10 for rows with sugar now < 120.
  Chosen on the 79 development patients (out-of-fold) as the weight needing the FEWEST alerts/day while still warning
  at least 64 of the 66 development low events; the threshold offset that achieves this is chosen on development too.
  Then the chosen weight is trained on all development patients and scored ONCE on the locked test group,
  next to the current model (models/forecast_*_mean.json).
Run from ml/:  python lows_weighted_forecast.py data/shanghai_t2dm_clean
"""
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from baseline import prepare
from forecast import add_targets, xgb

folder = sys.argv[1]
OUT = open("lows_weighted_results.txt", "w", encoding="utf8")


def say(s=""):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


data, feats = prepare(folder); data = add_targets(data, folder)
st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
LOWNOW, WEIGHTS, KEEP = 120, (1, 3, 6, 10), 64


def prep(df):
    L = df[df["low60"].notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    L["y"] = L["low60"].astype(bool)
    newrun = L.y & ~(L.y.shift(1, fill_value=False) & (L.recording_id == L.recording_id.shift(1)))
    L["ev"] = np.where(L.y, newrun.cumsum(), 0); L["heart"] = L.patient_id.map(heart).fillna(False)
    return L


def score(L, a):
    rid = L.recording_id.values; a = np.asarray(a, bool)
    start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])
    run = np.cumsum(start) * a; n = int(start.sum())
    ev = L.ev.values; ne = len(set(ev[ev > 0])); caught = int(pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum())
    hit = len(set(run[a & L.y.values])) if n else 0
    return n / (len(L) / 96), caught, ne, (hit / n if n else 0.0)


def alerts(L, offset=0):
    thr = np.where(L.heart, 100, 90) + offset
    return np.nan_to_num(np.minimum(L.p30, L.p60).values, nan=1e9) <= thr


def mae_low(L, h):
    ok = L[f"y{h}"].notna() & L.p30.notna() & L.p60.notna() & (L.cgm_now < 100)
    return float((L[f"p{h}"] - L[f"y{h}"]).abs()[ok].mean())


def fit_oof(dd, W):
    """out-of-fold +30 / +60 forecasts with weight W on rows with sugar now < LOWNOW; also the best tree counts."""
    dd = dd.copy(); iters = {30: [], 60: []}
    for h in (30, 60):
        dd["delta"] = dd[f"y{h}"] - dd.cgm_now; dd[f"p{h}"] = np.nan
        for k in range(5):
            tr, es, va = dd[~dd.group.isin([k, (k + 1) % 5])], dd[dd.group == (k + 1) % 5], dd[dd.group == k]
            trn, esn = tr[tr.delta.notna()], es[es.delta.notna()]
            w = np.where(trn.cgm_now < LOWNOW, W, 1.0)
            m = xgb().fit(trn[feats], trn.delta, sample_weight=w, eval_set=[(esn[feats], esn.delta)], verbose=False)
            iters[h].append(m.best_iteration + 1)
            ok = (va.delta.notna() & va.cgm_now.notna()).values
            dd.loc[va.index[ok], f"p{h}"] = va.cgm_now[ok] + m.predict(va[ok][feats])
    return dd, {h: int(np.median(v)) for h, v in iters.items()}


def min_alerts_keeping(L, keep):
    """smallest alerts/day over threshold offsets that still warns >= keep events; returns (offset, alerts/day, warned, precision)."""
    best = None
    for off in range(-20, 41, 2):
        r, c, ne, pr = score(L, alerts(L, off))
        if c >= keep and (best is None or r < best[1]): best = (off, r, c, pr)
    return best


dev = data[data.group != "test"]; dd0 = dev[dev.low60.notna()]
say("EXPERIMENT 3: weighting low-sugar rows in the forecast (training rows with sugar now < 120 mg/dL)")
say(f"=== development patients ({dd0.patient_id.nunique()}), out-of-fold; rule 90/100; offset 0 = unchanged rule ===")
res = {}
for W in WEIGHTS:
    d, trees = fit_oof(dd0, W); D = prep(d); res[W] = (D, trees)
    r, c, ne, pr = score(D, alerts(D))
    b = min_alerts_keeping(D, KEEP)
    say(f"  weight {W:2d}: MAE sugar<100  +30 {mae_low(D, 30):.2f}  +60 {mae_low(D, 60):.2f} | rule as is: {r:.2f} alerts/day, warned {c}/{ne}, precision {100 * pr:.1f}% | "
        f"to warn >= {KEEP}: offset {b[0]:+d} -> {b[1]:.2f} alerts/day, warned {b[2]}, precision {100 * b[3]:.1f}%")
    res[W] += (b,)
chosen = min(WEIGHTS, key=lambda W: res[W][2][1])
off = res[chosen][2][0]
say(f"chosen on development: weight {chosen}, threshold offset {off:+d} mg/dL")

say(f"\n=== locked test group, scored once ===")
test = data[data.group == "test"].copy()
cur = test.copy()
for h in (30, 60):
    m = XGBRegressor(); m.load_model(f"models/forecast_{h}_mean.json")
    ok = cur[f"y{h}"].notna() & cur.cgm_now.notna(); cur[f"p{h}"] = np.nan; cur.loc[ok, f"p{h}"] = cur.cgm_now[ok] + m.predict(cur[ok][feats])
T0 = prep(cur)
new = test.copy(); trees = res[chosen][1]
for h in (30, 60):
    d = dev[dev[f"y{h}"].notna() & dev.cgm_now.notna()]
    kw = dict(xgb().get_params()); kw.pop("early_stopping_rounds"); kw["n_estimators"] = trees[h]
    m = XGBRegressor(**kw).fit(d[feats], d[f"y{h}"] - d.cgm_now, sample_weight=np.where(d.cgm_now < LOWNOW, chosen, 1.0))
    ok = new[f"y{h}"].notna() & new.cgm_now.notna(); new[f"p{h}"] = np.nan; new.loc[ok, f"p{h}"] = new.cgm_now[ok] + m.predict(new[ok][feats])
T1 = prep(new)
for name, T, o in (("current model (weight 1)", T0, 0), (f"weight {chosen}, rule as is", T1, 0), (f"weight {chosen}, offset {off:+d} chosen on development", T1, off)):
    r, c, ne, pr = score(T, alerts(T, o))
    say(f"  {name:52s} MAE sugar<100 +30 {mae_low(T, 30):.2f} +60 {mae_low(T, 60):.2f} | {r:.2f} alerts/day, warned {c}/{ne}, precision {100 * pr:.1f}%")
