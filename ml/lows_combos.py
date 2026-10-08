"""Experiment 4: every combination of the three ideas for the low alert, numbered.
  Option 1  risk gate     : alert only for patients at risk (on insulin or a sulfonylurea; and/or sugar went low
                            in the last 24 h). Known before the moment, no future data.
  Option 2  confirmation  : alert only when the forecast is low on k readings in a row (k = 2 or 3), optionally
                            also needing the sugar to be falling (30-min change < 0).
  Option 3  worklist view : instead of a minute-by-minute alert, flag the patient-DAY ("review this patient today");
                            judged per patient-day. (A retrospective end-of-day list, not a pager.)
Base rule: forecast lowest of +30 and +60 min <= 90 mg/dL (100 with heart disease), plus a threshold shift (offset).

Plan fixed before looking: for each combination, settings chosen on the 79 development patients (out-of-fold forecasts) =
the lowest burden (alerts per patient-day, or share of patient-days flagged) that still warns at least 60 of the 66
development low events. All combinations are then shown on the locked test group once, with their development-chosen
settings. The winner is the one best on DEVELOPMENT; test numbers are the honest check, not used to choose.
Run from ml/:  python lows_combos.py data/shanghai_t2dm_clean
"""
import sys
import itertools
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from baseline import prepare
from forecast import add_targets, xgb

folder = sys.argv[1]
OUT = open("lows_combos_results.txt", "w", encoding="utf8")


def say(s=""):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


data, feats = prepare(folder); data = add_targets(data, folder)
st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
drug = st.set_index("recording_id")[["on_insulin", "on_sulfonylurea"]].astype(bool).any(axis=1)
data = data.sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
data["min24"] = data.groupby("recording_id").cgm_now.transform(lambda x: x.rolling(96, min_periods=1).min())   # lowest sugar in the last 24 h
KEEP = 60

GATES = [("none", None, None)] + [("drug", True, None)] + [(f"recent<={t}", False, t) for t in (80, 90, 100)] + [(f"drug|recent<={t}", True, t) for t in (80, 90, 100)]
CONF = [(1, False), (2, False), (3, False), (1, True), (2, True), (3, True)]
OFFS = (-5, 0, 5, 10)


def prep(df):
    L = df[df["low60"].notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    L["y"] = L["low60"].astype(bool)
    newrun = L.y & ~(L.y.shift(1, fill_value=False) & (L.recording_id == L.recording_id.shift(1)))
    L["ev"] = np.where(L.y, newrun.cumsum(), 0); L["heart"] = L.patient_id.map(heart).fillna(False)
    L["drug"] = L.recording_id.map(drug).fillna(False); L["day"] = L.recording_id.astype(str) + "_" + L.timestamp.dt.strftime("%Y%m%d")
    first = L[L.ev > 0].groupby("ev").first()
    L["evday"] = L.ev.map(first.day)                                # day of each event's first positive moment
    return L


def alerts(L, gate, conf, off):
    thr = np.where(L.heart, 100, 90) + off
    raw = pd.Series(np.nan_to_num(np.minimum(L.p30, L.p60).values, nan=1e9) <= thr, index=L.index)
    a = raw.copy(); g = L.groupby("recording_id")
    for s in range(1, conf[0]): a &= raw.groupby(L.recording_id).shift(s, fill_value=False)
    if conf[1]: a &= (L.cgm_change_30 < 0)
    name, drugreq, t = gate
    if name != "none":
        ok = np.zeros(len(L), bool)
        if drugreq: ok |= L.drug.values
        if t is not None: ok |= (L.min24 <= t).values
        a &= ok
    return a.values


def minute_stats(L, a):
    rid = L.recording_id.values
    start = a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]]); run = np.cumsum(start) * a; n = int(start.sum())
    ev = L.ev.values; ne = len(set(ev[ev > 0])); caught = int(pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum())
    hit = len(set(run[a & L.y.values])) if n else 0
    return n / (len(L) / 96), caught, ne, (hit / n if n else 0.0)


def day_stats(L, a):
    flagged = set(L.day[a]); days = L.day.nunique(); ed = L[L.ev > 0].groupby("ev").evday.first()
    caught = int(ed.isin(flagged).sum()); hitdays = len(flagged & set(ed))
    return len(flagged) / days, caught, len(ed), (hitdays / len(flagged) if flagged else 0.0)


COMBOS = [(1, "baseline: minute alerts", 0, 0, 0), (2, "1 risk gate", 1, 0, 0), (3, "2 confirmation", 0, 1, 0), (4, "3 worklist", 0, 0, 1),
          (5, "1+2 gate + confirmation", 1, 1, 0), (6, "1+3 gate + worklist", 1, 0, 1), (7, "2+3 confirmation + worklist", 0, 1, 1), (8, "1+2+3 all", 1, 1, 1)]


def settings(cid, gate_on, conf_on):
    return [(g, c, o) for g in (GATES if gate_on else GATES[:1]) for c in (CONF if conf_on else CONF[:1]) for o in OFFS]


def fit_oof(dd):
    dd = dd.copy()
    for h in (30, 60):
        dd["delta"] = dd[f"y{h}"] - dd.cgm_now; dd[f"p{h}"] = np.nan
        for k in range(5):
            tr, es, va = dd[~dd.group.isin([k, (k + 1) % 5])], dd[dd.group == (k + 1) % 5], dd[dd.group == k]
            trn, esn = tr[tr.delta.notna()], es[es.delta.notna()]
            m = xgb().fit(trn[feats], trn.delta, eval_set=[(esn[feats], esn.delta)], verbose=False)
            ok = (va.delta.notna() & va.cgm_now.notna()).values
            dd.loc[va.index[ok], f"p{h}"] = va.cgm_now[ok] + m.predict(va[ok][feats])
    return dd


dev = data[data.group != "test"]; D = prep(fit_oof(dev[dev.low60.notna()]))
test = data[data.group == "test"].copy()
for h in (30, 60):
    m = XGBRegressor(); m.load_model(f"models/forecast_{h}_mean.json")
    ok = test[f"y{h}"].notna() & test.cgm_now.notna(); test[f"p{h}"] = np.nan; test.loc[ok, f"p{h}"] = test.cgm_now[ok] + m.predict(test[ok][feats])
T = prep(test)
say("EXPERIMENT 4: every combination of risk gate (1), confirmation (2) and patient-day worklist (3)")
say(f"development: {D.patient_id.nunique()} patients, {len(set(D.ev[D.ev > 0]))} low events; test: {T.patient_id.nunique()} patients, {len(set(T.ev[T.ev > 0]))} low events")
say(f"setting search per combination, kept only if >= {KEEP} of the development events are warned; burden = alerts/patient-day (minute view) or share of patient-days flagged (worklist)\n")
rows = []
for cid, name, g_on, c_on, w_on in COMBOS:
    stat = day_stats if w_on else minute_stats; best = None
    for gate, conf, off in settings(cid, g_on, c_on):
        r, ca, ne, pr = stat(D, alerts(D, gate, conf, off))
        if ca >= KEEP and (best is None or (r, -pr) < (best[0], -best[3])): best = (r, ca, ne, pr, gate, conf, off)
    if best is None: say(f"{cid} {name}: no setting reaches {KEEP}"); continue
    gate, conf, off = best[4:]
    rt, ct, nt, pt = stat(T, alerts(T, gate, conf, off))
    unit = "of patient-days flagged" if w_on else "alerts/day"
    rows.append((cid, name, best, (rt, ct, nt, pt), unit, w_on))
    say(f"{cid} {name:30s} settings: gate {gate[0]}, confirm {conf[0]} in a row{' + falling' if conf[1] else ''}, offset {off:+d}")
    f = (lambda v: f"{100 * v:.0f}% {unit}") if w_on else (lambda v: f"{v:.2f} {unit}")
    say(f"     development: {f(best[0])}, warned {best[1]}/{best[2]}, precision {100 * best[3]:.1f}%   |   test: {f(rt)}, warned {ct}/{nt}, precision {100 * pt:.1f}%")
say("\nMinute-by-minute combinations (1, 2, 3, 5) are compared by alerts/day; worklist combinations (4, 6, 7, 8) by the share of patient-days flagged.")
