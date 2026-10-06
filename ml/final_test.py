"""FINAL locked-test scoring (run ONCE). Frozen plan, written before looking at any test result:

Training: the 80 development patients only (splits.csv, groups 0-4). Number of trees per model = median of the
best iterations found in 5-fold patient-wise CV on the development patients. No test patient is used to choose
anything. Test group = 21 patients (20 random + demo patient 2094).

Scored on the test group:
  1. spike120 and large_rise120: XGBoost vs the simple rule and logistic regression (AUC, PR AUC, Brier, calibration).
  2. Forecast at +15 / +30 / +60 / +120 min: XGBoost vs persistence, straight line, ridge; 10-90% band coverage.
  3. Lows: the alert design shown in the demo (forecast: lowest of the +30 and +60 min predictions, alert when at or
     under 90 mg/dL, or 100 for patients with heart disease; thresholds fixed in advance), compared with the plain
     sugar rule and the sugar + 0.5 x 30-min change rule at the same thresholds, and with the constrained warning
     model at the same number of alerts per patient-day.
  4. Demo patient 2094 reported separately.
Models trained here are saved in ml/models/ (these are the models the demo will use).

Run once:  python final_test.py data/shanghai_t2dm_clean      (refuses to run again unless --force)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from baseline import prepare, RULE_SCORE
from forecast import add_targets, HORIZONS
from constrained_lows import DOWN

DRY = "--dry" in sys.argv          # rehearsal: dev folds 0-3 train, fold 4 stands in for the test group; the real test is never touched
OUT = "dry_run_results.txt" if DRY else "final_test_results.txt"
MD = "models_dry" if DRY else "models"
if not DRY and os.path.exists(OUT) and "--force" not in sys.argv:
    sys.exit(f"{OUT} exists: the locked test was already scored. Refusing to score it again.")
LOG = open(OUT, "w", encoding="utf8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


CLS = dict(learning_rate=0.05, max_depth=4, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="logloss", random_state=42, n_jobs=4)
CON = dict(CLS, max_depth=3, min_child_weight=10)
REG = dict(learning_rate=0.05, max_depth=4, min_child_weight=10, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=4)


def cv_rounds(make, dev, feats, ycol):
    """median best iteration over the 5 patient-wise development folds (early stopping on fold k+1)."""
    its = []; ks = sorted(dev["group"].unique())
    for i, k in enumerate(ks):
        nxt = ks[(i + 1) % len(ks)]
        tr, es = dev[~dev["group"].isin([k, nxt])], dev[dev["group"] == nxt]
        m = make(1000, 50); m.fit(tr[feats], tr[ycol], eval_set=[(es[feats], es[ycol])], verbose=False); its.append(m.best_iteration + 1)
    return int(np.median(its)), its


def ece(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop"); g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q", observed=True)
    return float((g.size() / len(y) * (g["p"].mean() - g["y"].mean()).abs()).sum())


def episodes(df, ycol="y"):
    newrun = df[ycol] & ~(df[ycol].shift(1, fill_value=False) & (df.recording_id == df.recording_id.shift(1)))
    return np.where(df[ycol], newrun.cumsum(), 0)


def alert_stats(df, a):
    """alerts per patient-day and share of low events (runs of positive moments) with at least one alert inside."""
    a = np.asarray(a, bool); rid = df.recording_id.values
    runs = int((a & ~(np.r_[False, a[:-1]] & np.r_[False, rid[1:] == rid[:-1]])).sum())
    ev = df["ev"].values; days = len(df) / 96; n = len(set(ev[ev > 0]))
    caught = pd.Series(a[ev > 0]).groupby(ev[ev > 0]).any().sum() if n else 0
    return runs / days, int(caught), n


def main():
    folder = sys.argv[1]; t0 = time.time(); os.makedirs(MD, exist_ok=True)
    data, feats = prepare(folder); data = add_targets(data, folder)
    if DRY:
        data = data[data["group"] != "test"].copy(); data["group"] = data["group"].replace(4, "test")
    dev, test = data[data["group"] != "test"].copy(), data[data["group"] == "test"].copy()
    say(f"development patients {dev.patient_id.nunique()}, locked test patients {test.patient_id.nunique()}, features {len(feats)}")
    meta = dict(features=feats, test_patients=sorted(int(x) for x in test.patient_id.unique()), models={})

    # ---- 1. spike120 and large_rise120
    say("\n=== 1. Warning labels on the locked test group ===")
    for t in ("spike120", "large_rise120"):
        d, e = dev[dev[t].notna()], test[test[t].notna()]; yd, ye = d[t].astype(int), e[t].astype(int)
        dd = d.assign(_y=yd)
        n, its = cv_rounds(lambda ne, es: XGBClassifier(n_estimators=ne, early_stopping_rounds=es, **CLS), dd, feats, "_y")
        m = XGBClassifier(n_estimators=n, **CLS).fit(d[feats], yd); m.save_model(f"{MD}/{t}.json"); meta["models"][t] = dict(trees=n, cv_best_iterations=its)
        p = m.predict_proba(e[feats])[:, 1]
        lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(d[feats], yd)
        rs = RULE_SCORE[t](e).fillna(0)
        say(f"{t}: test moments {len(e)}, positive rate {ye.mean():.3f}, trees {n}")
        say(f"  XGBoost            AUC {roc_auc_score(ye, p):.3f}  PR AUC {average_precision_score(ye, p):.3f}  Brier {brier_score_loss(ye, p):.4f} (base rate {brier_score_loss(ye, np.full(len(ye), yd.mean())):.4f})  calibration error {ece(ye.values, p):.3f}")
        say(f"  logistic regression AUC {roc_auc_score(ye, lr.predict_proba(e[feats])[:, 1]):.3f}  PR AUC {average_precision_score(ye, lr.predict_proba(e[feats])[:, 1]):.3f}")
        say(f"  simple rule        AUC {roc_auc_score(ye, rs):.3f}  PR AUC {average_precision_score(ye, rs):.3f}")
        pa = []
        for pid, g in e.assign(p=p).groupby("patient_id"):
            if g[t].nunique() == 2: pa.append(roc_auc_score(g[t].astype(int), g["p"]))
        say(f"  per-patient AUC over {len(pa)} patients with both outcomes: min {min(pa):.2f}, median {np.median(pa):.2f}, max {max(pa):.2f}")

    # ---- 2. forecast
    say("\n=== 2. Forecast on the locked test group (mg/dL) ===")
    preds = {}
    for h in HORIZONS:
        d = dev[dev[f"y{h}"].notna() & dev["cgm_now"].notna()].copy(); e = test[test[f"y{h}"].notna() & test["cgm_now"].notna()].copy()
        d["delta"], e["delta"] = d[f"y{h}"] - d["cgm_now"], e[f"y{h}"] - e["cgm_now"]
        out = {}
        for name, alpha in (("mean", None), ("q10", 0.1), ("q90", 0.9)):
            kw = dict(objective="reg:quantileerror", quantile_alpha=alpha) if alpha else {}
            n, its = cv_rounds(lambda ne, es: XGBRegressor(n_estimators=ne, early_stopping_rounds=es, **REG, **kw), d, feats, "delta")
            m = XGBRegressor(n_estimators=n, **REG, **kw).fit(d[feats], d["delta"]); m.save_model(f"{MD}/forecast_{h}_{name}.json"); meta["models"][f"forecast_{h}_{name}"] = dict(trees=n)
            out[name] = e["cgm_now"] + m.predict(e[feats])
        y = e[f"y{h}"]; low = e["cgm_now"] < 100
        rg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)).fit(d[feats], d["delta"]); ridge = e["cgm_now"] + rg.predict(e[feats])
        line = e["cgm_now"] + (h / 30) * e["cgm_change_30"].fillna(0); pers = e["cgm_now"]
        mae = lambda p: float((p - y).abs().mean())
        lo, hi = np.minimum(out["q10"], out["q90"]), np.maximum(out["q10"], out["q90"])
        say(f"+{h:3d} min: MAE xgboost {mae(out['mean']):.2f} | persistence {mae(pers):.2f} | straight line {mae(line):.2f} | ridge {mae(ridge):.2f} || sugar<100: xgboost {float((out['mean'] - y).abs()[low].mean()):.2f}, persistence {float((pers - y).abs()[low].mean()):.2f} || band covers {100 * ((y >= lo) & (y <= hi)).mean():.0f}% (target 80%), mean width {float((hi - lo).mean()):.0f}")
        preds[h] = out["mean"].rename(f"p{h}")
    P = pd.concat([test[["recording_id", "timestamp"]].join(preds[15].rename("p15")), preds[30].rename("p30"), preds[60].rename("p60")], axis=1)

    # ---- 3. lows
    say("\n=== 3. Lows on the locked test group ===")
    st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st["patient_id"] = st["patient_id"].astype(int)
    heart = st.groupby("patient_id")[["has_coronary_heart_disease", "has_any_macrovascular", "has_atrial_fibrillation"]].max().any(axis=1)
    L = test[test["low60"].notna()].sort_values(["recording_id", "timestamp"]).reset_index(drop=True)
    L = L.merge(P[["recording_id", "timestamp", "p30", "p60"]], on=["recording_id", "timestamp"], how="left")
    L["y"] = L["low60"].astype(bool); L["ev"] = episodes(L)
    L["heart"] = L.patient_id.map(heart).fillna(False)
    thr = np.where(L["heart"], 100, 90)
    # constrained warning model (trained on development patients)
    mono = tuple(DOWN.get(f, 0) for f in feats); d = dev[dev["low60"].notna()].copy(); d["_y"] = d["low60"].astype(int)
    n, its = cv_rounds(lambda ne, es: XGBClassifier(n_estimators=ne, early_stopping_rounds=es, monotone_constraints=mono, **CON), d, feats, "_y")
    cm = XGBClassifier(n_estimators=n, monotone_constraints=mono, **CON).fit(d[feats], d["_y"]); cm.save_model(f"{MD}/low60_constrained.json"); meta["models"]["low60_constrained"] = dict(trees=n)
    L["warn"] = cm.predict_proba(L[feats])[:, 1]
    lowest = np.minimum(L.p30, L.p60).values
    say(f"low events in the test group: {L.ev.replace(0, np.nan).nunique()} (patients with a low event: {L[L.ev > 0].patient_id.nunique()}), patient-days {len(L) / 96:.0f}; heart-disease patients: {int(L.groupby('patient_id').heart.first().sum())} of {L.patient_id.nunique()}")
    say("Fixed alert rules (90 mg/dL, 100 for heart-disease patients):")
    cand = {"forecast alert (lowest of +30 and +60 min)": np.nan_to_num(lowest, nan=1e9) <= thr,
            "plain sugar rule": (L.cgm_now.values <= thr), "sugar + 0.5 x 30-min change": ((L.cgm_now + 0.5 * L.cgm_change_30).values <= thr)}
    for name, a in cand.items():
        r, c, ne = alert_stats(L, a); say(f"  {name:46s} {r:.2f} alerts/patient-day, warned {c} of {ne} low events ({100 * c / max(1, ne):.0f}%)")
    r0 = alert_stats(L, cand["forecast alert (lowest of +30 and +60 min)"])[0]
    s = L["warn"].values; order = np.quantile(s, np.linspace(0.999, 0.75, 250)); best = None
    for q in order:
        res = alert_stats(L, s >= q)
        if res[0] <= r0: best = res
    if best: say(f"  constrained warning model at a matched {r0:.2f} alerts/patient-day: warned {best[1]} of {best[2]} low events ({100 * best[1] / max(1, best[2]):.0f}%)")
    say("Per patient with low events (forecast alert): events, warned, alerts per day")
    a = cand["forecast alert (lowest of +30 and +60 min)"]
    for pid, g in L[L.ev > 0].groupby("patient_id"):
        sub = L[L.patient_id == pid]; ia = a[sub.index]; r, c, ne = alert_stats(sub.assign(ev=sub.ev), ia)
        say(f"  patient {pid}: {ne} events, warned {c}, {r:.2f} alerts/day")

    # ---- 4. demo patient 2094
    say("\n=== 4. Demo patient 2094 ===")
    demo = 2084 if DRY else 2094          # rehearsal: exercise this section on a stand-in patient
    g = L[L.patient_id == demo]
    if not len(g):
        say("demo patient 2094 is not in this group (rehearsal run)"); json.dump(meta, open(f"{MD}/meta.json", "w"), indent=1); return
    ia = a[g.index]; r, c, ne = alert_stats(g, ia)
    say(f"forecast alert: {ne} low events, warned {c}, {r:.2f} alerts/day over {len(g) / 96:.1f} days")
    w0, w1 = (g.timestamp.min(), g.timestamp.min() + pd.Timedelta(hours=6)) if DRY else (pd.Timestamp("2020-11-10 06:00"), pd.Timestamp("2020-11-10 11:30"))
    day = g[(g.timestamp >= w0) & (g.timestamp <= w1)]
    say("showcase window (time, sugar now, predicted lowest, alert):")
    for i, row in day.iterrows():
        if row.timestamp.minute in (7, 22, 37, 52):
            say(f"  {row.timestamp:%H:%M}  {row.cgm_now:6.1f}  {np.minimum(row.p30, row.p60):6.1f}  {'ALERT' if a[i] else ''}")
    json.dump(meta, open(f"{MD}/meta.json", "w"), indent=1)
    say(f"\nmodels saved in ml/models/ ; finished in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
