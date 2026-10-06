"""Export one dataset recording as a demo patient file for the demo copy of the app (backend/twin_demo_data/).

Readings are put on the same 15-minute grid the models were trained on. The output holds anonymised public
research data (ShanghaiT2DM, CC BY 4.0, Zhao et al. 2023), so it stays OUT of the public repo (git-ignored);
the demo copy is deployed from a private repo or branch that includes it.

Run:  python export_demo_data.py data/shanghai_t2dm_clean 2094_0_20211109 A
"""
import sys, json, re, os
import numpy as np
import pandas as pd
from labels import load, STEP_MIN


def export(folder, rid, tag, out_dir):
    ts = load(f"{folder}/shanghai_t2dm_timeseries.csv"); g = ts[ts.recording_id == rid]
    s = g.set_index("timestamp")["cgm_mgdl"].resample(f"{STEP_MIN}min", origin="start").mean().dropna()
    readings = [[t.isoformat(), round(float(v), 1)] for t, v in s.items()]
    events = []
    for _, r in g.iterrows():
        t = r.timestamp.isoformat()
        if pd.notna(r.diet_en):
            events.append([t, "meal", str(r.diet_en).split("\n")[0].strip(), None])
        if pd.notna(r.insulin_sc) or pd.notna(r.csii_bolus_iu) or pd.notna(r.insulin_iv):
            m = re.search(r"(\d+(?:\.\d+)?)\s*IU", str(r.insulin_sc)) if pd.notna(r.insulin_sc) else None
            iu = float(m.group(1)) if m else (float(pd.to_numeric(r.csii_bolus_iu, errors="coerce")) if pd.notna(r.csii_bolus_iu) else None)
            iu = None if iu is not None and np.isnan(iu) else iu
            txt = str(r.insulin_sc) if pd.notna(r.insulin_sc) else ("Insulin pump bolus" if pd.notna(r.csii_bolus_iu) else "Insulin by drip")
            events.append([t, "insulin", txt, iu])
        if pd.notna(r.oral_agents):
            events.append([t, "oral", str(r.oral_agents), None])
    st = pd.read_csv(f"{folder}/shanghai_t2dm_static.csv"); st = st[st.recording_id == rid].iloc[0]
    qc = pd.read_csv(f"{folder}/shanghai_t2dm_recording_qc.csv"); qc = qc[qc.recording_id == rid].iloc[0]
    heart = bool(st.has_coronary_heart_disease or st.has_any_macrovascular or st.has_atrial_fibrillation)
    off = qc.cgm_minus_cbg_mean
    note = None
    if pd.notna(off) and qc.cbg_pairs >= 5 and abs(off) >= 8:
        note = (f"The sensor reads about {abs(round(off))} mg/dL {'above' if off > 0 else 'below'} this patient's finger-prick checks, so real sugar may be "
                f"{'lower' if off > 0 else 'higher'}.")
    num = lambda c: None if pd.isna(st[c]) else float(st[c])
    conds = ["insulin"] if st.on_insulin else []
    if st.has_coronary_heart_disease: conds.append("coronary heart disease")
    elif heart: conds.append("heart disease")
    who = "Woman" if st["Gender (Female=1, Male=2)"] == 1 else "Man"
    data = dict(
        patient=dict(name=f"Demo patient {tag}", age=int(st["Age (years)"]), sex="F" if st["Gender (Female=1, Male=2)"] == 1 else "M"),
        profile=dict(label=f"Demo patient {tag}", summary=f"{who}, {int(st['Age (years)'])}. " + ", ".join(c.capitalize() if i == 0 else c for i, c in enumerate(conds)),
                     data_source="ShanghaiT2DM", sex=num("Gender (Female=1, Male=2)"), age=num("Age (years)"), bmi=num("BMI (kg/m2)"),
                     diabetes_years=num("Duration of diabetes (years)"), hba1c=num("HbA1c (mmol/mol)"),
                     egfr=num("Estimated Glomerular Filtration Rate (ml/min/1.73m2)"), hypertension=bool(st.has_hypertension),
                     on_insulin=bool(st.on_insulin), on_sulfonylurea=bool(st.on_sulfonylurea), on_metformin=bool(st.on_metformin),
                     heart_disease=heart, sensor_note=note, confidence="ok", confidence_reason=None, replay_start="2020-11-10T09:52:00"),
        readings=readings, events=events, heart_rate=[])
    os.makedirs(out_dir, exist_ok=True)
    json.dump(data, open(f"{out_dir}/demo_{tag.lower()}.json", "w"), separators=(",", ":"))
    print(tag, rid, len(readings), "readings", len(events), "events", "heart disease" if heart else "")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2], sys.argv[3], os.path.join("..", "backend", "twin_demo_data"))


def export_cgmacros(folder, participant, tag, out_dir):
    """CGMacros participant -> demo file: Dexcom on the 15-minute grid, meals, heart rate (15-minute means).
    Static facts the model needs but CGMacros lacks (diabetes years, kidney function) are left unknown, and the
    patient is marked low-confidence: other sensor, other population, partial facts."""
    t = pd.read_csv(f"{folder}/cgmacros_timeseries.csv.gz"); t = t[t.participant_id == participant].copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], format="mixed"); t = t.sort_values("timestamp")
    s = t.set_index("timestamp")["dexcom_mgdl"].resample(f"{STEP_MIN}min", origin="start").mean().dropna()
    readings = [[x.isoformat(), round(float(v), 1)] for x, v in s.items()]
    hr = t.set_index("timestamp")["hr_bpm"].resample(f"{STEP_MIN}min", origin="start").mean().dropna()
    heart_rate = [[x.isoformat(), round(float(v), 1)] for x, v in hr.items()]
    events = [[r.timestamp.isoformat(), "meal", f"{str(r.meal_type).capitalize()}, {r.carbs_g:g} g carbohydrate" if pd.notna(r.carbs_g) else str(r.meal_type).capitalize(), None]
              for _, r in t[t.meal_type.notna()].iterrows()]
    st = pd.read_csv(f"{folder}/cgmacros_static.csv"); st = st[st.participant_id == participant].iloc[0]
    # replay opens 30 minutes after the meal followed by the biggest rise
    best, when = -1, None
    for e in events:
        m0 = pd.Timestamp(e[0]); seg = s[(s.index >= m0) & (s.index <= m0 + pd.Timedelta(hours=2))]
        if len(seg) > 4 and seg.max() - seg.iloc[0] > best:
            best, when = seg.max() - seg.iloc[0], m0
    start = (s.index[s.index >= when + pd.Timedelta(minutes=30)][0]).isoformat()
    who = "Man" if st.gender == "M" else "Woman"
    data = dict(
        patient=dict(name=f"Demo patient {tag}", age=int(st.age_years), sex=st.gender),
        profile=dict(label=f"Demo patient {tag}", summary=f"{who}, {int(st.age_years)}. Type 2 diabetes, heart-rate data", data_source="CGMacros",
                     sex=2.0 if st.gender == "M" else 1.0, age=float(st.age_years), bmi=float(st.bmi_kgm2), diabetes_years=None,
                     hba1c=round((float(st.hba1c_pct) - 2.15) * 10.929, 1), egfr=None, hypertension=False, on_insulin=False, on_sulfonylurea=False,
                     on_metformin=False, heart_disease=False, sensor_note=None, confidence="low_confidence",
                     confidence_reason="this patient comes from another dataset (different sensor and population) and some patient facts are unknown, so treat the numbers as less certain.",
                     replay_start=start),
        readings=readings, events=events, heart_rate=heart_rate)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(data, open(f"{out_dir}/demo_{tag.lower()}.json", "w"), separators=(",", ":"))
    print(tag, f"CGMacros {participant}", len(readings), "readings", len(events), "meals", len(heart_rate), "heart-rate points; replay opens", start)
