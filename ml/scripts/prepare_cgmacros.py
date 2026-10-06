"""Clean CGMacros (PhysioNet v1.0.0) into the same three-file layout as ShanghaiT2DM.

Input : CGMacros/ folder from CGMacros_dateshifted365.zip (photos not needed).
Output: cgmacros_timeseries.csv, cgmacros_static.csv, cgmacros_participant_qc.csv

Key CGMacros-specific step: the export fills BOTH CGMs to every minute by straight-line
interpolation between real readings (Libre every 15 min, Dexcom every 5 min), and also
interpolates straight across sensor gaps. We keep only real readings:
  1. "knots" = points where the line's slope changes (each is a real reading);
  2. between two knots, points on the sensor's grid (every 15 / 5 min) are real readings
     that happen to lie on a straight line;
  3. a straight stretch of >= GAP_INTERVALS sensor intervals is treated as an interpolated
     gap, and its interior is set to missing. Thresholds chosen from the stretch-length
     distribution (it stops decaying at these lengths); heuristic, not documented by the authors.
"""
import pandas as pd, numpy as np, glob, os, re, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else 'CGMacros'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out'
os.makedirs(OUT, exist_ok=True)
SENSORS = {'libre': ('Libre GL', 15, 5),    # column, sampling minutes, gap if straight for >= N intervals
           'dexcom': ('Dexcom GL', 5, 7)}
SLOPE_TOL = 1e-3                            # mg/dL per minute; source values carry ~6 decimals
# One sensor's clock is ~1 hour late in these participants (found by cross-correlating Libre vs Dexcom
# and by which sensor peaks implausibly late after logged meals). Its readings are moved 60 min earlier.
CLOCK_FIX = {6: 'libre', 7: 'libre', 8: 'libre', 30: 'libre', 9: 'dexcom', 13: 'dexcom'}

def real_readings(t, v, P, gap_n):
    """Return (mask of real readings, list of removed-gap lengths in minutes)."""
    ok = v.notna().to_numpy(); idx = np.flatnonzero(ok)
    mask = np.zeros(len(v), bool); gaps = []
    if len(idx) < 2:
        mask[idx] = True; return mask, gaps
    tt = t[idx]; vv = v.to_numpy()[idx]
    slope = np.diff(vv) / np.diff(tt)
    knot = np.r_[True, np.abs(np.diff(slope)) > SLOPE_TOL, True]
    k = np.flatnonzero(knot)
    mask[idx[k]] = True
    for a, b in zip(k[:-1], k[1:]):
        span = tt[b] - tt[a]
        if span >= gap_n * P - 0.5:                       # interpolated across a gap
            gaps.append(span); continue
        inner = np.arange(a + 1, b)
        on_grid = np.abs(((tt[inner] - tt[a] + P / 2) % P) - P / 2) < 0.5
        mask[idx[inner[on_grid]]] = True
    return mask, gaps

def episodes(t, v, thr, P, below=True, min_minutes=15):
    """Runs of real readings beyond thr lasting >= min_minutes (consecutive = <= 1.5 sampling intervals apart).
    Libre: 2+ readings (same rule as ShanghaiT2DM); Dexcom: 4+ readings."""
    s = pd.DataFrame({'t': t, 'v': v}).dropna()
    if s.empty: return 0
    x = (s.v < thr) if below else (s.v > thr)
    brk = (x != x.shift()) | (s.t.diff() > 1.5 * P)
    g = brk.cumsum()
    r = s.assign(x=x).groupby(g).agg(x=('x', 'first'), n=('v', 'size'))
    need = int(np.ceil(min_minutes / P)) + 1
    return int((r.x & (r.n >= need)).sum())

def norm_meal(s):
    s = s.astype('string').str.strip().str.lower()
    return s.replace({'snacks': 'snack', 'snack 1': 'snack'})

frames, qc = [], []
for f in sorted(glob.glob(os.path.join(SRC, 'CGMacros-0*', 'CGMacros-0*.csv'))):
    pid = int(re.search(r'CGMacros-0*(\d+)\.csv', f).group(1))
    d = pd.read_csv(f); d.columns = d.columns.str.strip()          # 9 header variants; map by name
    d['Timestamp'] = pd.to_datetime(d['Timestamp'], errors='coerce')
    bad_ts = int(d.Timestamp.isna().sum()); d = d.dropna(subset=['Timestamp'])
    dups = int(d.duplicated('Timestamp').sum())
    d = d.sort_values('Timestamp').drop_duplicates('Timestamp').reset_index(drop=True)
    tmin = (d.Timestamp - d.Timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60
    o = pd.DataFrame({'participant_id': pid, 'timestamp': d.Timestamp})
    row = dict(participant_id=pid, rows=len(d), bad_timestamps=bad_ts, dup_timestamps=dups,
               start=d.Timestamp.min(), end=d.Timestamp.max(),
               days=round((d.Timestamp.max() - d.Timestamp.min()).total_seconds() / 86400, 2),
               time_jumps_over_1min=int((np.diff(tmin) > 1.5).sum()))
    sens = {}
    for name, (col, P, gap_n) in SENSORS.items():
        v = pd.to_numeric(d[col], errors='coerce')
        m, gaps = real_readings(tmin, v, P, gap_n)
        r = pd.DataFrame({'timestamp': d.Timestamp[m], f'{name}_mgdl': v[m]})
        if CLOCK_FIX.get(pid) == name:
            r['timestamp'] -= pd.Timedelta(minutes=60)
        sens[name] = (r, gaps)
    for name in SENSORS:
        o = o.merge(sens[name][0], on='timestamp', how='outer')
    o['participant_id'] = pid
    o = o.sort_values('timestamp').reset_index(drop=True)
    d = d.set_index('Timestamp').reindex(o.timestamp).reset_index().rename(columns={'index': 'Timestamp'})
    d.columns = ['Timestamp'] + list(d.columns[1:])
    tmin = (o.timestamp - o.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60
    row['clock_shift'] = f'{CLOCK_FIX[pid]} -60min' if pid in CLOCK_FIX else pd.NA
    for name, (col, P, gap_n) in SENSORS.items():
        r = o[f'{name}_mgdl']; gaps = sens[name][1]; tr = tmin[r.notna().to_numpy()]
        expected = (tr.max() - tr.min()) / P + 1
        row.update({f'{name}_readings': int(r.notna().sum()),
                    f'{name}_coverage_pct': round(100 * r.notna().sum() / expected, 1),
                    f'{name}_interp_gaps_removed': len(gaps),
                    f'{name}_interp_gap_hours_removed': round(sum(gaps) / 60, 1),
                    f'{name}_nonint_pct': round(100 * (r.dropna() % 1 != 0).mean(), 2),
                    f'{name}_median': r.median(),
                    f'{name}_pct_lo70': round(100 * (r.dropna() < 70).mean(), 2),
                    f'{name}_pct_hi180': round(100 * (r.dropna() > 180).mean(), 2),
                    f'{name}_at_floor40': int((r == 40).sum()),
                    f'{name}_lo70_episodes': episodes(tmin, r, 70, P),
                    f'{name}_lo54_episodes': episodes(tmin, r, 54, P),
                    f'{name}_hi180_episodes': episodes(tmin, r, 180, P, below=False)})
        o[f'{name}_lo70'] = r < 70; o[f'{name}_lo54'] = r < 54; o[f'{name}_hi180'] = r > 180
    # residual lag after any clock fix: shift of Libre (min) that best matches Dexcom
    a = o[['timestamp', 'libre_mgdl']].dropna(); b = o[['timestamp', 'dexcom_mgdl']].dropna(); cs = {}
    for L in range(-120, 121, 5):
        mm = pd.merge_asof(a.assign(timestamp=a.timestamp + pd.Timedelta(minutes=L)), b, on='timestamp',
                           direction='nearest', tolerance=pd.Timedelta('150s')).dropna()
        cs[L] = mm.libre_mgdl.corr(mm.dexcom_mgdl) if len(mm) > 50 else np.nan
    best = max((k for k in cs if pd.notna(cs[k])), key=cs.get)
    row.update(libre_to_dexcom_best_lag_min=best, libre_dexcom_corr=round(cs[0], 3))
    # Libre vs Dexcom agreement on real readings (nearest within 2.5 min)
    a = o[['timestamp', 'libre_mgdl']].dropna(); b = o[['timestamp', 'dexcom_mgdl']].dropna()
    pr = pd.merge_asof(a, b, on='timestamp', direction='nearest', tolerance=pd.Timedelta('150s')).dropna()
    row.update(libre_dexcom_pairs=len(pr),
               libre_minus_dexcom_mean=round((pr.libre_mgdl - pr.dexcom_mgdl).mean(), 1) if len(pr) else np.nan,
               libre_dexcom_mard_pct=round((100 * (pr.libre_mgdl - pr.dexcom_mgdl).abs() / pr.dexcom_mgdl).mean(), 1) if len(pr) else np.nan)
    # Fitbit
    o['hr_bpm'] = pd.to_numeric(d['HR'], errors='coerce')
    o['activity_kcal'] = pd.to_numeric(d['Calories (Activity)'], errors='coerce') if 'Calories (Activity)' in d else np.nan
    o['mets'] = pd.to_numeric(d['METs'], errors='coerce') / 10 if 'METs' in d else np.nan   # source is METs x 10
    o['fitbit_intensity'] = pd.to_numeric(d['Intensity'], errors='coerce') if 'Intensity' in d else np.nan
    o['steps'] = pd.to_numeric(d['Steps'], errors='coerce') if 'Steps' in d else np.nan
    span_min = tmin[-1] + 1
    hr_t = d.Timestamp[o.hr_bpm.notna()]
    hr_gap = hr_t.diff().dt.total_seconds().max() / 3600 if len(hr_t) > 1 else np.nan
    perday = o.set_index('timestamp').hr_bpm.notna().resample('1D').sum() / 1440
    row.update(hr_pct_of_span=round(100 * o.hr_bpm.notna().sum() / span_min, 1),
               hr_full_days=int((perday.iloc[1:-1] >= 0.8).sum()), hr_inner_days=max(len(perday) - 2, 0),
               hr_longest_gap_hours=round(hr_gap, 1),
               activity_measure='METs' if 'METs' in d else ('Intensity' if 'Intensity' in d else 'Steps'))
    # Meals
    o['meal_type'] = norm_meal(d['Meal Type'])
    for src, dst in [('Calories', 'meal_kcal'), ('Carbs', 'carbs_g'), ('Protein', 'protein_g'), ('Fat', 'fat_g'),
                     ('Fiber', 'fiber_g'), ('Sugar', 'sugar_g'), ('Amount Consumed', 'amount_consumed_raw')]:
        o[dst] = pd.to_numeric(d[src], errors='coerce') if src in d else np.nan
    o['image_path'] = d['Image path'].astype('string')
    meal = o.meal_type.notna()
    atwater = 4 * o.carbs_g + 4 * o.protein_g + 9 * o.fat_g
    ratio = atwater / o.meal_kcal
    flags = pd.Series('', index=o.index, dtype='string')
    flags[meal & (o.fiber_g > 60)] += 'fiber_implausible;'
    flags[meal & (o.meal_kcal > 0) & ((ratio < 0.5) | (ratio > 1.5))] += 'kcal_vs_macros_mismatch;'
    flags[meal & (o.meal_kcal > 2000)] += 'kcal_over_2000;'
    o['meal_flag'] = flags.where(meal & (flags != ''))
    o.loc[o.fiber_g > 60, 'fiber_g'] = np.nan                         # raw value kept in fiber_g_raw
    o['fiber_g_raw'] = pd.to_numeric(d['Fiber'], errors='coerce')
    ac = o.amount_consumed_raw[meal].dropna()
    enc = ('absent' if ac.empty else 'percent_0_100' if ac.max() <= 100 and (ac > 9).any()
           else 'small_integers_1_9' if ac.max() <= 9 else 'multiples_of_100_up_to_%d' % ac.max())
    row.update(meals_logged=int(meal.sum()), meals_flagged=int(o.meal_flag.notna().sum()),
               meal_types=dict(o.meal_type.value_counts()), amount_consumed_encoding=enc)
    frames.append(o); qc.append(row)

ts = pd.concat(frames, ignore_index=True); q = pd.DataFrame(qc)

# ---------------- static table from bio.csv ----------------
b = pd.read_csv(os.path.join(SRC, 'bio.csv')); b.columns = [c.strip() for c in b.columns]
s = pd.DataFrame({
    'participant_id': b['subject'], 'age_years': b['Age'], 'gender': b['Gender'],
    'ethnicity_self_identified': b['Self-identify'].str.strip(),
    'height_m': (b['Height'] * 0.0254).round(3), 'weight_kg': (b['Body weight'] * 0.45359237).round(1),
    'bmi_kgm2': b['BMI'].round(2),
    'hba1c_pct': b['A1c PDL (Lab)'],                       # dictionary says mmol/mol; values 4.6-8.5 are %
    'fasting_glucose_mgdl': b['Fasting GLU - PDL (Lab)'], 'fasting_insulin_uU_ml': b['Insulin'],
    'triglycerides_mgdl': b['Triglycerides'], 'total_cholesterol_mgdl': b['Cholesterol'], 'hdl_mgdl': b['HDL'],
    'non_hdl_mgdl': b['Non HDL'], 'ldl_calc_mgdl': b['LDL (Cal)'].replace(800, np.nan),
    'vldl_calc_mgdl': b['VLDL (Cal)'].replace(400, np.nan), 'chol_hdl_ratio': b['Cho/HDL Ratio'].replace(400, np.nan),
    'lab_collection_time': b['Collection time PDL (Lab)'],
    'fingerstick1_mgdl': b['#1 Contour Fingerstick GLU'], 'fingerstick1_time': b['Time (t)'],
    'fingerstick2_mgdl': b['#2 Contour Fingerstick GLU'], 'fingerstick2_time': b['Time (t).1'],
    'fingerstick3_mgdl': b['#3 Contour Fingerstick GLU'], 'fingerstick3_time': b['Time (t).2']})
s['bmi_recalc_kgm2'] = (s.weight_kg / s.height_m ** 2).round(2)
s['glycaemic_group_from_hba1c'] = pd.cut(s.hba1c_pct, [-np.inf, 5.65, 6.45, np.inf],
                                         labels=['no_diabetes', 'prediabetes', 'type2_diabetes'])
s['lipid_error_codes_removed'] = (b['LDL (Cal)'] == 800) | (b['VLDL (Cal)'] == 400)

# Finger-sticks are NOT paired with CGM: bio.csv gives times but no dates, finger-stick #1 precedes the
# recording start in 42 of 45 participants, and no recording day matches the three values better than chance
# (checked 2 Oct 2026). They are screening-morning values only.
q = q.merge(s[['participant_id', 'glycaemic_group_from_hba1c']], on='participant_id', how='left')

# ---------------- QC flag ----------------
z = (q.libre_minus_dexcom_mean - q.libre_minus_dexcom_mean.median()) / q.libre_minus_dexcom_mean.std()
notes = pd.Series('', index=q.index, dtype='string')
notes[q.libre_coverage_pct < 80] += 'libre_coverage<80%;'
notes[q.dexcom_coverage_pct < 80] += 'dexcom_coverage<80%;'
notes[z.abs() > 3] += 'libre_dexcom_bias_outlier;'
notes[(q.libre_median < 70) | (q.dexcom_median < 70)] += 'median_cgm<70;'
notes[q.time_jumps_over_1min > 20] += 'irregular_timestamps;'
notes[q.days > 14] += 'span>14d_check_gap;'
notes[q.hr_pct_of_span < 60] += 'hr<60%;'
notes[q.libre_to_dexcom_best_lag_min.abs() > 20] += 'residual_sensor_lag>20min;'
q['qc_notes'] = notes.replace('', pd.NA)
q['qc_flag'] = np.where(q.qc_notes.notna(), 'check', 'ok')

ts.to_csv(os.path.join(OUT, 'cgmacros_timeseries.csv'), index=False)
s.to_csv(os.path.join(OUT, 'cgmacros_static.csv'), index=False)
q.to_csv(os.path.join(OUT, 'cgmacros_participant_qc.csv'), index=False)
print(ts.shape, s.shape, q.shape)
