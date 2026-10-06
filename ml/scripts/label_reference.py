"""Reference implementation of the ShanghaiT2DM labels (decisions 21, 33, 39, 40, 42), 2 Oct 2026.

Purpose: produce the expected label counts that Garv's ml/labels.py must reproduce. Not the final
training code. Input: shanghai_t2dm_timeseries.csv (cleaned). Run: python label_reference.py

Rules
- Data fix: finger-prick values above 600 mg/dL are set to empty (recording 2044 has 2,044.8, the patient ID).
- Offset correction (21, 42): per recording, offset = median(sensor - finger-prick) over pairs within 15 min;
  recordings with fewer than 5 pairs use the dataset median. corrected = sensor - offset.
  Labels use corrected values; model INPUTS stay raw sensor readings.
- Grid: each recording on its own 15-minute grid; gaps stay empty, never filled.
- low60 (21, 27, 40): at moment t with corrected value >= 70, a low episode (2+ readings in a row below 70)
  STARTS within the next 60 min (next 4 readings). Needs the next 5 readings present.
  Decision 33: moments in the first 12 h of a recording get no low label (main run); the no-rule run is the check.
- spike120 (27, 39, 40, 42): at t with corrected value <= 180, a spike episode (2+ readings above 180)
  starts within the next 2 h (next 8 readings). Needs the next 9 readings present.
- large_rise120 (39, option D): usual = median of the previous 24 h (needs 72 of 96 readings);
  bar = max(own 90th percentile of the previous 24 h, usual + 30). At t below the bar, any of the next 8
  readings reaches the bar. Uses corrected values (the offset cancels out, so raw gives the same result).
"""
import sys, numpy as np, pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'shanghai_t2dm_timeseries.csv'
ts = pd.read_csv(path, low_memory=False, usecols=['patient_id', 'recording_id', 'timestamp', 'cgm_mgdl', 'cbg_mgdl'])
ts['timestamp'] = pd.to_datetime(ts['timestamp'], format='mixed')
ts.loc[ts['cbg_mgdl'] > 600, 'cbg_mgdl'] = np.nan
ts = ts.sort_values(['recording_id', 'timestamp'])

cgm = ts.dropna(subset=['cgm_mgdl'])
fp = ts.dropna(subset=['cbg_mgdl'])[['recording_id', 'timestamp', 'cbg_mgdl']]
pairs = pd.merge_asof(fp.sort_values('timestamp'), cgm[['recording_id', 'timestamp', 'cgm_mgdl']].sort_values('timestamp'),
                      on='timestamp', by='recording_id', direction='nearest', tolerance=pd.Timedelta('15min')).dropna()
pairs['d'] = pairs['cgm_mgdl'] - pairs['cbg_mgdl']
per = pairs.groupby('recording_id')['d'].agg(['size', 'median'])
DATASET_OFFSET = pairs['d'].median()
offset = {r: (per.loc[r, 'median'] if r in per.index and per.loc[r, 'size'] >= 5 else DATASET_OFFSET)
          for r in ts['recording_id'].unique()}

def starts(v, below, thr):
    x = (v < thr) if below else (v > thr)
    run2 = np.zeros(len(v), bool); run2[:-1] = x[:-1] & x[1:]
    return run2 & ~np.r_[False, x[:-1]]

rows = []
for rid, g in ts.groupby('recording_id'):
    s = g.set_index('timestamp')['cgm_mgdl'].resample('15min', origin='start').mean()
    for src, v in (('corrected', s.values - offset[rid]), ('raw', s.values)):
        n = len(v); hours = np.arange(n) * 0.25
        lo, hi = starts(v, True, 70), starts(v, False, 180)
        past = pd.Series(v).shift(1)
        usual = past.rolling(96, min_periods=72).median().values
        p90 = past.rolling(96, min_periods=72).quantile(0.9).values
        for i in range(n):
            if np.isnan(v[i]):
                continue
            r = dict(recording_id=rid, patient_id=int(rid[:4]), src=src, hours=hours[i])
            if i + 5 < n and not np.isnan(v[i + 1:i + 6]).any() and v[i] >= 70:
                r['low60'] = bool(lo[i + 1:i + 5].any())
            if i + 9 < n and not np.isnan(v[i + 1:i + 10]).any() and v[i] <= 180:
                r['spike120'] = bool(hi[i + 1:i + 9].any())
            if i + 8 < n and not np.isnan(v[i + 1:i + 9]).any() and not np.isnan(usual[i]):
                bar = max(p90[i], usual[i] + 30)
                if v[i] < bar:
                    r['large_rise120'] = bool((v[i + 1:i + 9] >= bar).any())
            rows.append(r)
d = pd.DataFrame(rows)

def report(x, col, name):
    y = x[col].dropna().astype(bool)
    print(f"{name:42s} moments {len(y):7d}  positive {int(y.sum()):6d} ({100 * y.mean():4.1f}%)  "
          f"patients with any {x.loc[y[y].index, 'patient_id'].nunique():3d}")

print(f"dataset median offset {DATASET_OFFSET:.1f}; recordings with own offset {(per['size'] >= 5).sum()}")
c, r = d[d.src == 'corrected'], d[d.src == 'raw']
report(c[c.hours >= 12], 'low60', 'MAIN low60 (corrected, 12 h rule)')
report(c, 'low60', 'check: low60 corrected, no 12 h rule')
report(r[r.hours >= 12], 'low60', 'check: low60 raw, 12 h rule')
report(c, 'spike120', 'MAIN spike120 (corrected)')
report(r, 'spike120', 'check: spike120 raw')
report(c, 'large_rise120', 'MAIN large_rise120 (option D)')
