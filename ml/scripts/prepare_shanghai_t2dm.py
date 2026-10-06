import pandas as pd, numpy as np, glob, re
STD=['timestamp','cgm_mgdl','cbg_mgdl','ketone_mmol','diet_en','diet_cn','insulin_sc','oral_agents','csii_bolus_iu','csii_basal_iu_h','insulin_iv']
frames=[]; qc=[]
for f in sorted(glob.glob('20*.xlsx')):
    rid=f[:-5]; df=pd.read_excel(f)
    df.columns=STD  # all 109 files share the same 11-column order (names vary only in spelling)
    df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
    bad_ts=df['timestamp'].isna().sum(); df=df.dropna(subset=['timestamp'])
    for c in ['cgm_mgdl','cbg_mgdl','ketone_mmol','csii_bolus_iu','csii_basal_iu_h']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    for c in ['diet_en','diet_cn','insulin_sc','oral_agents','insulin_iv']:
        df[c]=df[c].astype('string').str.strip().replace({'':pd.NA})
    # data fix (5 Oct): finger-prick values above 600 mg/dL are impossible; recording 2044 holds 2,044.8 (the patient ID) -> empty
    df.loc[df['cbg_mgdl']>600,'cbg_mgdl']=np.nan
    dups=df.duplicated('timestamp').sum()
    df=df.sort_values('timestamp').drop_duplicates('timestamp')
    df.insert(0,'recording_id',rid); df.insert(0,'patient_id',rid[:4])
    df['lo70']=df.cgm_mgdl<70; df['lo54']=df.cgm_mgdl<54; df['hi180']=df.cgm_mgdl>180
    # CGM vs finger-stick bias: nearest CGM within 15 min of each CBG
    c=df.dropna(subset=['cgm_mgdl'])[['timestamp','cgm_mgdl']]; b=df.dropna(subset=['cbg_mgdl'])[['timestamp','cbg_mgdl']]
    m=pd.merge_asof(b.sort_values('timestamp'),c.sort_values('timestamp'),on='timestamp',direction='nearest',tolerance=pd.Timedelta('15min')).dropna()
    gap=df.timestamp.diff().dt.total_seconds().div(60)
    qc.append(dict(recording_id=rid,patient_id=rid[:4],rows=len(df),cgm_missing=int(df.cgm_mgdl.isna().sum()),
        start=df.timestamp.min(),end=df.timestamp.max(),days=round((df.timestamp.max()-df.timestamp.min()).total_seconds()/86400,2),
        gaps_over_30min=int((gap>30).sum()),dup_timestamps=int(dups),bad_timestamps=int(bad_ts),
        median_cgm=df.cgm_mgdl.median(),pct_lo70=round(df.lo70.mean()*100,2),pct_hi180=round(df.hi180.mean()*100,2),
        meals_logged=int(df.diet_en.notna().sum()),insulin_sc_doses=int(df.insulin_sc.notna().sum()),oral_agent_entries=int(df.oral_agents.notna().sum()),
        cbg_pairs=len(m),cgm_minus_cbg_mean=round((m.cgm_mgdl-m.cbg_mgdl).mean(),1) if len(m) else np.nan))
    frames.append(df)
long=pd.concat(frames,ignore_index=True); q=pd.DataFrame(qc)
# episodes: 2+ consecutive readings
def ep(s,thr,below=True):
    x=(s<thr) if below else (s>thr); g=(x!=x.shift()).cumsum(); r=x.groupby(g).agg(['first','size']); return int(((r['first'])&(r['size']>=2)).sum())
e=long.dropna(subset=['cgm_mgdl']).groupby('recording_id').cgm_mgdl.agg(lo70_episodes=lambda s:ep(s,70),lo54_episodes=lambda s:ep(s,54),hi180_episodes=lambda s:ep(s,180,False)).reset_index()
q=q.merge(e,on='recording_id',how='left')
z=(q.cgm_minus_cbg_mean-q.cgm_minus_cbg_mean.median())/q.cgm_minus_cbg_mean.std()
q['qc_flag']=np.where((q.median_cgm<70)|(z.abs()>3),'suspect','ok')
# static table
s=pd.read_excel('/mnt/project/Shanghai_T2DM_Summary.xlsx'); s.columns=[re.sub(r'\s+',' ',c).strip() for c in s.columns]
s=s.rename(columns={'Patient Number':'recording_id'}); s.insert(1,'patient_id',s.recording_id.str[:4])
s=s.replace('/',np.nan)
low=lambda c: s[c].astype(str).str.lower()
s['has_hypertension']=low('Comorbidities').str.contains('hypertension')
s['has_coronary_heart_disease']=low('Diabetic Macrovascular Complications').str.contains('coronary')
s['has_any_macrovascular']=~low('Diabetic Macrovascular Complications').isin(['none','nan'])
s['has_atrial_fibrillation']=low('Comorbidities').str.contains('fibrillation')
ag=low('Hypoglycemic Agents')
s['on_insulin']=ag.str.contains('insulin|humulin|novolin|gansulin|lantus|glargine|aspart|lispro|detemir|degludec|levemir|humalog|tresiba|toujeo|basaglar|ryzodeg|scilin|novorapid|apidra|glulisine')
s['on_sulfonylurea']=ag.str.contains('glimepiride|gliclazide|glipizide|glibenclamide|gliquidone|glyburide')
s['on_metformin']=ag.str.contains('metformin')
s=s.rename(columns={'Hypoglycemia (yes/no)':'hypoglycemia_history_flag_UNRELIABLE'})
long.to_csv('/home/claude/out/shanghai_t2dm_timeseries.csv',index=False)
s.to_csv('/home/claude/out/shanghai_t2dm_static.csv',index=False)
q.to_csv('/home/claude/out/shanghai_t2dm_recording_qc.csv',index=False)
print(long.shape, s.shape, q.shape)
print(q[q.qc_flag=='suspect'][['recording_id','median_cgm','cgm_minus_cbg_mean','cbg_pairs','pct_lo70']])
print('bias median',q.cgm_minus_cbg_mean.median(),'std',round(q.cgm_minus_cbg_mean.std(),1))
print('totals: lo70',q.lo70_episodes.sum(),'lo54',q.lo54_episodes.sum(),'hi180',q.hi180_episodes.sum())
print('gaps>30',q.gaps_over_30min.sum(),'dups',q.dup_timestamps.sum(),'badts',q.bad_timestamps.sum(),'cgm missing',q.cgm_missing.sum())
print('insulin',s.on_insulin.sum(),'SU',s.on_sulfonylurea.sum(),'met',s.on_metformin.sum())
