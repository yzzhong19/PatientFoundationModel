"""Step 2: landmarks.parquet = index-cancer regimen starts strictly after the patient's first sequencing report day.
Labels (OS, imaging PFS) come from regimen-level os_g_* / pfs_i_g_* columns and are used only as targets."""
import numpy as np, pandas as pd
ev = pd.read_parquet('events.parquet')
lab = pd.read_pickle('labels_raw.pkl')
first_seq = ev[ev.etype == 'PANEL'].groupby('patient_id').day.min().rename('first_seq_day')
st = ev[ev.etype == 'STATIC'].assign(k=lambda d: d.value.str.split('=').str[0], v=lambda d: d.value.str.split('=').str[1]).pivot(index='patient_id', columns='k', values='v')
lm = lab.merge(first_seq, left_on='patient_id', right_index=True, how='inner')
n0 = len(lm)
lm = lm[lm.landmark_day > lm.first_seq_day]
lm = lm.dropna(subset=['tt_os_g_days', 'os_g_status', 'landmark_day'])
lm = lm.join(st, on='patient_id')
lm['os_time'] = lm.tt_os_g_days.clip(lower=0.5).astype(float); lm['os_event'] = lm.os_g_status.astype(int)
lm['pfs_time'] = lm.tt_pfs_i_g_days.astype(float).clip(lower=0.5); lm['pfs_event'] = lm.pfs_i_g_status
lm['ref'] = lm.ca_seq.astype(str) + '_' + lm.regimen_number.astype(str)
lm['reg_name'] = lm.names
lm['age_dx'] = lm.AGE
keep = ['patient_id', 'cohort', 'institution', 'ref', 'landmark_day', 'first_seq_day', 'line', 'reg_name', 'dclass', 'STAGE', 'AGE', 'SEX', 'HIST',
        'os_time', 'os_event', 'pfs_time', 'pfs_event']
lm = lm[keep].reset_index(drop=True)
lm['landmark_id'] = np.arange(len(lm))
lm.to_parquet('landmarks.parquet', index=False)
print('regimens', n0, '-> landmarks', len(lm), 'patients', lm.patient_id.nunique())
print(lm.groupby('cohort').agg(n=('os_time', 'size'), pts=('patient_id', 'nunique'), death=('os_event', 'mean'), med_os=('os_time', 'median')))
print(lm.groupby('institution').size())
print('pfs available', lm.pfs_event.notna().mean())
