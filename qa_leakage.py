"""Leakage QA: label/death alignment, sequencing-after-death, and printed NSCLC timelines (landmark marked)."""
import numpy as np, pandas as pd
ev = pd.read_parquet('events.parquet'); lm = pd.read_parquet('landmarks.parquet'); lab = pd.read_pickle('labels_raw.pkl')
lab['ref'] = lab.ca_seq.astype(str) + '_' + lab.regimen_number.astype(str)
m = lm.merge(lab[['patient_id', 'ref', 'hybrid_death_int', 'dob_lastalive_int', 'anchor_abs', 'start_abs']], on=['patient_id', 'ref'])
d = m[m.os_event == 1]
diff = (d.hybrid_death_int - d.anchor_abs) - (d.landmark_day + d.os_time)
print('death alignment (deceased): n=%d, |diff|<=1d: %.3f, median %.1f, p99 abs %.1f' % (len(d), (diff.abs() <= 1).mean(), diff.median(), diff.abs().quantile(.99)))
c = m[m.os_event == 0]
diffc = (c.dob_lastalive_int - c.anchor_abs) - (c.landmark_day + c.os_time)
print('censor alignment: n=%d, |diff|<=1d: %.3f' % (len(c), (diffc.abs() <= 1).mean()))
# events after death
death = (lab.drop_duplicates('patient_id').set_index('patient_id').pipe(lambda x: x.hybrid_death_int - x.anchor_abs)).dropna()
e = ev[ev.etype != 'STATIC'].join(death.rename('dd'), on='patient_id')
print('non-static events after death day: %d of %d (%.4f%%)' % ((e.day > e.dd).sum(), e.dd.notna().sum(), 100 * (e.day > e.dd).sum() / e.dd.notna().sum()))
print('landmark_day > death-day (should be 0):', int((m.landmark_day > (m.hybrid_death_int - m.anchor_abs)).sum()))
print('landmark == first_seq_day (should be 0):', int((lm.landmark_day <= lm.first_seq_day).sum()))
# timelines
rng = np.random.RandomState(0)
n = lm[lm.cohort == 'NSCLC']
pts = rng.choice(n.patient_id.unique(), 10, replace=False)
for p in pts:
    r = n[n.patient_id == p].sort_values('landmark_day').iloc[min(1, (n.patient_id == p).sum() - 1)]
    e = ev[ev.patient_id == p]
    past = e[(e.day < r.landmark_day) | ((e.day == r.landmark_day) & (e['rank'] <= 14) & ~((e.etype == 'REG_START') & (e.ref != r.ref)))]
    # events with (day,rank) after landmark token are excluded by the rule: day<L, or day==L and rank<=REG_START & not other regimens' later
    fut = e[(e.day > r.landmark_day)]
    print('\n=== %s inst=%s line=%d landmark day %d regimen=%s | OS %.0fd event=%d' % (p, r.institution, r.line, r.landmark_day, r.reg_name, r.os_time, r.os_event))
    toks = past.sort_values(['day', 'rank'])
    stat = ' '.join(toks[toks.etype == 'STATIC'].value)
    gen = toks[toks.etype.isin(['MUT', 'AMP', 'DEL', 'FUSION', 'TMB', 'PANEL', 'SAMPLE'])]
    print('  static:', stat); print('  genomic (day %s):' % sorted(set(gen.day.astype(int)))[:3], ' '.join(gen.token.head(14)), '...' if len(gen) > 14 else '')
    tl = toks[~toks.etype.isin(['STATIC', 'MUT', 'AMP', 'DEL', 'FUSION', 'TMB', 'PANEL', 'SAMPLE'])].tail(8)
    for _, x in tl.iterrows(): print('   d%5d %s' % (x.day, x.token))
    print('  [dropped as future: %d events, first at day %s]' % (len(fut), fut.day.min() if len(fut) else '-'))
