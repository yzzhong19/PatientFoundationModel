"""Biology checks on random5 (test predictions only, every landmark scored by the fold model that never saw that patient).
(1) Observed OS: STK11 / KEAP1 mutant vs tested wild-type in platinum-treated NSCLC; TMB high vs low in NSCLC on immunotherapy
    line >= 2. One landmark per patient (the first eligible one), KM medians + Cox HR (univariate and adjusted for line + stage).
(2) HSTU out-of-fold predicted 12-month risk in the same groups.
(3) In-silico flips with each fold's HSTU: edit only the genomic token in the visible prefix (add / remove MUT:<gene>; set every
    TMB token to the highest / lowest bucket) and record the change in predicted 12-month risk. Contrast groups: NSCLC not on
    platinum (STK11/KEAP1), NSCLC not on immunotherapy (TMB). Patient-level bootstrap CIs (B=1000).
Status at a landmark uses only samples sequenced on or before the landmark day. usage: python biology.py"""
import json, numpy as np, pandas as pd, torch
from lifelines import CoxPHFitter, KaplanMeierFitter
from data import Events, ETYPES, truncate
from simulate import dev, load_fold, token_id, arrays, add_token, remove_tokens, set_tokens, score

GENES = ['STK11', 'KEAP1']
TMB_HI, TMB_LO = 'TMB:>6', 'TMB:0-0.5'
rng = np.random.RandomState(0)
out_rows = []


def say(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()


log = open('biology.log', 'w')
E = Events(); lm = E.lm.copy()
nsclc = (lm.cohort == 'NSCLC').values
platinum = lm.reg_name.str.contains('platin').values
immuno = ((lm.dclass.values & 2) > 0)

# ---- genomic status at each landmark (1 mutated, 0 tested wild-type, -1 untested), from samples on/before the landmark day
gs = pd.read_pickle('genomics_status.pkl')
genes_sorted = sorted(pd.read_csv('top_genes.csv').gene)
gi = {g: genes_sorted.index(g) for g in GENES}
by_pat = {p: d for p, d in gs.groupby('patient_id')}
for g in GENES:
    st = np.full(len(lm), -1)
    for i, (p, day) in enumerate(zip(lm.patient_id.values, lm.landmark_day.values)):
        d = by_pat.get(p)
        if d is None: continue
        v = np.array([x[gi[g]] for x in d.genes.values])[d.day.values <= day]
        if len(v): st[i] = v.max()
    lm[g] = st
lm['STK11_KEAP1'] = np.where((lm.STK11 == 1) | (lm.KEAP1 == 1), 1, np.where((lm.STK11 == 0) & (lm.KEAP1 == 0), 0, -1))
# latest TMB bucket visible at the landmark
tmb = E.ev.etype.values == 'TMB'
lm['TMB'] = [next((E.tokens[j] for j in E.P[l][::-1] if tmb[j]), None) for l in lm.landmark_id.values]
lm['TMB_hi'] = lm.TMB.map({'TMB:>6': 1, 'TMB:3-6': 1, 'TMB:1.5-3': 0, 'TMB:0.5-1.5': 0, 'TMB:0-0.5': 0})

# ---- HSTU out-of-fold predictions
pred = pd.concat([pd.read_parquet(f'runs/HSTU/random5/fold{k}/predictions.parquet') for k in range(5)])
lm = lm.merge(pred[['landmark_id', 'fold', 'risk12']], on='landmark_id', how='left')
assert lm.risk12.notna().all()


def observed(name, sel, col):
    """first eligible landmark per patient; KM medians, Cox HR (univariate; adjusted for line + stage), mean HSTU risk12"""
    d = lm[sel & lm[col].isin([0, 1]).values].sort_values('landmark_day').groupby('patient_id').head(1)
    d = d.assign(x=d[col].astype(int), months=d.os_time / 30.44)
    res = dict(check=name, n0=int((d.x == 0).sum()), n1=int((d.x == 1).sum()), events=int(d.os_event.sum()))
    for v in (0, 1):
        km = KaplanMeierFitter().fit(d.months[d.x == v], d.os_event[d.x == v]); res[f'median_os_mo_{v}'] = float(km.median_survival_time_)
    for tag, cols in (('uni', ['x']), ('adj', ['x', 'line'])):
        X = d[['months', 'os_event'] + cols].copy()
        if tag == 'adj':
            X = X.join(pd.get_dummies(d.STAGE, prefix='st', drop_first=True, dtype=float))
        c = CoxPHFitter(penalizer=0.01).fit(X, 'months', 'os_event')
        res[f'HR_{tag}'] = float(np.exp(c.params_['x']))
        lo, hi = np.exp(c.confidence_intervals_.loc['x']); res[f'HR_{tag}_lo'], res[f'HR_{tag}_hi'] = float(lo), float(hi)
    res['hstu_risk12_0'], res['hstu_risk12_1'] = float(d.risk12[d.x == 0].mean()), float(d.risk12[d.x == 1].mean())
    say(json.dumps({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in res.items()}))
    out_rows.append(dict(part='observed', **res))


say('== (1)+(2) observed OS and HSTU out-of-fold risk; x = 1 is the mutant / TMB-high group')
for g in GENES + ['STK11_KEAP1']:
    observed(f'{g} mut vs WT, NSCLC platinum', nsclc & platinum, g)
    observed(f'{g} mut vs WT, NSCLC non-platinum (contrast)', nsclc & ~platinum, g)
observed('TMB high (>=3) vs low, NSCLC IO line>=2', nsclc & immuno & (lm.line.values >= 2), 'TMB_hi')
observed('TMB high (>=3) vs low, NSCLC non-IO (contrast)', nsclc & ~immuno, 'TMB_hi')


# ---- (3) in-silico flips
flips = []
for k in range(5):
    fold = f'fold{k}'; Fd, model = load_fold(E, fold)
    tid_of = lambda t: token_id(E, Fd, t)
    te = np.where(lm.patient_id.isin(set(Fd.test)).values & nsclc)[0]
    base = {i: arrays(E, Fd, truncate(E, E.P[lm.landmark_id.values[i]])) for i in te}
    for g in GENES:
        mut_id = tid_of(f'MUT:{g}'); assert mut_id >= len(ETYPES), f'MUT:{g} not in fold vocab'
        rm = [tid_of(t) for t in (f'MUT:{g}', f'DEL:{g}') if (E.tokens == t).any()]
        for direction, status in (('add (WT->mut)', 0), ('remove (mut->WT)', 1)):
            ids = [i for i in te if lm[g].values[i] == status]
            s0 = [base[i] for i in ids]
            s1 = [add_token(s, ETYPES.index('MUT'), mut_id) if status == 0 else remove_tokens(s, rm) for s in s0]
            if status == 1:                           # mutant by panel status but no removable MUT/DEL token: skip
                ok = [len(x['tok']) < len(y['tok']) for x, y in zip(s1, s0)]
                if not all(ok): say(f'{fold} {g}: {len(ok) - sum(ok)} mutant landmarks without a MUT/DEL token, skipped')
                ids, s0, s1 = [x for x, o in zip(ids, ok) if o], [x for x, o in zip(s0, ok) if o], [x for x, o in zip(s1, ok) if o]
            d = score(model, s1) - score(model, s0)
            sign = 1 if status == 0 else -1           # report as risk(mut) - risk(WT)
            flips += [dict(fold=fold, flip=f'{g} {direction}', group='platinum' if platinum[i] else 'non-platinum',
                           patient_id=lm.patient_id.values[i], delta=sign * x) for i, x in zip(ids, d)]
    hi, lo = tid_of(TMB_HI), tid_of(TMB_LO)
    ids = [i for i in te if lm.TMB.values[i] is not None]
    s0 = [base[i] for i in ids]
    d = score(model, [set_tokens(s, ETYPES.index('TMB'), hi) for s in s0]) - score(model, [set_tokens(s, ETYPES.index('TMB'), lo) for s in s0])
    flips += [dict(fold=fold, flip='TMB >6 vs 0-0.5', group='IO line>=2' if immuno[i] and lm.line.values[i] >= 2 else ('IO line 1' if immuno[i] else 'non-IO'),
                   patient_id=lm.patient_id.values[i], delta=x) for i, x in zip(ids, d)]
    # positive control: STAGE token set to Stage IV vs Stage I
    stage_ids = [tid_of(t) for t in sorted(set(E.tokens[E.is_static])) if t.startswith('STATIC:STAGE=')]
    def stage(s, t):
        s = {k: v.copy() for k, v in s.items()}; s['tok'][(s['static'] == 1) & np.isin(s['tok'], stage_ids)] = tid_of(t)
        return s
    s0 = [base[i] for i in te]
    d = score(model, [stage(s, 'STATIC:STAGE=Stage IV') for s in s0]) - score(model, [stage(s, 'STATIC:STAGE=Stage I') for s in s0])
    flips += [dict(fold=fold, flip='Stage IV vs I (positive control)', group='all NSCLC', patient_id=lm.patient_id.values[i], delta=x)
              for i, x in zip(te, d)]
    say(f'{fold}: flips done ({len(flips)} rows so far)')

fl = pd.DataFrame(flips); fl.to_parquet('biology_flips.parquet', index=False)
say('== (3) in-silico flips: mean change in HSTU 12-month risk (percentage points), risk(mut / TMB-high) - risk(WT / TMB-low)')
for (f, g), d in fl.groupby(['flip', 'group']):
    pm = d.groupby('patient_id').delta.mean().values
    bs = [rng.choice(pm, len(pm)).mean() for _ in range(1000)]
    r = dict(part='flip', check=f, group=g, n_patients=len(pm), n_landmarks=len(d), delta_pp=float(100 * pm.mean()),
             ci_lo=100 * np.percentile(bs, 2.5), ci_hi=100 * np.percentile(bs, 97.5), frac_positive=float((d.delta > 0).mean()))
    say(json.dumps({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}))
    out_rows.append(r)
pd.DataFrame(out_rows).to_csv('biology_results.csv', index=False)
say('wrote biology_results.csv, biology_flips.parquet')
