"""Step 6 evaluation: C-index + time-dependent AUC (12, 24 months) with patient-level bootstrap 95% CIs.
usage: python eval.py <B> <pred.parquet ...>   -> results.csv
Rows: model x split x fold(held-out group, or POOLED across folds; pooled mixes per-fold models) x metric.
IPCW censoring reference = that fold's training landmarks (POOLED: all landmarks)."""
import sys, json, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv
warnings.filterwarnings('ignore')
T12, T24 = 365.25, 730.5
lm = pd.read_parquet('landmarks.parquet').set_index('landmark_id')
sp = json.load(open('splits.json'))


def metrics(t, e, risk, ytr, r12=None, r24=None):
    out = {}
    out['cindex'] = concordance_index(t, -risk, e)
    ttr_max = ytr['time'].max()
    m = t < ttr_max
    yte = Surv.from_arrays(e[m].astype(bool), t[m])
    for name, h, rr in (('auc12', T12, r12), ('auc24', T24, r24)):
        sc = risk if rr is None else rr   # optional horizon-specific risk (e.g. 1 - S(12m) from the hazard head)
        try:
            out[name] = cumulative_dynamic_auc(ytr, yte, sc[m], [h])[0][0]
        except Exception:
            out[name] = np.nan
    return out


def job(args):
    model, stype, fold, ids, trids, B, seed = args
    d = lm.loc[ids]; t = d.os_time.values; e = d.os_event.values; r, r12, r24 = risk_of[(model, stype, fold)]
    dtr = lm.loc[trids]; ytr = Surv.from_arrays(dtr.os_event.astype(bool).values, dtr.os_time.values)
    pt = d.patient_id.values; up, inv = np.unique(pt, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(up))]
    g12 = (lambda a, ix=slice(None): None if a is None else a[ix])
    point = metrics(t, e, r, ytr, r12, r24)
    rng = np.random.RandomState(seed); boots = {k: [] for k in point}
    for _ in range(B):
        ix = np.concatenate([groups[i] for i in rng.randint(0, len(up), len(up))])
        mm = metrics(t[ix], e[ix], r[ix], ytr, g12(r12, ix), g12(r24, ix))
        for k in point: boots[k].append(mm[k])
    rows = []
    for k, v in point.items():
        b = np.array(boots[k], dtype=float)
        lo, hi = (np.nanpercentile(b, [2.5, 97.5]) if B and np.isfinite(b).any() else (np.nan, np.nan))
        rows.append(dict(model=model, split=stype, fold=fold, metric=k, value=v, ci_lo=lo, ci_hi=hi, n_landmarks=len(d), n_patients=len(up), n_boot=B))
    return rows


risk_of = {}
if __name__ == '__main__':
    B = int(sys.argv[1]); files = sys.argv[2:]
    P = pd.concat([pd.read_parquet(f) for f in files])
    tasks = []
    for (model, stype), g in P.groupby(['model', 'split']):
        allids = lm.index.values
        for fold, h in g.groupby('fold'):
            risk_of[(model, stype, fold)] = (h.risk.values, h.risk12.values if 'risk12' in h and h.risk12.notna().all() else None, h.risk24.values if 'risk24' in h and h.risk24.notna().all() else None)
            f = [x for x in sp[stype] if x['name'] == fold][0]
            trids = lm.index[lm.patient_id.isin(set(f['train']))].values
            tasks.append((model, stype, fold, h.landmark_id.values, trids, B, 1))
        gs_ = g.sort_values('landmark_id')
        risk_of[(model, stype, 'POOLED')] = (gs_.risk.values, gs_.risk12.values if 'risk12' in gs_ and gs_.risk12.notna().all() else None, gs_.risk24.values if 'risk24' in gs_ and gs_.risk24.notna().all() else None)
        tasks.append((model, stype, 'POOLED', np.sort(g.landmark_id.unique()), allids, B, 1))
    with Pool(4) as pool:
        res = pool.map(job, tasks, chunksize=1)
    out = pd.DataFrame([r for rr in res for r in rr])
    out.to_csv('results.csv', index=False)
    print(out[out.fold == 'POOLED'].pivot_table(index=['split', 'model'], columns='metric', values='value').round(3))
