"""Development reference: GBSA on token counts (same features/settings as baselines.py) fit on the inner-train patients and
scored on the same fixed 10% inner-validation patients that train.py uses for early stopping. Test patients are not touched.
usage: python gbsa_ref.py random5 -> gbsa_ref_<split>.csv"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sps
from joblib import Parallel, delayed
from lifelines.utils import concordance_index
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from common import load, prefix_index
ev, lm, sp = load()
ev = ev.sort_values(['patient_id', 'day', 'rank'], kind='stable').reset_index(drop=True)
P = prefix_index(ev, lm); lm = lm.reset_index(drop=True)
y = Surv.from_arrays(lm.os_event.astype(bool).values, lm.os_time.values)
vc = ev.token.value_counts(); vocab = {t: i for i, t in enumerate(vc[vc >= 30].index)}
tid = ev.token.map(vocab).fillna(-1).astype(int).values
rows, cols = [], []
for k, r in enumerate(lm.landmark_id.values):
    t = tid[P[r]]; t = t[t >= 0]; rows += [k] * len(t); cols += list(t)
X = sps.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lm), len(vocab))); X.sum_duplicates(); X = X.toarray()

def split_patients(ps, frac=0.1):
    ps = np.array(sorted(ps)); np.random.RandomState(12345).shuffle(ps); k = max(1, int(len(ps) * frac))
    return list(ps[k:]), list(ps[:k])

def run(stype, f):
    tr_p, va_p = split_patients(f['train'])
    tr = np.where(lm.patient_id.isin(set(tr_p)))[0]; va = np.where(lm.patient_id.isin(set(va_p)))[0]
    g = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.7, min_samples_leaf=20,
                                         max_features='sqrt', random_state=0).fit(X[tr], y[tr])
    c = concordance_index(lm.os_time.values[va], -g.predict(X[va]), lm.os_event.values[va])
    print(stype, f['name'], round(c, 4), flush=True)
    return dict(split=stype, fold=f['name'], gbsa_val_c=c, n_val=len(va))

stype = sys.argv[1]
res = Parallel(n_jobs=2)(delayed(run)(stype, f) for f in sp[stype])
pd.DataFrame(res).to_csv(f'gbsa_ref_{stype}.csv', index=False)
