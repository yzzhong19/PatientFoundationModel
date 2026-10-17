"""Paired patient-level bootstrap of the C-index difference between two models on the same test landmarks.
usage: python paired_diff.py <B> <model_a> <model_b> <split> <pred files...>  (a - b, per fold and POOLED)"""
import sys, json, numpy as np, pandas as pd
from lifelines.utils import concordance_index
B, ma, mb, stype = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
lm = pd.read_parquet('landmarks.parquet').set_index('landmark_id')
P = pd.concat([pd.read_parquet(f) for f in sys.argv[5:]]); P = P[P.split == stype]
W = P.pivot_table(index=['fold', 'landmark_id'], columns='model', values='risk').dropna(subset=[ma, mb]).reset_index()
rows = []
for fold, g in list(W.groupby('fold')) + [('POOLED', W)]:
    d = lm.loc[g.landmark_id.values]; t, e = d.os_time.values, d.os_event.values
    up, inv = np.unique(d.patient_id.values, return_inverse=True); groups = [np.where(inv == i)[0] for i in range(len(up))]
    c = lambda ix, m: concordance_index(t[ix], -g[m].values[ix], e[ix])
    allix = np.arange(len(g)); point = c(allix, ma) - c(allix, mb)
    rng = np.random.RandomState(1); bs = []
    for _ in range(B):
        ix = np.concatenate([groups[i] for i in rng.randint(0, len(up), len(up))]); bs.append(c(ix, ma) - c(ix, mb))
    bs = np.array(bs)
    rows.append(dict(split=stype, fold=fold, a=ma, b=mb, diff=point, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5), p_le0=(bs <= 0).mean(), n_landmarks=len(g)))
out = pd.DataFrame(rows); print(out.round(4).to_string(index=False))
out.to_csv(f'paired_{ma}_vs_{mb}_{stype}.csv', index=False)
