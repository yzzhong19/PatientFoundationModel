"""Run from the repo root: python analysis/emb_patients.py (writes figs/). Out-of-fold patient embeddings at the first eligible landmark; 5 fold models aligned by orthogonal Procrustes to fold0."""
import os, sys, json, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Events, truncate
from simulate import load_fold, dev
from scipy.linalg import orthogonal_procrustes
E = Events(); lm = E.lm
first = lm.sort_values(['patient_id', 'landmark_day']).drop_duplicates('patient_id').reset_index(drop=True)
seqs = [truncate(E, E.P[l]) for l in first.landmark_id]
Z, R, test_of = [], [], {}
for f in range(5):
    Fd, model = load_fold(E, f'fold{f}')
    for p in Fd.test: test_of[p] = f
    zs, rs = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), 256):
            b = {k: v.to(dev) for k, v in Fd.pack(seqs[i:i + 256]).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z = model.landmark_repr(b).float(); lg = model.h_surv(z).float()
            zs.append(z.cpu().numpy()); rs.append((1 - torch.exp(torch.cumsum(torch.nn.functional.logsigmoid(-lg), 1)))[:, 11].cpu().numpy())
    Z.append(np.concatenate(zs)); R.append(np.concatenate(rs))
mu = [z.mean(0) for z in Z]
for f in range(1, 5):
    Q, _ = orthogonal_procrustes(Z[f] - mu[f], Z[0] - mu[0]); Z[f] = (Z[f] - mu[f]) @ Q + mu[0]
fo = first.patient_id.map(test_of).values
print('patients', len(first), 'with test fold', np.isfinite(fo.astype(float)).sum())
# agreement of aligned spaces: correlation between fold0 and fold f embeddings of the same patient
print('Procrustes alignment R^2 to fold0:', ' '.join(f'{1 - ((Z[f]-Z[0])**2).sum()/((Z[0]-mu[0])**2).sum():.2f}' for f in range(1, 5)))
emb = np.stack([Z[int(f)][i] for i, f in enumerate(fo)]); risk = np.array([R[int(f)][i] for i, f in enumerate(fo)])
np.save('figs/patient_emb.npy', emb); first.assign(fold=fo, risk12=risk).to_parquet('figs/patient_emb_meta.parquet')
