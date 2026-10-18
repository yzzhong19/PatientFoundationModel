"""Raw input token embeddings (e_tok) of the random5 HSTU models: fold stability, neighbours, probes. Run from the repo root: python analysis/emb_tokens_raw.py"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Events, Fold, ETYPES, build_vocab
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr
E = Events(); ev = E.ev
base = len(ETYPES)
# per fold: token -> embedding (fine-tuned models)
embs = []
for f in range(5):
    Fd = Fold(E, 'random5', f'fold{f}')
    W = torch.load(f'runs/HSTU/random5/fold{f}/model.pt', map_location='cpu')['e_tok.weight'].float().numpy()
    toks = pd.Series(E.tokens).groupby(Fd.tid).first()
    toks = toks[toks.index >= base]
    embs.append(pd.DataFrame(W[toks.index], index=toks.values))
common = sorted(set.intersection(*[set(e.index) for e in embs]))
def cos(M): M = M / np.linalg.norm(M, axis=1, keepdims=True); return M @ M.T
S = np.mean([cos(e.loc[common].values) for e in embs], 0)        # fold-averaged cosine sim
# fold stability: spearman of pairwise sims between folds
iu = np.triu_indices(len(common), 1)
Ss = [cos(e.loc[common].values)[iu] for e in embs]
print(f'{len(common)} tokens in all 5 folds; between-fold Spearman of pairwise similarity: ' +
      ' '.join(f'{spearmanr(Ss[0], Ss[k])[0]:.2f}' for k in range(1, 5)))
et = np.array([t.split(':')[0] for t in common]); ci = {t: i for i, t in enumerate(common)}
# 1) same-type neighbours
nn = np.argsort(-S, 1)[:, 1:6]
print('top-5 neighbours same event type: %.2f (chance %.2f)' % ((et[nn] == et[:, None]).mean(),
      (pd.Series(et).value_counts(normalize=True) ** 2).sum()))
# 2) probes
for p in ['MET:liver', 'MET:brain', 'MD:progressing', 'SCAN:improving', 'STATIC:STAGE=Stage IV', 'MUT:KRAS', 'MUT:EGFR', 'MUT:STK11',
          'AMP:ERBB2', 'REG_START:pembrolizumab', 'REG_START:carboplatin+pemetrexed', 'REG_START:osimertinib', 'TM:PSA=>20x']:
    if p in ci:
        j = np.argsort(-S[ci[p]])[1:6]; print(f'{p:34s} -> ' + ', '.join(f'{common[k]} ({S[ci[p], k]:.2f})' for k in j))
# 3) regimens: similarity vs shared drug class
dc = ev.groupby('token').dclass.first()
R = [t for t in common if t.startswith('REG_START:') and t != 'REG_START:investigational']
d = dc.loc[R].values; Ri = [ci[t] for t in R]; sub = S[np.ix_(Ri, Ri)]; m = ~np.eye(len(R), dtype=bool)
same = (d[:, None] == d[None, :]) & m
print(f'REG_START ({len(R)}): mean cos same drug-class combo {sub[same].mean():.3f} vs different {sub[m & ~same].mean():.3f}')
# cancer specificity: regimen used mostly in one cohort
coh = ev[ev.token.isin(R)].groupby('token').cohort.agg(lambda x: x.value_counts().index[0]).loc[R].values
same = (coh[:, None] == coh[None, :]) & m
print(f'REG_START: mean cos same main cohort {sub[same].mean():.3f} vs different {sub[m & ~same].mean():.3f}')
# 4) genes: similarity vs co-occurrence (patient level)
G = [t for t in common if t.startswith('MUT:')]
X = pd.crosstab(ev.patient_id[ev.token.isin(G)], ev.token[ev.token.isin(G)]).clip(upper=1)[G].values.astype(float)
co = (X.T @ X); n = X.sum(0); jac = co / (n[:, None] + n[None, :] - co)
Gi = [ci[t] for t in G]; gi = np.triu_indices(len(G), 1)
print(f'MUT ({len(G)}): Spearman(embedding cos, co-mutation Jaccard) = {spearmanr(S[np.ix_(Gi, Gi)][gi], jac[gi])[0]:.2f}')
gcoh = ev[ev.token.isin(G)].groupby('token').cohort.agg(lambda x: x.value_counts(normalize=True).max()).loc[G]
# 5) prognosis probe: token-level observed death rate at first landmark, predicted from embedding (5-fold CV ridge)
lm = E.lm.sort_values('landmark_day').drop_duplicates('patient_id')
first = lm.set_index('patient_id')
seen = ev.merge(first[['landmark_day']], left_on='patient_id', right_index=True)
seen = seen[seen.day <= seen.landmark_day].drop_duplicates(['patient_id', 'token'])
y = seen.groupby('token').patient_id.apply(lambda p: np.log(first.loc[p].os_event.sum() + .5) - np.log(first.loc[p].os_time.sum()))
npt = seen.groupby('token').patient_id.size()
T = [t for t in common if t in y.index and npt[t] >= 30]
Xe = np.mean([e.loc[T].values for e in embs], 0); yy = y.loc[T].values
for name, mask in [('all', np.ones(len(T), bool))] + [(k, np.array([t.startswith(k + ':') for t in T])) for k in ['REG_START', 'MUT', 'STATIC', 'TM']]:
    if mask.sum() < 20: continue
    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 12)), Xe[mask], yy[mask], cv=5)
    print(f'prognosis probe {name:9s} n={mask.sum():4d}  Spearman(pred, observed log death rate) = {spearmanr(pr, yy[mask])[0]:.2f}')
