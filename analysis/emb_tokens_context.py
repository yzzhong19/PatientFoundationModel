"""Contextual token embeddings (mean hidden state per token) of the random5 HSTU models: fold stability and neighbours. Run from the repo root: python analysis/emb_tokens_context.py"""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Events, ETYPES
from simulate import load_fold, dev
from scipy.stats import spearmanr
E = Events(); ev = E.ev; base = len(ETYPES)
sd = torch.load('runs/HSTU/random5/fold0/model.pt', map_location='cpu')
for k in ['e_tok.weight', 'e_type.weight', 'e_coh.weight', 'e_day.weight', 'e_cls.weight']:
    print(f'{k:14s} mean row norm {sd[k].float().norm(dim=-1 if k != "e_cls.weight" else 0).mean():.2f}')
embs = []
for f in range(5):
    Fd, model = load_fold(E, f'fold{f}')
    seqs = Fd.pretrain_seqs(Fd.pretrain, np.random.RandomState(0))
    H = np.zeros((Fd.V, 128)); N = np.zeros(Fd.V)
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            b = {k: v.to(dev) for k, v in Fd.pack(seqs[i:i + 128]).items()}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = model.encode(b).float()
            m = (torch.arange(b['tok'].shape[1], device=dev)[None] < b['len'][:, None])
            t = b['tok'][m].cpu().numpy(); hh = h[m].cpu().numpy()
            np.add.at(H, t, hh); np.add.at(N, t, 1)
    toks = pd.Series(E.tokens).groupby(Fd.tid).first(); toks = toks[(toks.index >= base)]
    ok = toks.index[N[toks.index] >= 20]
    embs.append(pd.DataFrame(H[ok] / N[ok, None], index=toks.loc[ok].values))
common = sorted(set.intersection(*[set(e.index) for e in embs]))
def cos(M): M = M - M.mean(0); M = M / np.linalg.norm(M, axis=1, keepdims=True); return M @ M.T
iu = np.triu_indices(len(common), 1); Ss = [cos(e.loc[common].values) for e in embs]
print(f'\nCONTEXTUAL: {len(common)} tokens; between-fold Spearman of pairwise sims: ' + ' '.join(f'{spearmanr(Ss[0][iu], Ss[k][iu])[0]:.2f}' for k in range(1, 5)))
S = np.mean(Ss, 0); et = np.array([t.split(':')[0] for t in common]); ci = {t: i for i, t in enumerate(common)}
nn = np.argsort(-S, 1)[:, 1:6]
print('top-5 neighbours same event type: %.2f (chance %.2f)' % ((et[nn] == et[:, None]).mean(), (pd.Series(et).value_counts(normalize=True) ** 2).sum()))
for p in ['MET:liver', 'MET:brain', 'MD:progressing', 'SCAN:improving', 'STATIC:STAGE=Stage IV', 'MUT:KRAS', 'MUT:EGFR', 'MUT:STK11',
          'AMP:ERBB2', 'REG_START:pembrolizumab', 'REG_START:carboplatin+pemetrexed', 'REG_START:osimertinib', 'REG_START:leuprolide', 'TM:PSA=>20x']:
    if p in ci:
        j = np.argsort(-S[ci[p]])[1:6]; print(f'{p:34s} -> ' + ', '.join(f'{common[k]} ({S[ci[p], k]:.2f})' for k in j))
dc = ev.groupby('token').dclass.first()
R = [t for t in common if t.startswith('REG_START:') and t != 'REG_START:investigational']; Ri = [ci[t] for t in R]
sub = S[np.ix_(Ri, Ri)]; m = ~np.eye(len(R), dtype=bool); d = dc.loc[R].values
coh = ev[ev.token.isin(R)].groupby('token').cohort.agg(lambda x: x.value_counts().index[0]).loc[R].values
for lab, g in [('drug-class combo', d), ('main cohort', coh)]:
    same = (g[:, None] == g[None, :]) & m; print(f'REG_START ({len(R)}): cos same {lab} {sub[same].mean():.3f} vs different {sub[m & ~same].mean():.3f}')
G = [t for t in common if t.startswith('MUT:')]
X = pd.crosstab(ev.patient_id[ev.token.isin(G)], ev.token[ev.token.isin(G)]).clip(upper=1)[G].values.astype(float)
co = X.T @ X; n = X.sum(0); jac = co / (n[:, None] + n[None, :] - co); Gi = [ci[t] for t in G]; gi = np.triu_indices(len(G), 1)
print(f'MUT ({len(G)}): Spearman(cos, co-mutation Jaccard) = {spearmanr(S[np.ix_(Gi, Gi)][gi], jac[gi])[0]:.2f}')
gc = ev[ev.token.isin(G)].groupby('token').cohort.agg(lambda x: x.value_counts().index[0]).loc[G].values
same = (gc[:, None] == gc[None, :]) & ~np.eye(len(G), dtype=bool); sg = S[np.ix_(Gi, Gi)]
print(f'MUT: cos same main cohort {sg[same].mean():.3f} vs different {sg[~same & ~np.eye(len(G), dtype=bool)].mean():.3f}')
