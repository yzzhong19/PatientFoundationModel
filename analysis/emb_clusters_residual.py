"""Probe confounding by targeted therapy, and clusters after regressing out predicted risk and line. Needs analysis/emb_patients.py output."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Events
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
E = Events(); emb = np.load('figs/patient_emb.npy'); M = pd.read_parquet('figs/patient_emb_meta.parquet')
X = StandardScaler().fit_transform(emb); toks = [set(E.tokens[E.P[l]]) for l in M.landmark_id]
def has(f): return np.array([any(f(t) for t in s) for s in toks])
def probe(ix, y):
    p = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000), X[ix], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
    return roc_auc_score(y, p)
reg = lambda ds: (lambda t: t.startswith('REG_') and any(d in t for d in ds))
tki = has(reg(('osimertinib', 'erlotinib', 'gefitinib', 'afatinib', 'dacomitinib', 'crizotinib', 'alectinib', 'brigatinib', 'lorlatinib', 'ceritinib')))
her2 = has(reg(('trastuzumab', 'pertuzumab', 'lapatinib', 'neratinib', 'tucatinib', 'ado-trastuzumab')))
print('Probe AUC, all vs among patients never given the matching targeted drug:')
for c, mk, excl, lab in [('NSCLC', 'MUT:EGFR', tki, 'TKI-naive'), ('NSCLC', 'FUSION:ALK', tki, 'TKI-naive'), ('Breast', 'AMP:ERBB2', her2, 'anti-HER2-naive')]:
    y = has(lambda t: t == mk); a = np.where(M.cohort == c)[0]; b = a[~excl[a]]
    print(f'  {c} {mk}: all AUC {probe(a, y[a]):.2f} (n={len(a)}, pos {y[a].sum()}) | {lab} AUC ' +
          (f'{probe(b, y[b]):.2f} (n={len(b)}, pos {y[b].sum()})' if y[b].sum() >= 15 else f'n/a (pos {y[b].sum()})'))
print('\nClusters after regressing out predicted risk and line (within cohort):')
for c in ['NSCLC', 'Breast', 'CRC']:
    ix = np.where(M.cohort == c)[0]; d = M.iloc[ix]
    C = np.c_[np.log(d.risk12 / (1 - d.risk12)), d.line, d.line ** 2]
    Xr = X[ix] - LinearRegression().fit(C, X[ix]).predict(C)
    Pc = PCA(10, random_state=0).fit_transform(Xr)
    sil = {k: silhouette_score(Pc, KMeans(k, n_init=10, random_state=0).fit_predict(Pc)) for k in range(2, 7)}
    k = max(sil, key=sil.get); lab = KMeans(k, n_init=10, random_state=0).fit_predict(Pc)
    lr = multivariate_logrank_test(d.os_time, lab, d.os_event).p_value
    print(f'=== {c}: k={k} silhouette {sil[k]:.2f} logrank p={lr:.1e}')
    vocab = np.array(sorted({t for s in np.array(toks, dtype=object)[ix] for t in s}))
    T = np.array([[t in s for t in vocab] for s in np.array(toks, dtype=object)[ix]]); keep = T.mean(0) >= .03
    for g in range(k):
        m = lab == g; km = KaplanMeierFitter().fit(d.os_time[m] / 30.44, d.os_event[m])
        a, b = T[m][:, keep].sum(0), T[~m][:, keep].sum(0)
        lo = np.log((a + .5) / (m.sum() - a + .5)) - np.log((b + .5) / ((~m).sum() - b + .5)); vk = vocab[keep]
        desc = ', '.join(f'{vk[j].replace("REG_START:", "start:").replace("STATIC:", "")} {a[j]/m.sum():.0%} vs {b[j]/(~m).sum():.0%}' for j in np.argsort(-lo)[:6] if a[j]/m.sum() > b[j]/(~m).sum() + .05)
        print(f'  cl{g}: n={m.sum():4d} pred12 {d.risk12[m].mean():.2f} median OS {km.median_survival_time_:.1f} mo | {desc}')
