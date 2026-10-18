"""Subtype probes and within-cohort k-means of the patient embeddings from analysis/emb_patients.py. Run from the repo root."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import Events
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import fisher_exact, spearmanr
E = Events(); emb = np.load('figs/patient_emb.npy'); M = pd.read_parquet('figs/patient_emb_meta.parquet')
X = StandardScaler().fit_transform(emb)
# tokens seen before the landmark (presence)
toks = [set(E.tokens[E.P[l]]) for l in M.landmark_id]
def has(pred): return np.array([any(pred(t) for t in s) for s in toks])
pc = PCA(20).fit(X); P = pc.transform(X)
print('PCA var explained (first 5):', np.round(pc.explained_variance_ratio_[:5], 3))
print('Spearman(PC1, predicted 12m risk) = %.2f' % spearmanr(P[:, 0], M.risk12)[0])
r2 = 1 - sum(((P[M.cohort == c] - P[M.cohort == c].mean(0)) ** 2).sum() for c in M.cohort.unique()) / ((P - P.mean(0)) ** 2).sum()
print('share of top-20-PC variance explained by cohort: %.2f' % r2)
# ---- subtype probes: can a linear probe on the embedding recover marker status? (5-fold CV AUC, within cohort)
probes = {'NSCLC': {'EGFR mut': lambda t: t == 'MUT:EGFR', 'KRAS mut': lambda t: t == 'MUT:KRAS', 'STK11 mut': lambda t: t == 'MUT:STK11',
                    'ALK fusion': lambda t: t == 'FUSION:ALK', 'squamous': lambda t: t == 'STATIC:HIST=Squamous cell', 'prior IO': lambda t: any(d in t for d in ('pembrolizumab', 'nivolumab', 'atezolizumab', 'durvalumab')) and t.startswith('REG_')},
          'Breast': {'ERBB2 amp': lambda t: t == 'AMP:ERBB2', 'prior anti-HER2': lambda t: t.startswith('REG_') and 'trastuzumab' in t, 'prior endocrine': lambda t: t.startswith('REG_') and any(d in t for d in ('letrozole', 'anastrozole', 'tamoxifen', 'fulvestrant', 'exemestane')),
                     'PIK3CA mut': lambda t: t == 'MUT:PIK3CA', 'TP53 mut': lambda t: t == 'MUT:TP53'},
          'CRC': {'KRAS mut': lambda t: t == 'MUT:KRAS', 'BRAF mut': lambda t: t == 'MUT:BRAF', 'TMB >6 (MSI-like)': lambda t: t == 'TMB:>6', 'APC mut': lambda t: t == 'MUT:APC'},
          'Prostate': {'AR amp': lambda t: t == 'AMP:AR', 'TMPRSS2 fusion': lambda t: t == 'FUSION:TMPRSS2', 'prior ARSI': lambda t: t.startswith('REG_') and any(d in t for d in ('abiraterone', 'enzalutamide'))},
          'Pancreatic': {'KRAS mut': lambda t: t == 'MUT:KRAS', 'SMAD4 alt': lambda t: t in ('MUT:SMAD4', 'DEL:SMAD4')},
          'Bladder': {'FGFR3 mut': lambda t: t == 'MUT:FGFR3', 'TMB >6': lambda t: t == 'TMB:>6'}}
print('\nLinear probe AUC on embedding (5-fold CV, within cohort):')
for c, d in probes.items():
    ix = np.where(M.cohort == c)[0]
    for name, f in d.items():
        y = has(f)[ix]
        if y.sum() < 15 or (~y).sum() < 15: continue
        p = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000), X[ix], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
        print(f'  {c:10s} {name:18s} prevalence {y.mean():.2f}  AUC {roc_auc_score(y, p):.2f}')
# ---- clusters within cohort
vocab = sorted({t for s in toks for t in s})
rows = []
for c in ['NSCLC', 'CRC', 'Breast', 'Pancreatic', 'Prostate', 'Bladder']:
    ix = np.where(M.cohort == c)[0]; Pc = PCA(10, random_state=0).fit_transform(X[ix])
    sil = {k: silhouette_score(Pc, KMeans(k, n_init=10, random_state=0).fit_predict(Pc)) for k in range(2, 7)}
    k = max(sil, key=sil.get); lab = KMeans(k, n_init=10, random_state=0).fit_predict(Pc)
    # stability: recluster with different seeds / on half samples
    ari = np.mean([adjusted_rand_score(lab, KMeans(k, n_init=10, random_state=s).fit_predict(Pc)) for s in range(1, 6)])
    d = M.iloc[ix].assign(cl=lab)
    lr = multivariate_logrank_test(d.os_time, d.cl, d.os_event).p_value
    # does cluster add beyond predicted risk?
    cx = pd.get_dummies(d.cl, prefix='cl', drop_first=True).astype(float).assign(r=np.log(d.risk12 / (1 - d.risk12)), T=d.os_time, Ev=d.os_event)
    base = CoxPHFitter().fit(cx[['r', 'T', 'Ev']], 'T', 'Ev'); full = CoxPHFitter().fit(cx, 'T', 'Ev')
    from scipy.stats import chi2
    p_add = chi2.sf(2 * (full.log_likelihood_ - base.log_likelihood_), k - 1)
    print(f'\n=== {c}: n={len(ix)}  k={k} (silhouette {sil[k]:.2f}; seed ARI {ari:.2f})  logrank p={lr:.1e}  cluster beyond HSTU risk p={p_add:.2f}')
    T = np.array([[t in s for t in vocab] for s in np.array(toks, dtype=object)[ix]])
    keep = T.mean(0) >= 0.03
    for g in sorted(d.cl.unique(), key=lambda g: d.risk12[d.cl == g].mean()):
        m = (d.cl == g).values; km = KaplanMeierFitter().fit(d.os_time[m] / 30.44, d.os_event[m])
        a, b = T[m][:, keep].sum(0), T[~m][:, keep].sum(0); n1, n0 = m.sum(), (~m).sum()
        lo = np.log((a + .5) / (n1 - a + .5)) - np.log((b + .5) / (n0 - b + .5))
        top = np.argsort(-lo)[:6]; vk = np.array(vocab)[keep]
        desc = ', '.join(f'{vk[j].replace("REG_START:", "start:").replace("STATIC:", "")} {a[j] / n1:.0%} vs {b[j] / n0:.0%}' for j in top if a[j] / n1 > b[j] / n0 + .05)
        print(f'  cl{g}: n={n1:4d} line {d.line[m].median():.0f}  pred12 {d.risk12[m].mean():.2f}  median OS {km.median_survival_time_:.1f} mo | {desc}')
        rows.append(dict(cohort=c, cluster=g, n=n1, median_os=km.median_survival_time_, risk12=d.risk12[m].mean(), top=desc))
pd.DataFrame(rows).to_csv('figs/patient_clusters.csv', index=False)
