"""Aggregate-only figures (no per-patient points) -> plots_public/. Run from the repo root after analysis/emb_patients.py,
analysis/emb_plots.py (regimen map) and the evaluation scripts.
1 perf_cindex.png: pooled C-index per model and split; 2 forest_hstu_vs_gbsa.png: paired HSTU - GBSA per held-out group;
3 km_clusters_by_cohort.png: Kaplan-Meier by within-cohort embedding cluster; 4 breast_subtypes.png: residual breast
clusters, KM + defining features; 5 regimen_map.png: copy of the regimen token map (token level, not patients)."""
import os, sys, shutil, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from data import Events
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

os.makedirs('plots_public', exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
PAL = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#72B7B2']
COH = ['NSCLC', 'CRC', 'Breast', 'Pancreatic', 'Prostate', 'Bladder']

# ---- 1 pooled C-index
R = pd.concat([pd.read_csv(f) for f in ['results_final/results_baselines.csv', 'results_final/results_hstu_delphi_B200.csv',
                                        'results_final/results_rsf_random5_B200.csv']])
R = R[(R.fold == 'POOLED') & (R.metric == 'cindex')]
models = ['Cox_clinical', 'Cox_clinical_genomics', 'GBSA_counts', 'RSF_counts', 'Delphi', 'HSTU']
lab = {'Cox_clinical': 'Cox clinical', 'Cox_clinical_genomics': 'Cox + genomics', 'GBSA_counts': 'GBSA', 'RSF_counts': 'RSF',
       'Delphi': 'Delphi', 'HSTU': 'HSTU'}
col = dict(zip(models, ['#BDBDBD', '#9E9E9E', '#72B7B2', '#4C9F9A', '#B279A2', '#E45756']))
splits = [('random5', 'Random 5-fold'), ('loio', 'Unseen institution'), ('loco', 'Unseen cancer type')]
fig, ax = plt.subplots(figsize=(10, 4.2)); w = 0.13
for j, m in enumerate(models):
    for i, (s, _) in enumerate(splits):
        r = R[(R.model == m) & (R.split == s)]
        if len(r):
            r = r.iloc[0]; x = i + (j - 2.5) * w
            ax.bar(x, r.value, w, color=col[m], label=lab[m] if i == 0 or (m in ('Delphi',) and i == 1) else None)
            ax.errorbar(x, r.value, [[r.value - r.ci_lo], [r.ci_hi - r.value]], color='k', lw=.8, capsize=2)
            if m == 'HSTU': ax.text(x, r.ci_hi + .004, f'{r.value:.3f}', ha='center', fontsize=7.5, fontweight='bold')
ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in splits]); ax.set_ylim(.55, .78); ax.set_ylabel('pooled C-index (95% CI)')
h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend([u[lab[m]] for m in models if lab[m] in u], [lab[m] for m in models if lab[m] in u],
                                                                       ncol=6, frameon=False, fontsize=8, loc='upper center')
ax.set_title('Overall survival at each regimen start: 10,573 landmarks, 4,065 patients (RSF random split only; Delphi not run on unseen cancer)', fontsize=9)
fig.tight_layout(); fig.savefig('plots_public/perf_cindex.png'); plt.close(fig)

# ---- 2 forest plot HSTU - GBSA
P = pd.concat([pd.read_csv(f'results_final/paired_HSTU_vs_GBSA_counts_{s}.csv') for s in ('random5', 'loio', 'loco')])
P = P[(P.split != 'random5') | (P.fold == 'POOLED')]
rows = [('Random 5-fold (pooled)', P[(P.split == 'random5')].iloc[0])]
for s, t in (('loio', 'Unseen institution'), ('loco', 'Unseen cancer')):
    for _, r in P[P.split == s].iterrows():
        rows.append((f'{t}: {"pooled" if r.fold == "POOLED" else r.fold}' + (f' (n={r.n_landmarks:,})' if r.fold != 'POOLED' else ''), r))
fig, ax = plt.subplots(figsize=(7.5, 5.6))
for k, (name, r) in enumerate(rows[::-1]):
    pooled = 'pooled' in name
    ax.errorbar(r['diff'], k, xerr=[[r['diff'] - r.ci_lo], [r.ci_hi - r['diff']]], fmt='D' if pooled else 'o', color='#E45756' if pooled else '#4C78A8',
                ms=6 if pooled else 4, capsize=2)
    ax.text(.125, k, f'{r["diff"]:+.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]', va='center', fontsize=7.5)
ax.axvline(0, color='k', lw=.8); ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for n, _ in rows[::-1]], fontsize=8)
ax.set_xlim(-.01, .175); ax.set_xlabel('C-index difference, HSTU minus GBSA (paired patient bootstrap, B=1,000)')
ax.set_title('HSTU beats GBSA in every held-out institution and cancer type')
fig.tight_layout(); fig.savefig('plots_public/forest_hstu_vs_gbsa.png'); plt.close(fig)

# ---- 3 / 4 cluster survival curves (same clustering as analysis/emb_plots.py)
E = Events(); emb = np.load('figs/patient_emb.npy'); M = pd.read_parquet('figs/patient_emb_meta.parquet')
X = StandardScaler().fit_transform(emb); toks = [set(E.tokens[E.P[l]]) for l in M.landmark_id]


def km_panel(ax, d, lab, names):
    for g in sorted(set(lab)):
        m = lab == g; k = KaplanMeierFitter().fit(d.os_time.values[m] / 30.44, d.os_event.values[m])
        ax.step(k.timeline, k.survival_function_.values[:, 0], where='post', color=PAL[g % 6], lw=1.6, label=f'{names[g]} (n={m.sum()})')
    p = multivariate_logrank_test(d.os_time, lab, d.os_event).p_value
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02); ax.set_xlabel('months from landmark'); ax.set_ylabel('overall survival')
    ax.text(.98, .97, f'log-rank p={p:.1e}', transform=ax.transAxes, ha='right', va='top', fontsize=8); ax.legend(fontsize=7, frameon=False, loc='lower left')


NAMES = {'Breast': ['early / curative-intent', 'metastatic, heavily treated'],
         'Prostate': ['hormone-sensitive, PSA normal', 'intermediate', 'post-taxane CRPC'],
         'Bladder': ['non-muscle-invasive (BCG)', 'intermediate', 'progressing on chemo'],
         'CRC': ['curative-intent / resected', 'de novo metastatic, early', 'refractory metastatic']}
fig, ax = plt.subplots(2, 3, figsize=(13, 7.5))
for i, c in enumerate(COH):
    ix = np.where(M.cohort == c)[0]; d = M.iloc[ix]; P10 = PCA(10, random_state=0).fit_transform(X[ix])
    sil = {k: silhouette_score(P10, KMeans(k, n_init=10, random_state=0).fit_predict(P10)) for k in range(2, 7)}
    lb = KMeans(max(sil, key=sil.get), n_init=10, random_state=0).fit_predict(P10)
    o = np.argsort([d.risk12.values[lb == g].mean() for g in range(lb.max() + 1)]); r = np.empty_like(o); r[o] = np.arange(len(o)); lb = r[lb]
    nm = NAMES.get(c) if NAMES.get(c) and len(NAMES[c]) == lb.max() + 1 else [f'cluster {g + 1}' for g in range(lb.max() + 1)]
    if c == 'CRC':   # equal mean risk in the two higher-risk clusters: name them by size
        small = 1 + int((lb == 2).sum() < (lb == 1).sum()); nm = [nm[0]] + [nm[1] if g == small else nm[2] for g in (1, 2)]
    km_panel(ax[i // 3, i % 3], d, lb, nm); ax[i // 3, i % 3].set_title(c)
fig.suptitle('Overall survival by within-cancer cluster of HSTU patient embeddings (unsupervised; out-of-fold embeddings)')
fig.tight_layout(); fig.savefig('plots_public/km_clusters_by_cohort.png'); plt.close(fig)

ix = np.where(M.cohort == 'Breast')[0]; d = M.iloc[ix]
C = np.c_[np.log(d.risk12 / (1 - d.risk12)), d.line, d.line ** 2]
lb = KMeans(4, n_init=10, random_state=0).fit_predict(PCA(10, random_state=0).fit_transform(X[ix] - LinearRegression().fit(C, X[ix]).predict(C)))
tk = np.array(toks, dtype=object)[ix]
def frac(f): return np.array([np.mean([any(f(t) for t in s) for s in tk[lb == g]]) for g in range(4)])
her2, cdh1, trial = frac(lambda t: t.startswith('REG_') and 'trastuzumab' in t), frac(lambda t: t == 'MUT:CDH1'), frac(lambda t: t == 'REG_START:investigational')
nm = {}
nm[int(np.argmax(her2))] = 'HER2+ (anti-HER2, CDK12 amp)'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, cdh1)))] = 'CDH1 / lobular, HR+'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, trial)))] = 'clinical-trial heavy'
for g in range(4): nm.setdefault(g, 'later-line chemo')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
km_panel(ax[0], d, lb, [nm[g] for g in range(4)]); ax[0].set_title('Breast: clusters after removing predicted risk and line')
feats = [('anti-HER2 drug', lambda t: t.startswith('REG_') and 'trastuzumab' in t), ('AMP:ERBB2', lambda t: t == 'AMP:ERBB2'),
         ('AMP:CDK12', lambda t: t == 'AMP:CDK12'), ('MUT:CDH1', lambda t: t == 'MUT:CDH1'),
         ('CDK4/6 inhibitor', lambda t: t.startswith('REG_') and any(x in t for x in ('palbociclib', 'ribociclib', 'abemaciclib'))),
         ('investigational', lambda t: t == 'REG_START:investigational')]
H = np.array([frac(f) for _, f in feats]); w = .2
for g in range(4): ax[1].bar(np.arange(len(feats)) + (g - 1.5) * w, H[:, g], w, color=PAL[g], label=nm[g])
ax[1].set_xticks(range(len(feats))); ax[1].set_xticklabels([f for f, _ in feats], rotation=30, ha='right')
ax[1].set_ylabel('share of patients in cluster'); ax[1].set_title('What defines each cluster'); ax[1].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig('plots_public/breast_subtypes.png'); plt.close(fig)

shutil.copy('plots/emb_regimen_map.png', 'plots_public/regimen_map.png')
print('wrote', sorted(os.listdir('plots_public')))
