"""Aggregate-only UMAP maps of the HSTU patient embeddings -> plots_public/. Same embeddings, UMAP settings and clusters as
analysis/emb_plots.py, but drawn as hexagonal bins: a bin is shown only when it holds >= MINCNT patients, coloured by its
majority cancer / cluster or its mean predicted risk, so no individual patient is plotted.
1 emb_umap_all.png: all patients by cancer and by predicted 12-month risk; 2 emb_breast_umap.png: breast residual clusters."""
import os, sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import umap
from data import Events
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

MINCNT = 5
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
PAL = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#72B7B2']
COH = ['NSCLC', 'CRC', 'Breast', 'Pancreatic', 'Prostate', 'Bladder']
CC = dict(zip(COH, ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#B279A2']))
os.makedirs('plots_public', exist_ok=True)
E = Events(); emb = np.load('figs/patient_emb.npy'); M = pd.read_parquet('figs/patient_emb_meta.parquet')
X = StandardScaler().fit_transform(emb)


def hex_majority(ax, u, lab, colors, grid):
    k = len(colors)
    ax.hexbin(u[:, 0], u[:, 1], C=lab.astype(float), reduce_C_function=lambda v: np.bincount(np.asarray(v, int), minlength=k).argmax(),
              gridsize=grid, mincnt=MINCNT, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(-.5, k), k), linewidths=.2, edgecolors='white')


def tidy(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')


# ---- 1 all patients
U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(PCA(20, random_state=0).fit_transform(X))
ci = M.cohort.map({c: i for i, c in enumerate(COH)}).values
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
hex_majority(ax[0], U, ci, [CC[c] for c in COH], 45)
ax[0].legend([Patch(color=CC[c]) for c in COH], [f'{c} ({(M.cohort == c).sum()})' for c in COH], frameon=False, fontsize=8)
ax[0].set_title('Patient embeddings at first eligible landmark, by cancer')
hb = ax[1].hexbin(U[:, 0], U[:, 1], C=M.risk12.values, reduce_C_function=np.mean, gridsize=45, mincnt=MINCNT, cmap='magma_r',
                  vmin=0, vmax=.8, linewidths=.2, edgecolors='white')
plt.colorbar(hb, ax=ax[1], label='mean HSTU predicted 12-month risk'); ax[1].set_title('Same map, by predicted risk')
for a in ax: tidy(a)
fig.text(.5, .005, f'{len(M):,} patients, out-of-fold embeddings (5 folds, Procrustes-aligned); hexagons with < {MINCNT} patients not shown',
         ha='center', fontsize=8, color='#555'); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig('plots_public/emb_umap_all.png'); plt.close(fig)

# ---- 2 breast after regressing out risk and line (same clusters and names as emb_plots.py / public_plots.py)
ix = np.where(M.cohort == 'Breast')[0]; d = M.iloc[ix]
C = np.c_[np.log(d.risk12 / (1 - d.risk12)), d.line, d.line ** 2]
P = PCA(10, random_state=0).fit_transform(X[ix] - LinearRegression().fit(C, X[ix]).predict(C))
lab = KMeans(4, n_init=10, random_state=0).fit_predict(P)
tk = np.array([set(E.tokens[E.P[l]]) for l in d.landmark_id], dtype=object)
def frac(f): return np.array([np.mean([any(f(t) for t in s) for s in tk[lab == g]]) for g in range(4)])
her2, cdh1, trial = frac(lambda t: t.startswith('REG_') and 'trastuzumab' in t), frac(lambda t: t == 'MUT:CDH1'), frac(lambda t: t == 'REG_START:investigational')
nm = {}
nm[int(np.argmax(her2))] = 'HER2+ (anti-HER2, CDK12 amp)'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, cdh1)))] = 'CDH1 / lobular, HR+'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, trial)))] = 'clinical-trial heavy'
for g in range(4): nm.setdefault(g, 'later-line chemo')
u = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=0).fit_transform(P)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
hex_majority(ax, u, lab, PAL[:4], 22)
ax.legend([Patch(color=PAL[g]) for g in range(4)], [f'{nm[g]} (n={(lab == g).sum()})' for g in range(4)], frameon=False, fontsize=8)
ax.set_title('Breast: embedding clusters after removing predicted risk and line'); tidy(ax)
fig.text(.5, .005, f'{len(ix)} patients; hexagons coloured by majority cluster, < {MINCNT} patients not shown', ha='center', fontsize=8, color='#555')
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig('plots_public/emb_breast_umap.png'); plt.close(fig)
print('done')
