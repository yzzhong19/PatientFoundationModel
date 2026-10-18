"""Figures for the patient embeddings from analysis/emb_patients.py -> plots/emb_*.png. Run from the repo root.
1 UMAP of all patients by cohort and by predicted 12-month risk; 2 per-cohort clusters (UMAP + Kaplan-Meier);
3 breast clusters after regressing out risk and line; 4 contextual token map of regimens coloured by drug class."""
import os, sys, numpy as np, pandas as pd, torch, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import umap
from data import Events, ETYPES
from simulate import load_fold, dev
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

os.makedirs('plots', exist_ok=True)
plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
COH = ['NSCLC', 'CRC', 'Breast', 'Pancreatic', 'Prostate', 'Bladder']
CC = dict(zip(COH, ['#4C78A8', '#F58518', '#E45756', '#72B7B2', '#54A24B', '#B279A2']))
PAL = ['#4C78A8', '#E45756', '#54A24B', '#F58518', '#B279A2', '#72B7B2']
E = Events(); emb = np.load('figs/patient_emb.npy'); M = pd.read_parquet('figs/patient_emb_meta.parquet')
X = StandardScaler().fit_transform(emb)
toks = [set(E.tokens[E.P[l]]) for l in M.landmark_id]


def km_panel(ax, d, lab, names=None):
    for g in sorted(set(lab)):
        m = lab == g; k = KaplanMeierFitter().fit(d.os_time.values[m] / 30.44, d.os_event.values[m])
        ax.step(k.timeline, k.survival_function_.values[:, 0], where='post', color=PAL[g % 6], lw=1.6,
                label=(names[g] if names else f'cluster {g + 1}') + f' (n={m.sum()})')
    p = multivariate_logrank_test(d.os_time, lab, d.os_event).p_value
    ax.set_xlim(0, 60); ax.set_ylim(0, 1.02); ax.set_xlabel('months from landmark'); ax.set_ylabel('overall survival')
    ax.text(0.98, 0.95, f'log-rank p={p:.1e}', transform=ax.transAxes, ha='right', va='top', fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc='lower left')


def kmeans_best(P):
    sil = {k: silhouette_score(P, KMeans(k, n_init=10, random_state=0).fit_predict(P)) for k in range(2, 7)}
    k = max(sil, key=sil.get); return KMeans(k, n_init=10, random_state=0).fit_predict(P)


def order_by_risk(lab, risk):
    """relabel clusters 0..k-1 from lowest to highest mean predicted risk"""
    o = np.argsort([risk[lab == g].mean() for g in range(lab.max() + 1)]); r = np.empty_like(o); r[o] = np.arange(len(o))
    return r[lab]


# ---- 1 all patients
U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(PCA(20, random_state=0).fit_transform(X))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for c in COH:
    m = (M.cohort == c).values; ax[0].scatter(U[m, 0], U[m, 1], s=3, c=CC[c], label=f'{c} ({m.sum()})', alpha=.7, lw=0)
ax[0].legend(markerscale=4, frameon=False, fontsize=8); ax[0].set_title('Patient embeddings at first eligible landmark, by cancer')
sc = ax[1].scatter(U[:, 0], U[:, 1], s=3, c=M.risk12, cmap='magma_r', vmin=0, vmax=.8, lw=0)
plt.colorbar(sc, ax=ax[1], label='HSTU predicted 12-month risk'); ax[1].set_title('Same map, by predicted risk')
for a in ax: a.set_xticks([]); a.set_yticks([]); a.set_xlabel('UMAP 1'); a.set_ylabel('UMAP 2')
fig.text(0.5, 0.005, f'{len(M):,} patients; each embedded by the fold model that never trained on them (5 folds, Procrustes-aligned)',
         ha='center', fontsize=8, color='#555'); fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig('plots/emb_umap_all.png'); plt.close(fig)

# ---- 2 per-cohort clusters
NAMES = {'Breast': ['early / curative-intent', 'metastatic, heavily treated'],
         'Prostate': ['hormone-sensitive, PSA normal', 'intermediate', 'post-taxane CRPC'],
         'Bladder': ['non-muscle-invasive (BCG)', 'intermediate', 'progressing on chemo'],
         'CRC': ['CEA normal', 'stage IV, small group', 'progressing, later lines']}
fig, ax = plt.subplots(4, 3, figsize=(13, 15))
for i, c in enumerate(COH):
    ix = np.where(M.cohort == c)[0]; d = M.iloc[ix]; P = PCA(10, random_state=0).fit_transform(X[ix])
    lab = order_by_risk(kmeans_best(P), d.risk12.values)
    u = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=0).fit_transform(P)
    a = ax[(i // 3) * 2, i % 3]
    for g in range(lab.max() + 1):
        m = lab == g; a.scatter(u[m, 0], u[m, 1], s=5, c=PAL[g], lw=0, alpha=.8)
    a.set_title(f'{c}: {lab.max() + 1} clusters'); a.set_xticks([]); a.set_yticks([])
    nm = NAMES.get(c) if NAMES.get(c) and len(NAMES[c]) == lab.max() + 1 else None
    if c == 'CRC' and nm:   # the two higher-risk CRC clusters have equal mean risk: name them by size instead
        small = 1 + int((lab == 2).sum() < (lab == 1).sum())
        nm = [nm[0]] + [nm[1] if g == small else nm[2] for g in (1, 2)]
    km_panel(ax[(i // 3) * 2 + 1, i % 3], d, lab, nm)
fig.suptitle('Within-cancer clusters of HSTU patient embeddings separate disease phase (k by silhouette; clusters ordered by predicted risk)', y=.995)
fig.tight_layout(); fig.savefig('plots/emb_clusters_by_cohort.png'); plt.close(fig)

# ---- 3 breast after regressing out risk and line
ix = np.where(M.cohort == 'Breast')[0]; d = M.iloc[ix]
C = np.c_[np.log(d.risk12 / (1 - d.risk12)), d.line, d.line ** 2]
Xr = X[ix] - LinearRegression().fit(C, X[ix]).predict(C); P = PCA(10, random_state=0).fit_transform(Xr)
lab = KMeans(4, n_init=10, random_state=0).fit_predict(P)
tk = np.array(toks, dtype=object)[ix]
def frac(f): return np.array([np.mean([any(f(t) for t in s) for s in tk[lab == g]]) for g in range(4)])
her2 = frac(lambda t: t.startswith('REG_') and 'trastuzumab' in t); cdh1 = frac(lambda t: t == 'MUT:CDH1')
trial = frac(lambda t: t == 'REG_START:investigational')
nm = {}
nm[int(np.argmax(her2))] = 'HER2+ (anti-HER2, CDK12 amp)'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, cdh1)))] = 'CDH1 / lobular, HR+'
nm[int(np.argmax(np.where([g in nm for g in range(4)], -1, trial)))] = 'clinical-trial heavy'
for g in range(4): nm.setdefault(g, 'later-line chemo')
u = umap.UMAP(n_neighbors=20, min_dist=0.3, random_state=0).fit_transform(P)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for g in range(4):
    m = lab == g; ax[0].scatter(u[m, 0], u[m, 1], s=7, c=PAL[g], lw=0, alpha=.85, label=nm[g])
ax[0].legend(frameon=False, fontsize=8, markerscale=2.5); ax[0].set_xticks([]); ax[0].set_yticks([])
ax[0].set_title('Breast, after removing risk and line')
km_panel(ax[1], d, lab, [nm[g] for g in range(4)]); ax[1].set_title('Overall survival by residual cluster')
feats = [('anti-HER2 drug', lambda t: t.startswith('REG_') and 'trastuzumab' in t), ('AMP:ERBB2', lambda t: t == 'AMP:ERBB2'),
         ('AMP:CDK12', lambda t: t == 'AMP:CDK12'), ('MUT:CDH1', lambda t: t == 'MUT:CDH1'),
         ('CDK4/6 inhibitor', lambda t: t.startswith('REG_') and any(x in t for x in ('palbociclib', 'ribociclib', 'abemaciclib'))),
         ('investigational', lambda t: t == 'REG_START:investigational')]
H = np.array([frac(f) for _, f in feats]); w = .2
for g in range(4): ax[2].bar(np.arange(len(feats)) + (g - 1.5) * w, H[:, g], w, color=PAL[g], label=nm[g])
ax[2].set_xticks(range(len(feats))); ax[2].set_xticklabels([f for f, _ in feats], rotation=30, ha='right')
ax[2].set_ylabel('share of patients'); ax[2].set_title('What defines each cluster')
fig.tight_layout(); fig.savefig('plots/emb_breast_subtypes.png'); plt.close(fig)

# ---- 4 contextual token map of regimens (fold 0 model, pretrain patients)
Fd, model = load_fold(E, 'fold0'); base = len(ETYPES)
seqs = Fd.pretrain_seqs(Fd.pretrain, np.random.RandomState(0)); Hs = np.zeros((Fd.V, 128)); N = np.zeros(Fd.V)
with torch.no_grad():
    for i in range(0, len(seqs), 128):
        b = {k: v.to(dev) for k, v in Fd.pack(seqs[i:i + 128]).items()}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'): h = model.encode(b).float()
        m = torch.arange(b['tok'].shape[1], device=dev)[None] < b['len'][:, None]
        np.add.at(Hs, b['tok'][m].cpu().numpy(), h[m].cpu().numpy()); np.add.at(N, b['tok'][m].cpu().numpy(), 1)
names = pd.Series(E.tokens).groupby(Fd.tid).first()
ids = [i for i in names.index if i >= base and names[i].startswith('REG_START:') and N[i] >= 20 and names[i] != 'REG_START:investigational']
V = Hs[ids] / N[ids, None]; V = V - V.mean(0); V /= np.linalg.norm(V, axis=1, keepdims=True)
u = umap.UMAP(n_neighbors=8, min_dist=0.4, metric='cosine', random_state=0).fit_transform(V)
dc = E.ev.groupby('token').dclass.first(); cls = ['chemo', 'immuno', 'targeted', 'hormonal', 'antiangio']
def main_cls(b): return 'immuno' if b & 2 else 'targeted' if b & 4 else 'hormonal' if b & 8 else 'chemo' if b & 1 else 'other'
CL = {'chemo': '#9E9E9E', 'immuno': '#E45756', 'targeted': '#4C78A8', 'hormonal': '#54A24B', 'other': '#B279A2'}
lab = [main_cls(int(dc[names[i]])) for i in ids]
fig, ax = plt.subplots(figsize=(10, 7.5))
for c in CL:
    m = np.array(lab) == c; ax.scatter(u[m, 0], u[m, 1], s=np.sqrt(N[np.array(ids)[m]]) * 2, c=CL[c], alpha=.75, lw=0, label=c)
show = set(np.argsort(-N[ids])[:30])
for j, i in enumerate(ids):
    if j in show: ax.annotate(names[i].split(':', 1)[1][:28], u[j], fontsize=6.5, alpha=.85, xytext=(3, 2), textcoords='offset points')
ax.legend(frameon=False, title='drug class (dominant)', markerscale=1); ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'Regimen tokens in context: {len(ids)} regimens, point size ~ frequency (fold-0 model)')
fig.tight_layout(); fig.savefig('plots/emb_regimen_map.png'); plt.close(fig)
print('wrote', sorted(f for f in os.listdir('plots') if f.startswith('emb_')))
