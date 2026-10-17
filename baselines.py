"""Step 4 baselines 1-4 on the fixed splits. Writes predictions.parquet (one row per test landmark x model x split x fold).
Risk score: higher = higher hazard. Only pre-landmark information is used (see common.prefix_index)."""
import sys, warnings, numpy as np, pandas as pd, scipy.sparse as sps
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time as rmst_fn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from common import load, prefix_index
warnings.filterwarnings('ignore')
ev, lm, sp = load()
ev = ev.sort_values(['patient_id', 'day', 'rank'], kind='stable').reset_index(drop=True)
P = prefix_index(ev, lm)
lm = lm.reset_index(drop=True)
y_all = Surv.from_arrays(lm.os_event.astype(bool).values, lm.os_time.values)

# ---- features
vc = ev.token.value_counts()
vocab = {t: i for i, t in enumerate(vc[vc >= 30].index)}          # label-free frequency filter
tid = ev.token.map(vocab).fillna(-1).astype(int).values
rows, cols = [], []
for k, r in enumerate(lm.landmark_id.values):
    t = tid[P[r]]; t = t[t >= 0]
    rows += [k] * len(t); cols += list(t)
Xc = sps.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(lm), len(vocab)))
Xc.sum_duplicates()
Xcounts = Xc.toarray()
print('count matrix', Xcounts.shape, flush=True)

stage = pd.get_dummies(lm.STAGE.fillna('NA'), prefix='stage').astype(float)
coh = pd.get_dummies(lm.cohort, prefix='coh').astype(float)
age = lm.AGE.map({'<40': 0, '40s': 1, '50s': 2, '60s': 3, '70s': 4, '80+': 5}).fillna(3).astype(float)
cls = pd.DataFrame({f'cls{b}': (lm.dclass.values >> b) & 1 for b in range(5)}).astype(float)
sex = (lm.SEX == 'Male').astype(float)
Xclin = pd.concat([coh, stage, age.rename('age'), np.log(lm.line).rename('logline'), cls, sex.rename('male')], axis=1)

gs = pd.read_pickle('genomics_status.pkl')
genes = pd.read_csv('top_genes.csv').gene.sort_values().values
A = np.stack(gs.genes.values)
covered = np.where((A >= 0).all(axis=0))[0]
print('genes covered by every sample:', len(covered), list(genes[covered])[:15], flush=True)
gdf = gs.set_index('patient_id')
Xgen = np.zeros((len(lm), len(covered)), dtype=float)
byp = {p: (g.day.values, np.stack(g.genes.values)[:, covered]) for p, g in gs.groupby('patient_id')}
for k, r in enumerate(lm.itertuples(index=False)):
    d, M = byp[r.patient_id]
    m = d <= r.landmark_day
    if m.any(): Xgen[k] = (M[m] == 1).any(axis=0)
Xcg = np.hstack([Xclin.values, Xgen])


def cox_fit_predict(X, tr, te, alphas, seed=0):
    sc = StandardScaler().fit(X[tr]); sc.scale_[sc.scale_ == 0] = 1
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    pats = lm.patient_id.values[tr]
    best, ba = -1, alphas[0]
    for a in alphas:
        s = []
        for i, (a_, b_) in enumerate(GroupKFold(3).split(Xtr, groups=pats)):
            try:
                m = CoxPHSurvivalAnalysis(alpha=a).fit(Xtr[a_], y_all[tr][a_]); s.append(m.score(Xtr[b_], y_all[tr][b_]))
            except Exception: s.append(0.5)
        if np.mean(s) > best: best, ba = np.mean(s), a
    m = CoxPHSurvivalAnalysis(alpha=ba).fit(Xtr, y_all[tr])
    return m.predict(Xte), ba


def run_fold(stype, f):
    tr = np.where(lm.patient_id.isin(set(f['train'])))[0]; te = np.where(lm.patient_id.isin(set(f['test'])))[0]
    out = []
    def emit(model, risk, note=''):
        out.append(pd.DataFrame({'landmark_id': lm.landmark_id.values[te], 'split': stype, 'fold': f['name'], 'model': model, 'risk': risk, 'note': note}))
    # 1. KM by cohort (risk = -restricted mean survival time up to 3 years; unseen cohort -> pooled curve)
    def rmst(t, e):
        k = KaplanMeierFitter().fit(t, e); return rmst_fn(k, t=1095)
    pooled = rmst(lm.os_time.values[tr], lm.os_event.values[tr])
    r = {c: rmst(lm.os_time.values[tr][lm.cohort.values[tr] == c], lm.os_event.values[tr][lm.cohort.values[tr] == c]) for c in set(lm.cohort.values[tr])}
    emit('KM_cohort', -np.array([r.get(c, pooled) for c in lm.cohort.values[te]]))
    # 2. Cox clinical
    risk, a = cox_fit_predict(Xclin.values, tr, te, [0.01, 0.1, 1, 10, 100]); emit('Cox_clinical', risk, f'alpha={a}')
    # 3. Cox + genomics
    risk, a = cox_fit_predict(Xcg, tr, te, [1, 10, 100, 1000]); emit('Cox_clinical_genomics', risk, f'alpha={a}')
    # 4. GBSA on token counts
    g = GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.7, min_samples_leaf=20,
                                         max_features='sqrt', random_state=0).fit(Xcounts[tr], y_all[tr])
    emit('GBSA_counts', g.predict(Xcounts[te]))
    print('done', stype, f['name'], flush=True)
    return pd.concat(out)


if __name__ == '__main__':
    jobs = [(k, f) for k, fs in sp.items() for f in fs]
    if len(sys.argv) > 1: jobs = [j for j in jobs if j[0] in sys.argv[1:]]
    res = Parallel(n_jobs=4)(delayed(run_fold)(k, f) for k, f in jobs)
    pd.concat(res).to_parquet('predictions_baselines.parquet', index=False)
