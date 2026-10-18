"""Random survival forest baseline on the GBSA token-count features (baselines.py), random5 only.
Training times are rounded up to monthly bins (30.44 d, as the HSTU hazard head) so the forest's per-leaf survival curves fit
in memory; test landmarks are scored with risk = predicted cumulative hazard summed over time (sksurv predict),
risk12 / risk24 = 1 - S(12 / 24 months). Writes runs/RSF_counts/random5/<fold>/predictions.parquet (same contract as train.py)."""
import os, sys, time, numpy as np, pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from baselines import lm, Xcounts, sp

t0 = time.time()
mon = np.ceil(lm.os_time.values / 30.44).clip(min=1)
y = Surv.from_arrays(lm.os_event.astype(bool).values, mon)
folds = [f for f in sp['random5'] if len(sys.argv) < 2 or f['name'] in sys.argv[1:]]
for f in folds:
    odir = f'runs/RSF_counts/random5/{f["name"]}'; os.makedirs(odir, exist_ok=True)
    tr = np.where(lm.patient_id.isin(set(f['train'])))[0]; te = np.where(lm.patient_id.isin(set(f['test'])))[0]
    m = RandomSurvivalForest(n_estimators=300, min_samples_leaf=20, max_features='sqrt', n_jobs=4, random_state=0).fit(Xcounts[tr], y[tr])
    S = m.predict_survival_function(Xcounts[te], return_array=True); ut = m.unique_times_
    s_at = lambda mo: S[:, np.searchsorted(ut, mo, side='right') - 1] if (ut <= mo).any() else np.ones(len(te))
    pd.DataFrame({'landmark_id': lm.landmark_id.values[te], 'split': 'random5', 'fold': f['name'], 'model': 'RSF_counts',
                  'risk': m.predict(Xcounts[te]), 'risk12': 1 - s_at(12), 'risk24': 1 - s_at(24)}).to_parquet(f'{odir}/predictions.parquet', index=False)
    print(f'done {f["name"]} ({time.time() - t0:.0f}s)', flush=True)
    del m, S
