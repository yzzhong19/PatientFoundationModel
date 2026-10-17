"""Step 3: patient-level splits, fixed before any model is trained.
Test sets contain only patients that have >=1 landmark. For every fold we also store the pretraining pool:
all patients (incl. those without landmarks) except the test patients and, for leave-one-out splits, the whole held-out group."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
ev = pd.read_parquet('events.parquet', columns=['patient_id', 'cohort'])
pc = ev.drop_duplicates('patient_id').set_index('patient_id').cohort
lm = pd.read_parquet('landmarks.parquet')
pt = lm.drop_duplicates('patient_id').set_index('patient_id')[['cohort', 'institution']]
allp = sorted(pc.index)
S = {}
def fold(name, test, heldout_mask_all=None):
    test = set(test)
    train = sorted(set(pt.index) - test - (set(heldout_mask_all) if heldout_mask_all else set()))
    pre = sorted(set(allp) - test - (set(heldout_mask_all) if heldout_mask_all else set()))
    return {'name': name, 'train': train, 'test': sorted(test), 'pretrain': pre}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
idx = np.array(pt.index)
S['random5'] = [fold(f'fold{i}', idx[te]) for i, (_, te) in enumerate(skf.split(idx, pt.cohort))]
inst = lm.drop_duplicates('patient_id').set_index('patient_id').institution
# institution of non-landmark patients from labels_raw
lab = pd.read_pickle('labels_raw.pkl').drop_duplicates('patient_id').set_index('patient_id').institution
S['loio'] = [fold(i, pt.index[pt.institution == i], [p for p in allp if lab.get(p) == i]) for i in sorted(pt.institution.unique())]
S['loco'] = [fold(c, pt.index[pt.cohort == c], [p for p in allp if pc[p] == c]) for c in sorted(pt.cohort.unique())]
# checks
for k, fs in S.items():
    for f in fs:
        assert not set(f['train']) & set(f['test']), (k, f['name'])
        assert not set(f['pretrain']) & set(f['test']), (k, f['name'])
    if k == 'random5':
        assert sorted(sum([f['test'] for f in fs], [])) == sorted(pt.index)
    print(k, [(f['name'], len(f['train']), len(f['test']), len(f['pretrain'])) for f in fs])
json.dump(S, open('splits.json', 'w'))
