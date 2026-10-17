"""Shared loading + prefix logic. A landmark's visible history = that patient's events with (day, rank) strictly before
(landmark_day, REG_START rank) plus the landmark's own REG_START token. Concurrent REG_START tokens of other regimens on
the same day are NOT visible. Nothing after the landmark day is ever visible."""
import json, numpy as np, pandas as pd
REG_START_RANK = 14

def load():
    ev = pd.read_parquet('events.parquet')
    lm = pd.read_parquet('landmarks.parquet')
    sp = json.load(open('splits.json'))
    return ev, lm, sp

def prefix_index(ev, lm):
    """returns {landmark_id: row positions (np.array) into ev (sorted by patient, day, rank)}"""
    ev = ev.reset_index(drop=True)
    key = ev.day.values * 100 + ev['rank'].values
    starts = ev.groupby('patient_id').indices
    out = {}
    for r in lm.itertuples(index=False):
        ix = starts[r.patient_id]
        k = key[ix]
        cut = np.searchsorted(k, r.landmark_day * 100 + REG_START_RANK, side='left')
        sel = list(ix[:cut])
        own = ix[(ev.etype.values[ix] == 'REG_START') & (ev.ref.values[ix] == r.ref) & (ev.day.values[ix] == r.landmark_day)]
        sel += list(own[:1])
        out[r.landmark_id] = np.array(sel, dtype=np.int64)
    return out
