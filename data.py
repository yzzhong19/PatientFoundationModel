"""Tokenizer + tensors for HSTU. Per fold: vocabulary from the fold's `pretrain` events (count >= 5, else per-etype UNK).
Landmark inputs = common.prefix_index (nothing after the landmark day), truncated to MAXLEN keeping the 5 STATIC tokens,
all genomic tokens (most recent samples if > MAXGEN) and the most recent remaining events."""
import numpy as np, pandas as pd, torch
from common import load, prefix_index

MAXLEN, MAXGEN = 256, 200
ETYPES = ['PAD', 'STATIC', 'PANEL', 'SAMPLE', 'TMB', 'MUT', 'FUSION', 'AMP', 'DEL', 'PATH', 'TM', 'MET', 'MD', 'SCAN', 'REG_END', 'REG_START']
GENOMIC = {'PANEL', 'SAMPLE', 'TMB', 'MUT', 'FUSION', 'AMP', 'DEL'}
COHORTS = ['NSCLC', 'CRC', 'Breast', 'Pancreatic', 'Prostate', 'Bladder']   # id 0 = UNKNOWN, cohort i -> i + 1
NTB = 13            # time-gap buckets floor(log2(gap + 1)) capped at 12; bucket 13 = STATIC key
NBINS, BINW = 60, 30.44


def gap_bucket(g):
    return np.minimum(np.floor(np.log2(np.maximum(g, 0) + 1)), NTB - 1).astype(np.int64)


def day_bucket(d):
    """signed log bucket of days since diagnosis: 0..15 for d >= 0, 16..31 for d < 0"""
    b = np.minimum(np.floor(np.log2(np.abs(d) + 1)), 15).astype(np.int64)
    return np.where(d < 0, b + 16, b)


class Events:
    """All events once, sorted; per-fold vocab is built on top."""
    def __init__(self):
        ev, lm, sp = load()
        self.ev = ev.sort_values(['patient_id', 'day', 'rank'], kind='stable').reset_index(drop=True)
        self.lm, self.sp = lm.reset_index(drop=True), sp
        self.P = prefix_index(self.ev, self.lm)
        e = self.ev
        self.etype = e.etype.map({t: i for i, t in enumerate(ETYPES)}).values.astype(np.int64)
        self.day = e.day.values.astype(np.float64)
        self.dclass = e.dclass.values.astype(np.int64)
        self.cohort = e.cohort.map({c: i + 1 for i, c in enumerate(COHORTS)}).values.astype(np.int64)
        self.is_static = self.etype == 1
        self.is_gen = e.etype.isin(GENOMIC).values
        self.pat_ix = e.groupby('patient_id', sort=False).indices
        self.tokens = e.token.values
        # every landmark's own REG_START is the last visible token and no visible event is after the landmark day
        for r in self.lm.itertuples(index=False):
            ix = self.P[r.landmark_id]
            assert self.etype[ix[-1]] == ETYPES.index('REG_START') and self.day[ix].max() <= r.landmark_day


def build_vocab(E, pretrain_patients):
    """token id: 0 PAD, 1..len(ETYPES)-1 per-etype UNK, then tokens with count >= 5 in the fold's pretrain events"""
    m = E.ev.patient_id.isin(set(pretrain_patients)).values
    vc = pd.Series(E.tokens[m]).value_counts()
    keep = sorted(vc[vc >= 5].index)
    base = len(ETYPES)
    tok2id = {t: base + i for i, t in enumerate(keep)}
    tid = np.array([tok2id.get(t, 0) for t in E.tokens], dtype=np.int64)
    tid = np.where(tid == 0, E.etype, tid)          # unknown token -> its etype's UNK id
    vtype = np.zeros(base + len(keep), dtype=np.int64)
    vtype[:base] = np.arange(base)
    vtype[base:] = [ETYPES.index(t.split(':', 1)[0]) for t in keep]
    return tid, vtype


def truncate(E, ix, no_genomic=False):
    """positions to keep (in time order) for index array ix"""
    if no_genomic:
        ix = ix[~E.is_gen[ix]]
    if len(ix) <= MAXLEN:
        return ix
    st, gen = E.is_static[ix], E.is_gen[ix]
    s, g, o = ix[st], ix[gen], ix[~st & ~gen]
    if len(g) > MAXGEN:
        g = g[-MAXGEN:]                              # most recent samples' genomic tokens
    o = o[len(o) - (MAXLEN - len(s) - len(g)):] if MAXLEN - len(s) - len(g) > 0 else o[:0]
    return np.sort(np.concatenate([s, g, o]))


class Fold:
    """Pre-tensorized inputs for one fold."""
    def __init__(self, E, stype, fold, no_genomic=False, restrict_cohort=None):
        self.E, self.no_genomic = E, no_genomic
        f = [x for x in E.sp[stype] if x['name'] == fold][0]
        keep = (lambda ps: ps) if restrict_cohort is None else \
            (lambda ps: [p for p in ps if E.ev.cohort.values[E.pat_ix[p][0]] == restrict_cohort])
        self.pretrain, self.train, self.test = keep(f['pretrain']), keep(f['train']), keep(f['test'])
        assert not set(self.test) & set(self.pretrain) and not set(self.test) & set(self.train)
        self.tid, self.vtype = build_vocab(E, self.pretrain)
        self.V = len(self.vtype)

    def pack(self, seqs, unk_cohort=None):
        """list of index arrays -> dict of padded tensors (right padded)"""
        E, n = self.E, len(seqs)
        L = max(len(s) for s in seqs)
        out = {k: np.zeros((n, L), dtype=np.int64) for k in ('etype', 'tok', 'dclass', 'cohort', 'dayb', 'static')}
        day = np.zeros((n, L), dtype=np.float64)
        length = np.zeros(n, dtype=np.int64)
        for i, s in enumerate(seqs):
            k = len(s); length[i] = k
            out['etype'][i, :k] = E.etype[s]; out['tok'][i, :k] = self.tid[s]; out['dclass'][i, :k] = E.dclass[s]
            out['cohort'][i, :k] = E.cohort[s]; out['dayb'][i, :k] = day_bucket(E.day[s]); out['static'][i, :k] = E.is_static[s]
            day[i, :k] = E.day[s]; day[i, k:] = E.day[s[-1]]
        if unk_cohort is not None:
            out['cohort'][np.isin(out['cohort'], unk_cohort)] = 0
        t = {k: torch.from_numpy(v) for k, v in out.items()}
        t['day'] = torch.from_numpy(day).float()
        t['len'] = torch.from_numpy(length)
        return t

    def landmark_seqs(self, patients):
        lm = self.E.lm
        sel = np.where(lm.patient_id.isin(set(patients)).values)[0]
        seqs = [truncate(self.E, self.E.P[lid], self.no_genomic) for lid in lm.landmark_id.values[sel]]
        return sel, seqs

    def pretrain_seqs(self, patients, rng):
        """whole history; random window (by end point) if longer than MAXLEN, same static/genomic protection"""
        out = []
        for p in patients:
            ix = self.E.pat_ix[p]
            if len(ix) > MAXLEN:
                end = rng.randint(MAXLEN, len(ix) + 1)
                ix = ix[:end]
            out.append(truncate(self.E, ix, self.no_genomic))
        return out


def surv_targets(t, e):
    """discrete-time targets: bin index k = floor(t / 30.44); >= 60 bins -> censored after the last bin"""
    k = np.floor(t / BINW).astype(np.int64)
    ev = (e == 1) & (k < NBINS)
    k = np.minimum(k, NBINS)
    return torch.from_numpy(k), torch.from_numpy(ev.astype(np.float32))
