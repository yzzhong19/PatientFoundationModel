"""Unit tests: causal mask, no visible event after the landmark day, time-bias buckets, truncation rule.
usage: python test_hstu.py"""
import numpy as np, torch
from data import Events, Fold, truncate, gap_bucket, MAXLEN, MAXGEN, NTB
from hstu import HSTU

torch.manual_seed(0)
E = Events()                                   # asserts: last token = own REG_START, nothing after the landmark day
Fd = Fold(E, 'random5', 'fold0')

# 1. truncation keeps all 5 STATIC tokens, all genomic tokens (<= MAXGEN), the landmark token, and time order
for lid in E.lm.landmark_id.values:
    ix = E.P[lid]; t = truncate(E, ix)
    assert len(t) <= MAXLEN and (np.diff(t) > 0).all() and t[-1] == ix[-1]
    assert E.is_static[t].sum() == 5
    assert E.is_gen[t].sum() == min(E.is_gen[ix].sum(), MAXGEN) or len(ix) <= MAXLEN
print('truncation ok')

# 2. time buckets
assert list(gap_bucket(np.array([0, 0.5, 1, 2, 3, 7, 8, 1e6]))) == [0, 0, 1, 1, 2, 3, 3, NTB - 1]
print('time buckets ok')

# 3. causality: perturbing token j must not change hidden states at positions < j
model = HSTU(Fd.V).eval()
sel, seqs = Fd.landmark_seqs(Fd.train[:50])
b = Fd.pack(seqs[:8])
with torch.no_grad():
    h0 = model.encode(b)
    b2 = {k: v.clone() for k, v in b.items()}
    j = 10
    b2['tok'][:, j:] = 7; b2['day'][:, j:] += 500
    h1 = model.encode(b2)
assert torch.allclose(h0[:, :j], h1[:, :j], atol=1e-5) and not torch.allclose(h0[:, j:], h1[:, j:])
# padding must not change outputs
with torch.no_grad():
    b3 = Fd.pack(seqs[:1]); h_single = model.encode(b3)[0, :int(b3['len'][0])]
assert torch.allclose(h_single, h0[0, :int(b['len'][0])], atol=1e-5)
print('causal mask ok')
print('all tests passed')
