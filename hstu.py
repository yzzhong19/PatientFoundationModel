"""HSTU over patient timelines. Layer: U,V,Q,K = split(SiLU(W1 LN(x))); A = SiLU(QK^T/sqrt(d) + pos_bias + time_bias) / N
(causal); out = x + W2(LayerNorm(A V) * U). time_bias: learned per head per bucket floor(log2(gap_days + 1)) (cap 12), STATIC keys
use a dedicated bucket. Heads: pretraining (next event type / value | type / log-time bucket) and a 60-bin discrete-time hazard
head on concat(h_landmark, static_emb)."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from data import ETYPES, COHORTS, NTB, NBINS

MAXREL = 256


class HSTULayer(nn.Module):
    def __init__(self, d, heads, dropout, time_bias=True):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln_in = nn.LayerNorm(d)
        self.W1 = nn.Linear(d, 4 * d)
        self.ln_av = nn.LayerNorm(d)
        self.W2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.pos_bias = nn.Parameter(torch.zeros(heads, MAXREL))
        self.time_bias = nn.Parameter(torch.zeros(heads, NTB + 1)) if time_bias else None

    def forward(self, x, rel, tb, mask, n):
        B, L, d = x.shape
        u, v, q, k = F.silu(self.W1(self.ln_in(x))).chunk(4, dim=-1)
        sh = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + self.pos_bias[:, rel]
        if self.time_bias is not None:
            s = s + self.time_bias[:, tb].permute(1, 0, 2, 3)
        a = F.silu(s) * mask / n
        av = (self.drop(a) @ v).transpose(1, 2).reshape(B, L, d)
        return x + self.drop(self.W2(self.ln_av(av) * u))


class GPTLayer(nn.Module):
    """Delphi-style baseline block: softmax attention (same biases) + standard GELU MLP, no gating."""
    def __init__(self, d, heads, dropout, time_bias=True):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.pos_bias = nn.Parameter(torch.zeros(heads, MAXREL))
        self.time_bias = nn.Parameter(torch.zeros(heads, NTB + 1)) if time_bias else None

    def forward(self, x, rel, tb, mask, n):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        sh = lambda t: t.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + self.pos_bias[:, rel]
        if self.time_bias is not None:
            s = s + self.time_bias[:, tb].permute(1, 0, 2, 3)
        a = torch.softmax(s.masked_fill(mask == 0, -1e9), dim=-1)
        x = x + self.drop(self.o((self.drop(a) @ v).transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class HSTU(nn.Module):
    def __init__(self, V, d=128, layers=3, heads=4, dropout=0.2, time_bias=True, static_head=True, gpt=False):
        super().__init__()
        self.static_head = static_head
        self.e_type = nn.Embedding(len(ETYPES), d)
        self.e_tok = nn.Embedding(V, d)
        self.e_cls = nn.Linear(5, d, bias=False)
        self.e_coh = nn.Embedding(len(COHORTS) + 1, d)
        self.e_day = nn.Embedding(32, d)
        self.drop = nn.Dropout(dropout)
        Layer = GPTLayer if gpt else HSTULayer
        self.layers = nn.ModuleList([Layer(d, heads, dropout, time_bias) for _ in range(layers)])
        self.ln_out = nn.LayerNorm(d)
        # pretraining heads
        self.h_type = nn.Linear(d, len(ETYPES))
        self.h_val = nn.Linear(d, V)
        self.h_time = nn.Linear(d, NTB)
        # survival head
        self.h_surv = nn.Sequential(nn.Linear(2 * d if static_head else d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, NBINS))

    def encode(self, b):
        et, L = b['etype'], b['etype'].shape[1]
        bits = ((b['dclass'].unsqueeze(-1) >> torch.arange(5, device=et.device)) & 1).float()
        x = self.e_type(et) + self.e_tok(b['tok']) + self.e_cls(bits) + self.e_coh(b['cohort']) + self.e_day(b['dayb'])
        x = self.drop(x)
        ar = torch.arange(L, device=et.device)
        rel = (ar[:, None] - ar[None, :]).clamp(0, MAXREL - 1)
        valid = ar[None, :] < b['len'][:, None]                              # B, L
        mask = ((ar[None, :] <= ar[:, None])[None] & valid[:, None, :]).float().unsqueeze(1)   # B,1,L,L
        n = mask.sum(-1, keepdim=True).clamp(min=1)
        gap = b['day'][:, :, None] - b['day'][:, None, :]
        tb = torch.minimum(torch.floor(torch.log2(gap.clamp(min=0) + 1)), torch.tensor(NTB - 1., device=et.device)).long()
        tb = torch.where(b['static'][:, None, :].bool(), torch.full_like(tb, NTB), tb)
        for layer in self.layers:
            x = layer(x, rel, tb, mask, n)
        return self.ln_out(x)

    def landmark_repr(self, b):
        h = self.encode(b)
        idx = (b['len'] - 1).clamp(min=0)
        hl = h[torch.arange(h.shape[0], device=h.device), idx]
        if not self.static_head:
            return hl
        st = b['static'].float().unsqueeze(-1)
        hs = (h * st).sum(1) / st.sum(1).clamp(min=1)
        return torch.cat([hl, hs], -1)

    def hazard_logits(self, b):
        return self.h_surv(self.landmark_repr(b))


def surv_loss(logits, k, ev):
    """discrete-time hazard NLL. k = bin index (60 = survived all bins), ev = event in bin k.
    event: sum_{j<k} log(1-h_j) + log h_k; censored: sum_{j<k} log(1-h_j)"""
    ls = F.logsigmoid(-logits)                 # log(1 - h)
    lh = F.logsigmoid(logits)                  # log h
    ar = torch.arange(logits.shape[1], device=logits.device)
    at_risk = (ar[None, :] < k[:, None]).float()
    ll = (ls * at_risk).sum(1) + ev * lh.gather(1, k.clamp(max=NBINS - 1)[:, None]).squeeze(1)
    return -ll.mean()


def risks(logits):
    """risk = -(expected months survived within 60), risk12 = 1 - S(12m), risk24 = 1 - S(24m)"""
    S = torch.exp(torch.cumsum(F.logsigmoid(-logits), 1))
    return -S.sum(1), 1 - S[:, 11], 1 - S[:, 23]
