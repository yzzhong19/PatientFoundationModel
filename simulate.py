"""Counterfactual token edits for a trained HSTU fold model (shared by biology.py and dashboard.py).
A prefix is a dict of per-position arrays (etype, tok, dclass, cohort, static, day, gen); edits return a new dict."""
import json, numpy as np, torch, torch.nn.functional as F
from data import Fold, ETYPES, MAXLEN, day_bucket
from hstu import HSTU, risks

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_fold(E, fold, model_name='HSTU', split='random5', device=None):
    """(Fold, model in eval mode) for runs/<model_name>/<split>/<fold>"""
    Fd = Fold(E, split, fold)
    a = json.load(open(f'runs/{model_name}/{split}/{fold}/summary.json'))
    model = HSTU(Fd.V, a['d'], a['layers'], a['heads'], a['dropout'], time_bias=not a['no_time_bias'],
                 static_head=not a['no_static_head'], gpt=a['softmax_gpt']).to(device or dev)
    model.load_state_dict(torch.load(f'runs/{model_name}/{split}/{fold}/model.pt', map_location=device or dev)); model.eval()
    return Fd, model


def token_id(E, Fd, tok):
    """fold vocabulary id of a token string (its etype's UNK id if rare in the fold's pretrain events); None if never seen"""
    ix = np.where(E.tokens == tok)[0]
    return int(Fd.tid[ix[0]]) if len(ix) else None


def arrays(E, Fd, ix):
    return dict(etype=E.etype[ix].copy(), tok=Fd.tid[ix].copy(), dclass=E.dclass[ix].copy(), cohort=E.cohort[ix].copy(),
                static=E.is_static[ix].astype(np.int64), day=E.day[ix].copy(), gen=E.is_gen[ix].copy())


def pack(seqs, device=None):
    n, L = len(seqs), max(len(s['tok']) for s in seqs)
    t = {k: np.zeros((n, L), np.int64) for k in ('etype', 'tok', 'dclass', 'cohort', 'dayb', 'static')}
    day = np.zeros((n, L)); ln = np.zeros(n, np.int64)
    for i, s in enumerate(seqs):
        k = len(s['tok']); ln[i] = k
        for c in ('etype', 'tok', 'dclass', 'cohort', 'static'): t[c][i, :k] = s[c]
        t['dayb'][i, :k] = day_bucket(s['day']); day[i, :k] = s['day']; day[i, k:] = s['day'][-1]
    b = {k: torch.from_numpy(v).to(device or dev) for k, v in t.items()}
    b['day'] = torch.from_numpy(day).float().to(device or dev); b['len'] = torch.from_numpy(ln).to(device or dev)
    return b


def add_token(s, etype, tok):
    """insert after the last genomic token (most recent sequencing day); if over MAXLEN drop the oldest ordinary event"""
    g = np.where(s['gen'])[0]; p = g[-1] + 1
    new = {k: np.insert(v, p, v[g[-1]]) for k, v in s.items()}
    new['etype'][p], new['tok'][p], new['dclass'][p], new['static'][p], new['gen'][p] = etype, tok, 0, 0, True
    if len(new['tok']) > MAXLEN:
        drop = np.where(~new['gen'] & (new['static'] == 0))[0][0]
        new = {k: np.delete(v, drop) for k, v in new.items()}
    return new


def insert_event(s, etype, tok, day):
    """insert an ordinary (non-genomic) event on `day`, before trailing same-day tokens that sort after it (within-day order
    follows ETYPES, so a SCAN / MD / MET on the landmark day goes before the landmark's REG_START); keeps MAXLEN like add_token"""
    p = len(s['tok'])
    while p > 0 and s['day'][p - 1] == day and s['etype'][p - 1] > etype: p -= 1
    new = {k: np.insert(v, p, v[max(p - 1, 0)]) for k, v in s.items()}
    new['etype'][p], new['tok'][p], new['dclass'][p], new['static'][p], new['gen'][p], new['day'][p] = etype, tok, 0, 0, False, day
    if len(new['tok']) > MAXLEN:
        drop = np.where(~new['gen'] & (new['static'] == 0))[0][0]
        new = {k: np.delete(v, drop) for k, v in new.items()}
    return new


def remove_tokens(s, toks):
    keep = ~np.isin(s['tok'], toks)
    return {k: v[keep] for k, v in s.items()}


def set_tokens(s, etype, tok):
    s = {k: v.copy() for k, v in s.items()}; s['tok'][s['etype'] == etype] = tok
    return s


def logits(model, seqs, device=None, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device or dev) == 'cuda'):
                out.append(model.hazard_logits(pack(seqs[i:i + bs], device)).float())
    return torch.cat(out)


def score(model, seqs, device=None):
    """predicted 12-month risk 1 - S(12 months)"""
    return risks(logits(model, seqs, device))[1].cpu().numpy()


def survival(model, seqs, device=None):
    """monthly survival curves S(1..60 months), shape (n, 60)"""
    return torch.exp(torch.cumsum(F.logsigmoid(-logits(model, seqs, device)), 1)).cpu().numpy()
