"""Delphi baseline (Shmatko et al., Nature 2025; github.com/gerstung-lab/Delphi), retrained from scratch per fold on our tokens.
Model = vendored delphi_model.py unchanged (GPT-2, continuous-time sinusoidal encoding instead of positions, tie masking,
next-token CE + exponential waiting-time loss). Adaptations: time axis = days since index-cancer diagnosis (Delphi: age in days);
DEATH token appended at the death day for the fold's pretrain patients only (training target, never an input at a landmark);
NOEVENT tokens at random days up to death / last-alive (Delphi's censoring mechanism; one per NOEVENT_RATE days on average);
STATIC tokens ignored as targets (as Delphi ignores sex / lifestyle). Same fold vocabulary, truncation, landmark prefixes and
fixed 10% validation hold-out as train.py. Risk at a landmark = Delphi's death rate exp(logit_DEATH) at the landmark token.
No survival fine-tuning (Delphi native). usage: python delphi_train.py --split random5 --fold fold0"""
import argparse, json, os, time, numpy as np, pandas as pd, torch
from data import Events, Fold, truncate, MAXLEN, MAXGEN
from delphi_model import Delphi, DelphiConfig

p = argparse.ArgumentParser()
p.add_argument('--split', default='random5'); p.add_argument('--fold', default='fold0'); p.add_argument('--model', default='Delphi')
p.add_argument('--n_layer', type=int, default=12); p.add_argument('--n_head', type=int, default=12); p.add_argument('--n_embd', type=int, default=120)
p.add_argument('--dropout', type=float, default=0.1); p.add_argument('--wd', type=float, default=0.2)
p.add_argument('--lr', type=float, default=6e-4); p.add_argument('--bs', type=int, default=64); p.add_argument('--epochs', type=int, default=60)
p.add_argument('--patience', type=int, default=10); p.add_argument('--noevent_rate', type=float, default=180.)
p.add_argument('--t_min', type=float, default=0.0); p.add_argument('--seed', type=int, default=0); p.add_argument('--out', default='runs')
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
odir = os.path.join(a.out, a.model, a.split, a.fold); os.makedirs(odir, exist_ok=True)
log = open(os.path.join(odir, 'log.txt'), 'w')
def say(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
say(vars(a)); t0 = time.time()

E = Events(); Fd = Fold(E, a.split, a.fold); lm = E.lm
V = Fd.V; DEATH, NOEVENT = V, V + 1
static_ids = [int(i) for i in np.where(Fd.vtype == 1)[0]]
say(f'vocab {V}+2  pretrain {len(Fd.pretrain)}  train {len(Fd.train)}  test {len(Fd.test)}')

# per-patient end of follow-up (death or last alive), from the same regimen-level OS labels as the landmarks
lab = pd.read_pickle('labels_raw.pkl')
end = lab.assign(end=lab.landmark_day + lab.tt_os_g_days).groupby('patient_id').agg(end=('end', 'max'), ev=('os_g_status', 'max'))
END = end.end.to_dict(); DIED = end.ev.to_dict()
pretrain_set = set(Fd.pretrain)


def patient_seq(pid, r):
    """full pretraining trajectory: events (<= end of follow-up) + NOEVENT + DEATH; returns (tok, day, protected)"""
    assert pid in pretrain_set
    ix = E.pat_ix[pid]; tok = Fd.tid[ix]; day = E.day[ix]; prot = E.is_static[ix] | E.is_gen[ix]; order = np.arange(len(ix), dtype=float)
    stop = END.get(pid, day.max())
    keep = day <= stop
    tok, day, prot, order = tok[keep], day[keep], prot[keep], order[keep]
    n = r.poisson(max(stop - day.min(), 0) / a.noevent_rate)
    nd = np.sort(r.uniform(day.min(), stop, n)) if n else np.zeros(0)
    xt = [np.full(n, NOEVENT)]; xd = [nd]; xo = [np.full(n, 1e6)]
    if DIED.get(pid, 0) == 1:
        xt.append([DEATH]); xd.append([stop]); xo.append([2e6])
    tok = np.concatenate([tok] + xt); day = np.concatenate([day] + xd); order = np.concatenate([order] + xo)
    prot = np.concatenate([prot, np.zeros(len(tok) - len(prot), bool)])
    s = np.lexsort((order, day))
    return tok[s], day[s], prot[s]


def trunc_arrays(tok, day, prot, r):
    """window (random end, or the full trajectory half the time) + truncation keeping static/genomic (as data.truncate)"""
    L = MAXLEN + 1                                       # +1: inputs and targets are shifted copies
    if len(tok) > L and r.rand() < 0.5:
        e = r.randint(L, len(tok) + 1); tok, day, prot = tok[:e], day[:e], prot[:e]
    if len(tok) <= L:
        return tok, day
    pi, oi = np.where(prot)[0], np.where(~prot)[0]
    st = pi                                              # static + genomic positions
    if len(st) > MAXGEN: st = np.concatenate([st[:5], st[5:][-(MAXGEN - 5):]])
    oi = oi[len(oi) - max(L - len(st), 0):] if L - len(st) > 0 else oi[:0]
    k = np.sort(np.concatenate([st, oi]))
    return tok[k], day[k]


def pack(seqs):
    n = len(seqs); T = max(len(t) for t, _ in seqs) - 1
    X = np.zeros((n, T), np.int64); A = np.full((n, T), -10000., np.float32); Y = np.zeros((n, T), np.int64); B = np.full((n, T), -10000., np.float32)
    for i, (t, d) in enumerate(seqs):
        k = len(t) - 1
        X[i, :k], A[i, :k], Y[i, :k], B[i, :k] = t[:-1], d[:-1], t[1:], d[1:]
    return [torch.from_numpy(z).to(dev) for z in (X, A, Y, B)]


def split_patients(ps, frac=0.1):
    ps = np.array(sorted(ps)); np.random.RandomState(12345).shuffle(ps); k = max(1, int(len(ps) * frac))
    return list(ps[k:]), list(ps[:k])


cfg = DelphiConfig(block_size=MAXLEN + 1, vocab_size=V + 2, n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd, dropout=a.dropout,
                   t_min=a.t_min, bias=False, mask_ties=True, ignore_tokens=[0] + static_ids)
model = Delphi(cfg).to(dev)
opt = model.configure_optimizers(a.wd, a.lr, (0.9, 0.99), 'cuda')
ptr, pva = split_patients(Fd.pretrain)
vr = np.random.RandomState(123); va_seqs = [trunc_arrays(*patient_seq(q, vr), vr) for q in pva]


def run_loss(seqs, train):
    model.train(train); tot = n = 0
    lens = np.array([len(t) for t, _ in seqs]); o = np.argsort(lens + rng.rand(len(lens)) * 8)
    batches = [o[i:i + a.bs] for i in range(0, len(o), a.bs)]
    if train: batches = [batches[i] for i in rng.permutation(len(batches))]
    for bi in batches:
        X, A, Y, B = pack([seqs[i] for i in bi])
        with torch.set_grad_enabled(train):
            _, loss, _ = model(X, A, Y, B)
        l = loss['loss_ce'] + loss['loss_dt']
        if train:
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += l.item() * len(bi); n += len(bi)
    return tot / n


def landmark_rates(sel):
    """death rate at each landmark from its visible prefix only. Tie masking needs next-event times: within the prefix they
    are the prefix's own next tokens; for the landmark position a placeholder time after the landmark day (no future info)."""
    out = []
    for i in range(0, len(sel), 32):                    # small batches: Delphi returns all layers' attention maps
        seqs = []
        for lid in lm.landmark_id.values[sel[i:i + 32]]:
            ix = truncate(E, E.P[lid]); t = Fd.tid[ix]; d = E.day[ix]
            seqs.append((np.append(t, 0), np.append(d, d[-1] + 1)))       # dummy target (token 0 = ignored) after the landmark
        X, A, Y, B = pack(seqs)
        with torch.no_grad():
            logits, _, _ = model(X, A, Y, B)
        last = torch.tensor([len(s[0]) - 2 for s in seqs], device=dev)
        out.append(logits.float()[torch.arange(len(seqs), device=dev), last, DEATH].cpu().numpy())
    return np.concatenate(out)


from lifelines.utils import concordance_index
va_sel = np.where(lm.patient_id.isin(set(pva)).values)[0]                # landmarks of patients held out of Delphi's training


def val_c():
    model.eval()
    return concordance_index(lm.os_time.values[va_sel], -landmark_rates(va_sel), lm.os_event.values[va_sel])


def fit(patients, epochs, va=None, tag='ep'):
    """train for `epochs`; with va: early-stop on inner-val C-index of the death rate at landmarks of held-out pretrain patients
    (Delphi's generative loss keeps improving after death-risk ranking peaks); returns (best C, best epoch count, best state)"""
    best, bad, best_ep, best_state = -1, 0, epochs, None
    for ep in range(epochs):
        lr = a.lr * min(1, (ep + 1) / 3) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * ep / a.epochs)))   # warmup + cosine to lr/10
        for g in opt.param_groups: g['lr'] = lr
        tl = run_loss([trunc_arrays(*patient_seq(q, rng), rng) for q in patients], True)
        if va is None:
            say(f'{tag} {ep} train {tl:.4f} ({time.time() - t0:.0f}s)'); continue
        vl = run_loss(va, False); vc = val_c()
        say(f'{tag} {ep} train {tl:.4f} val {vl:.4f} val_C {vc:.4f} lr {lr:.2e} ({time.time() - t0:.0f}s)')
        if vc > best + 1e-4:
            best, bad, best_ep, best_state = vc, 0, ep + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    return best, best_ep, best_state


best, best_ep, best_state = fit(ptr, a.epochs, va_seqs)
model.load_state_dict(best_state); model.eval()


from lifelines.utils import concordance_index
vc = best
say(f'best inner-val C (death rate) {vc:.4f} at {best_ep} epochs on {len(va_sel)} landmarks')
# refit from scratch on ALL pretrain patients for best_ep epochs (same data as HSTU, which refits on all train patients)
torch.manual_seed(a.seed); model = Delphi(cfg).to(dev); opt = model.configure_optimizers(a.wd, a.lr, (0.9, 0.99), 'cuda')
fit(Fd.pretrain, best_ep, tag='refit'); model.eval()
te_sel = np.where(lm.patient_id.isin(set(Fd.test)).values)[0]
lr_ = landmark_rates(te_sel); rate = np.exp(lr_)                         # events per day
pred = pd.DataFrame({'landmark_id': lm.landmark_id.values[te_sel], 'split': a.split, 'fold': a.fold, 'model': a.model,
                     'risk': lr_, 'risk12': 1 - np.exp(-rate * 365.25), 'risk24': 1 - np.exp(-rate * 730.5)})
pred.to_parquet(os.path.join(odir, 'predictions.parquet'), index=False)
torch.save(model.state_dict(), os.path.join(odir, 'model.pt'))
json.dump(dict(vars(a), val_c=vc, epochs_chosen=best_ep, minutes=(time.time() - t0) / 60), open(os.path.join(odir, 'summary.json'), 'w'))
say(f'done {len(pred)} test landmarks ({(time.time() - t0) / 60:.1f} min)')
