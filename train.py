"""Per fold, per variant, from scratch: (a) next-event pretraining on the fold's `pretrain` patients, (b) discrete-time hazard
fine-tuning on `train` landmarks, (c) predictions for TEST landmarks only -> runs/<model>/<split>/<fold>/predictions.parquet.
Fine-tuning: the epoch count is chosen by early stopping on a fixed 10% patient-level hold-out of `train`, then the model is refit
from the same pretrained weights on ALL `train` patients for that many epochs (same labelled data as the baselines). Never test.
usage: python train.py --split random5 --fold fold0 [--model HSTU] [--no_time_bias] [--no_genomic] [--no_static_head] [--softmax_gpt]"""
import argparse, json, os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from lifelines.utils import concordance_index
from data import Events, Fold, ETYPES, COHORTS, NTB, MAXLEN, gap_bucket, surv_targets
from hstu import HSTU, surv_loss, risks

p = argparse.ArgumentParser()
p.add_argument('--split', default='random5'); p.add_argument('--fold', default='fold0'); p.add_argument('--model', default='HSTU')
p.add_argument('--cohort', default=None, help='restrict to one cohort (pilot)')
p.add_argument('--d', type=int, default=128); p.add_argument('--layers', type=int, default=3); p.add_argument('--heads', type=int, default=4)
p.add_argument('--dropout', type=float, default=0.2); p.add_argument('--seed', type=int, default=0)
p.add_argument('--bs', type=int, default=64); p.add_argument('--pt_lr', type=float, default=1e-3); p.add_argument('--ft_lr', type=float, default=3e-4)
p.add_argument('--pt_epochs', type=int, default=40); p.add_argument('--ft_epochs', type=int, default=40); p.add_argument('--patience', type=int, default=4)
p.add_argument('--no_pretrain', action='store_true')
p.add_argument('--shuffle_labels', action='store_true', help='control: permute (os_time, os_event) across train landmarks; test labels untouched')
p.add_argument('--no_time_bias', action='store_true'); p.add_argument('--no_genomic', action='store_true')
p.add_argument('--no_static_head', action='store_true'); p.add_argument('--softmax_gpt', action='store_true')
p.add_argument('--out', default='runs')
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed); rng = np.random.RandomState(a.seed)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
odir = os.path.join(a.out, a.model, a.split, a.fold); os.makedirs(odir, exist_ok=True)
log = open(os.path.join(odir, 'log.txt'), 'w')
def say(*s):
    m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
say(vars(a))
t0 = time.time()

E = Events()
Fd = Fold(E, a.split, a.fold, no_genomic=a.no_genomic, restrict_cohort=a.cohort)
say(f'vocab {Fd.V}  pretrain {len(Fd.pretrain)}  train {len(Fd.train)}  test {len(Fd.test)}  ({time.time() - t0:.0f}s)')
unk = [COHORTS.index(a.fold) + 1] if a.split == 'loco' else None     # held-out cohort -> UNKNOWN cohort id
vtype = torch.from_numpy(Fd.vtype).to(dev)
model = HSTU(Fd.V, a.d, a.layers, a.heads, a.dropout, time_bias=not a.no_time_bias,
             static_head=not a.no_static_head, gpt=a.softmax_gpt).to(dev)
say('params', sum(x.numel() for x in model.parameters()))


ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda')


def to_dev(b, ix=None):
    if ix is not None:
        L = int(b['len'][ix].max())
        b = {k: (v[ix, :L] if v.dim() == 2 else v[ix]) for k, v in b.items()}
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def cohort_dropout(b):
    m = torch.rand(b['cohort'].shape[0], device=dev) < 0.15
    b['cohort'] = torch.where(m[:, None], torch.zeros_like(b['cohort']), b['cohort'])
    return b


def length_batches(lengths, bs):
    """batches of similar length (less padding), in random order"""
    o = np.argsort(np.asarray(lengths) + rng.rand(len(lengths)) * 8)
    bs_ = [o[i:i + bs] for i in range(0, len(o), bs)]
    return [bs_[i] for i in rng.permutation(len(bs_))]


def split_patients(ps, frac=0.1):
    """fixed 10% patient-level hold-out (same for every model and seed, so gbsa_ref.py can reproduce it)"""
    ps = np.array(sorted(ps)); np.random.RandomState(12345).shuffle(ps); k = max(1, int(len(ps) * frac))
    return list(ps[k:]), list(ps[:k])


# ---------------- (a) pretraining
def pt_loss(b):
    h = model.encode(b)[:, :-1]
    et, tok, day, L = b['etype'][:, 1:], b['tok'][:, 1:], b['day'], b['len']
    ar = torch.arange(h.shape[1], device=dev)
    w = (ar[None, :] + 1 < L[:, None]).float()
    w = w * torch.where(et == 1, 0.1, 1.0)                       # static targets down-weighted
    gap = (day[:, 1:] - day[:, :-1]).clamp(min=0)
    tb = torch.minimum(torch.floor(torch.log2(gap + 1)), torch.tensor(NTB - 1., device=dev)).long()
    lt = F.cross_entropy(model.h_type(h).transpose(1, 2), et, reduction='none')
    lv_logits = model.h_val(h).masked_fill(vtype[None, None, :] != et[..., None], -1e4)
    lv = F.cross_entropy(lv_logits.transpose(1, 2), tok, reduction='none')
    ltime = F.cross_entropy(model.h_time(h).transpose(1, 2), tb, reduction='none')
    l = lt + lv + 0.5 * ltime
    return (l * w).sum() / w.sum(), w.sum()


if not a.no_pretrain:
    ptr, pva = split_patients(Fd.pretrain)
    va_b = Fd.pack(Fd.pretrain_seqs(pva, np.random.RandomState(123)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.pt_lr, weight_decay=0.01)
    steps_ep = (len(ptr) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.pt_lr, total_steps=steps_ep * a.pt_epochs, pct_start=0.05)
    best, bad, best_state = 1e9, 0, None
    for ep in range(a.pt_epochs):
        model.train()
        tr_b = Fd.pack(Fd.pretrain_seqs(ptr, rng))
        tl = 0
        for ix in length_batches(tr_b['len'].numpy(), a.bs):
            b = cohort_dropout(to_dev(tr_b, torch.from_numpy(ix)))
            with ac(): loss, _ = pt_loss(b)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item()
        model.eval(); vl = vw = 0
        with torch.no_grad():
            for i in range(0, len(pva), 256):
                with ac(): l, w = pt_loss(to_dev(va_b, torch.arange(i, min(i + 256, len(pva)))))
                vl += l.item() * w.item(); vw += w.item()
        vl /= vw
        say(f'pt ep {ep} train {tl / steps_ep:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)')
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state)

# ---------------- (b) fine-tuning
lm = E.lm
ftr, fva = split_patients(Fd.train)
os_t, os_e = lm.os_time.values.copy(), lm.os_event.values.copy()
if a.shuffle_labels:                                   # shuffled-label control: labels of train landmarks permuted among themselves
    tri = np.where(lm.patient_id.isin(set(Fd.train)).values)[0]; perm = np.random.RandomState(999).permutation(tri)
    os_t[tri], os_e[tri] = os_t[perm], os_e[perm]


def landmark_batch(patients, unk_cohort=None):
    sel, seqs = Fd.landmark_seqs(patients)
    b = Fd.pack(seqs, unk_cohort)
    k, ev = surv_targets(os_t[sel], os_e[sel])
    b['k'], b['ev'] = k, ev
    return sel, b

def patient_groups(sel):
    up, inv = np.unique(lm.patient_id.values[sel], return_inverse=True)
    return [np.where(inv == i)[0] for i in range(len(up))]


def predict(b, n):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, n, 256):
            bb = to_dev(b, torch.arange(i, min(i + 256, n)))
            with ac(): out.append(model.hazard_logits(bb).float())
    return torch.cat(out)


def fine_tune(tr_b, groups, epochs, va=None, tag='ft'):
    """train the hazard head + backbone; with va=(sel, batch) early-stop on val loss and return the best epoch count"""
    head = list(model.h_surv.parameters()); hid = {id(x) for x in head}
    opt = torch.optim.AdamW([{'params': [x for x in model.parameters() if id(x) not in hid], 'lr': a.ft_lr},
                             {'params': head, 'lr': a.ft_lr * 3}], weight_decay=0.01)
    glen = [int(tr_b['len'][g].max()) for g in groups]
    best, bad, best_ep, best_c = 1e9, 0, epochs, np.nan
    for ep in range(epochs):
        model.train(); tl = nb = 0
        for pbatch in length_batches(glen, max(1, a.bs // 2)):             # batches of patients, all their landmarks
            ix = torch.from_numpy(np.concatenate([groups[j] for j in pbatch]))
            b = cohort_dropout(to_dev(tr_b, ix))
            with ac(): lg = model.hazard_logits(b).float()
            loss = surv_loss(lg, b['k'], b['ev'])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += loss.item(); nb += 1
        if va is None:
            say(f'{tag} ep {ep} train {tl / nb:.4f} ({time.time() - t0:.0f}s)'); continue
        va_sel, va_b = va
        lg = predict(va_b, len(va_sel))
        vl = surv_loss(lg, va_b['k'].to(dev), va_b['ev'].to(dev)).item()
        vc = concordance_index(os_t[va_sel], -risks(lg)[0].cpu().numpy(), os_e[va_sel])
        say(f'{tag} ep {ep} train {tl / nb:.4f} val {vl:.4f} val_C {vc:.4f} ({time.time() - t0:.0f}s)')
        if vl < best - 1e-4:
            best, bad, best_ep, best_c = vl, 0, ep + 1, vc
        else:
            bad += 1
            if bad >= a.patience: break
    return best, best_ep, best_c


# (b1) choose the number of fine-tuning epochs on the 10% inner-validation patients
tr_sel, tr_b = landmark_batch(ftr); va_sel, va_b = landmark_batch(fva); te_sel, te_b = landmark_batch(Fd.test, unk)
pt_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
best, best_ep, best_c = fine_tune(tr_b, patient_groups(tr_sel), a.ft_epochs, va=(va_sel, va_b))
say(f'best val loss {best:.4f} val_C {best_c:.4f} at {best_ep} epochs')
# (b2) refit from the same pretrained weights on ALL train patients (same labelled data as the baselines) for best_ep epochs
model.load_state_dict(pt_state)
all_sel, all_b = landmark_batch(Fd.train)
fine_tune(all_b, patient_groups(all_sel), best_ep, tag='refit')

# ---------------- (c) test predictions only
lg = predict(te_b, len(te_sel))
r, r12, r24 = (x.cpu().numpy() for x in risks(lg))
pred = pd.DataFrame({'landmark_id': lm.landmark_id.values[te_sel], 'split': a.split, 'fold': a.fold, 'model': a.model,
                     'risk': r, 'risk12': r12, 'risk24': r24})
pred.to_parquet(os.path.join(odir, 'predictions.parquet'), index=False)
torch.save(model.state_dict(), os.path.join(odir, 'model.pt'))
json.dump(dict(vars(a), val_loss=best, val_c=best_c, ft_epochs_chosen=best_ep, minutes=(time.time() - t0) / 60), open(os.path.join(odir, 'summary.json'), 'w'))
say(f'done {len(pred)} test landmarks ({(time.time() - t0) / 60:.1f} min)')
