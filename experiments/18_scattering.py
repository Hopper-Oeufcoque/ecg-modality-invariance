"""Experiment 18 — Scattering transform features (I4, novel).

Wavelet scattering coefficients are provably Lipschitz-stable to time-warp and
amplitude deformations, and translation-invariant — exactly the perturbations
modality shift induces. Mallat's theory: scattering is invariant to
deformations by construction, no training needed, so it can't overfit the
clinical distribution. This is the strongest "invariant-by-construction"
feature candidate from the synthesis report (I4).

Implementation: first- and second-order scattering via a filter bank (no Kymatio
dependency — hand-rolled so it runs anywhere). Features -> linear classifier.
Compare to E2 lead-masking and E17 single-lead+sim on the watch task.

Scientific question: does a provably deformation-stable feature representation
match or beat learned representations for modality invariance? If yes, a
training-free front-end is a strong complement (per SignalMC-MED F6, hand-
crafted features are complementary to FMs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "18_scattering"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)
WATCH_CFG = _cfg(seed=None)


# ---------------------------------------------------------------------------
# Scattering transform (hand-rolled, no external dep)
# ---------------------------------------------------------------------------

def morlet_filter(n, fs, f0, sigma):
    """Complex Morlet wavelet at frequency f0, width sigma."""
    t = np.arange(n) - n/2
    g = np.exp(-(t**2) / (2 * (n/(2*sigma))**2))
    w = g * np.exp(2j * np.pi * f0 * t / fs)
    return w


def _filterbank(fs, J=6, Q=4):
    """Log-spaced Morlet filterbank: J octaves, Q per octave."""
    filters = []
    fmax = fs / 2 * 0.9
    for j in range(J):
        for q in range(Q):
            f = fmax / (2**(j + q/Q))
            if f < 0.5: continue
            n = min(1024, int(8 * fs / f))
            filters.append((f, morlet_filter(n, fs, f, 4)))
    return filters


_FB = None
def fb():
    global _FB
    if _FB is None: _FB = _filterbank(FS)
    return _FB


def scattering_first_order(sig):
    """|x ★ ψ_j| averaged over time -> translation-invariant, deformation-stable."""
    sig = np.asarray(sig, dtype=np.float64)
    feats = []
    n = len(sig)
    X = np.fft.fft(sig)  # full FFT (Morlet is complex -> need full, not rfft)
    for f, w in fb():
        m = len(w)
        if m > n: m = n; w = w[:n]
        wp = np.zeros(n, dtype=np.complex128); wp[:m] = w[:n]
        conv = np.fft.ifft(X * np.fft.fft(wp)).real
        feats.append(np.abs(conv[:n//2]).mean())  # average over half (coarser invariance)
    return np.array(feats)


def scattering_second_order(sig):
    """||x ★ ψ_j1| ★ ψ_j2| averaged — second-order scattering (captures modulation)."""
    out = []
    bank = fb()
    n = len(sig)
    X = np.fft.fft(np.asarray(sig, dtype=np.float64))
    # subsample bank for tractability
    for f1, w1 in bank[::3]:
        m1 = len(w1); wp1 = np.zeros(n, dtype=np.complex128); wp1[:min(m1,n)] = w1[:n]
        conv1 = np.fft.ifft(X * np.fft.fft(wp1)).real
        x1 = np.abs(conv1[:n//2])
        X1 = np.fft.fft(x1, n)
        for f2, w2 in bank[::4]:
            if f2 >= f1 * 0.8: continue  # only j2 < j1
            m2 = len(w2); wp2 = np.zeros(n, dtype=np.complex128); wp2[:min(m2,n)] = w2[:n]
            conv2 = np.fft.ifft(X1 * np.fft.fft(wp2)).real
            out.append(np.abs(conv2[:n//2]).mean())
    return np.array(out)[:64]  # cap dims


def scattering_features(sig):
    s1 = scattering_first_order(sig)
    s2 = scattering_second_order(sig)
    return np.concatenate([s1, s2])


# ---------------------------------------------------------------------------
# Dataset + model
# ---------------------------------------------------------------------------

class ScatterDataset(Dataset):
    def __init__(self, records, labels_idx):
        self.records = records; self.labels_idx = labels_idx
        # precompute features (cache)
        self._cache = {}

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        if i in self._cache:
            return self._cache[i]
        rec = self.records[i]
        x = rec["ecg"][:SIGLEN]
        if x.shape[0] < SIGLEN:
            x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, WATCH_CFG, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-9)
        feat = scattering_features(leadI)
        feat = (feat - feat.mean()) / (feat.std() + 1e-9)
        item = (torch.from_numpy(feat[None].astype(np.float32)),
                torch.tensor(self.labels_idx[i]))
        self._cache[i] = item
        return item


class MLPHead(nn.Module):
    def __init__(self, dim, n_classes=N_CLASSES, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes))
    def forward(self, x):  # x: (B, 1, D) -> (B, D)
        return self.net(x.squeeze(1))


def train_model(records, labels_idx, dim, epochs=30, lr=1e-3, batch_size=64, tag="m"):
    ds = ScatterDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = MLPHead(dim, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model, ds


@torch.no_grad()
def evaluate(model, records, labels_idx, batch_size=64):
    model.eval()
    ds = ScatterDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_l=[]; all_y=[]
    for xb, yb in dl:
        all_l.append(model(xb).numpy()); all_y.append(yb.numpy())
    return np.concatenate(all_l), np.concatenate(all_y)


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    from sklearn.metrics import roc_auc_score
    aucs=[]
    for c in range(n_classes):
        yb=(y==c).astype(int)
        if yb.sum()==0 or yb.sum()==len(yb): aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, logits[:,c])))
        except Exception: aucs.append(float("nan"))
    return aucs
def macro_auroc(logits, y):
    aucs=[a for a in per_class_auroc(logits,y) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i={c:i for i,c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r['label']] for r in rs])
    tr,te=splits['train'],splits['test']; ytr,yte=lab(tr),lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    # determine feature dim
    x=splits['train'][0]['ecg'][:1000,0]; x=(x-x.mean())/(x.std()+1e-9)
    feat=scattering_features(x); dim=len(feat)
    print(f"  scattering feature dim: {dim}", flush=True)

    print("\nTraining scattering + MLP (30 ep) ...", flush=True)
    model,_=train_model(tr,ytr,dim,epochs=30,tag="scatter")
    lg,_=evaluate(model,te,yte)
    m=macro_auroc(lg,yte); pc=per_class_auroc(lg,yte)
    print(f"  Scattering @ L4: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)

    metrics={"n_train":len(tr),"n_test":len(te),"feat_dim":dim,"classes":SUPERCLASSES,
             "scattering_L4":{"macro_auroc":m,"per_class":pc}}
    (RESULTS/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    print("\nDONE.", flush=True)


if __name__=="__main__":
    main()
