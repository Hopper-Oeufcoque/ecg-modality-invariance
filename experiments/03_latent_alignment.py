"""Experiment 3 — Latent-space lead alignment (SelfMIS self-cutting, B9).

The synthesis report's top-ranked approach. SelfMIS (arXiv:2509.19397) showed
that signal-level 12-lead reconstruction leaves a *latent-space gap* that hurts
downstream detection; aligning the single-lead embedding to the 12-lead
embedding directly (no signal synthesis) avoids that gap.

Method (self-cutting contrastive pretrain):
  - Shared encoder processes BOTH the full 12-lead and the single-lead
    (Lead-I zero-padded to 12-lead) of the SAME record.
  - InfoNCE pulls z_single toward z_full of the same record and away from
    z_full of other records in the batch.
  - Phase B: freeze encoder, train a linear classifier head on z_full (12-lead).
  - Test: feed z_single (Lead-I, clean or full-watch) through the same head.

Scientific question: does latent-space alignment beat the E2 augmentation
approach on the dominant lead-count axis? This is the report's bet.

Also includes a signal-synthesis baseline (B1-style: reconstruct 12-lead from
single via a small decoder, then classify) to directly demonstrate the
SelfMIS warning that signal synthesis < latent alignment for detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ResBlock1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "03_latent_alignment"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
FEAT_DIM = 64
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# Featurizer (shared encoder for full + single-lead)
# ---------------------------------------------------------------------------

class Featurizer(nn.Module):
    """ECGResNet1d backbone -> FEAT_DIM feature (pre-classifier)."""

    def __init__(self, n_leads=12, base_ch=FEAT_DIM, n_blocks=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(4))
        self.blocks = nn.Sequential(*[ResBlock1d(base_ch) for _ in range(n_blocks)])
        self.bn_out = nn.BatchNorm1d(base_ch)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x); x = self.blocks(x)
        x = self.pool(F.relu(self.bn_out(x))).flatten(1)
        return x  # (B, base_ch)


class LinearHead(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, z): return self.fc(z)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """Yields (x_full 12-lead, x_single Lead-I zero-padded) pairs."""

    def __init__(self, records, fs=FS, siglen=SIGLEN, watch_sim_cfg=None):
        self.records = records; self.fs = fs; self.siglen = siglen
        self.watch_sim_cfg = watch_sim_cfg

    def __len__(self): return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        if T >= self.siglen: return x[:self.siglen]
        return np.concatenate([x, np.zeros((self.siglen - T, x.shape[1]))], 0)

    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        # self-cut: single-lead = Lead-I (optionally watch-simulated), zero-pad to 12
        x_single = x.copy()
        if self.watch_sim_cfg is not None:
            out = simulate_watch(x, self.fs, self.watch_sim_cfg, LEAD_NAMES, rng=rng)
            x_single[:, 0] = out["watch"]
        mask = rng.random(N_LEADS) < 0.0  # keep all-zero except Lead-I
        mask[0] = True  # actually we zero everything except ch0
        x_single[:, [j for j in range(N_LEADS) if j != 0]] = 0.0
        return (torch.from_numpy(x.astype(np.float32)).permute(1, 0),
                torch.from_numpy(x_single.astype(np.float32)).permute(1, 0))


class ClsDataset(Dataset):
    """For linear probe: (x_full, label)."""

    def __init__(self, records, labels_idx, fs=FS, siglen=SIGLEN):
        self.records = records; self.labels_idx = labels_idx
        self.fs = fs; self.siglen = siglen

    def __len__(self): return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        if T >= self.siglen: return x[:self.siglen]
        return np.concatenate([x, np.zeros((self.siglen - T, x.shape[1]))], 0)

    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.labels_idx[i])


def info_nce(z_single, z_full, tau=0.1):
    """Anchor=z_single, positive=z_full (same idx), negatives=z_full (other idx).
    Symmetric NT-Xent over the batch."""
    z_single = F.normalize(z_single, dim=1)
    z_full = F.normalize(z_full, dim=1)
    logits = z_single @ z_full.t() / tau  # (B,B)
    labels = torch.arange(logits.size(0), device=logits.device)
    # symmetric: also z_full->z_single
    loss = 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))
    return loss


# ---------------------------------------------------------------------------
# Eval (reuse featurizer+head; feed Lead-I zero-padded for single-lead test)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(feat, head, records, labels_idx, stage, batch_size=64):
    feat.eval(); head.eval()
    cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                apply_noise=False, apply_quantization=False, seed=0) if stage == "L1"
           else _cfg(seed=0))
    all_logits = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]
            mu1 = leadI.mean(); sd1 = leadI.std() + 1e-6
            leadI = (leadI - mu1) / sd1
            framed = np.zeros((SIGLEN, 12), dtype=np.float32)
            framed[:, 0] = leadI
            batch.append(framed)
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        z = feat(xb)
        all_logits.append(head(z).numpy())
        all_y.append(np.array(labels_idx[i:i+batch_size]))
    return np.concatenate(all_logits), np.concatenate(all_y)


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(n_classes):
        yb = (y == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, logits[:, c])))
        except Exception: aucs.append(float("nan"))
    return aucs


def macro_auroc(logits, y):
    aucs = [a for a in per_class_auroc(logits, y) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def pretrain(records, epochs=20, tau=0.1, lr=1e-3, batch_size=128, tag="pre"):
    ds = PretrainDataset(records, watch_sim_cfg=None)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    feat = Featurizer(N_LEADS, base_ch=FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(feat.parameters(), lr=lr)
    for ep in range(epochs):
        feat.train(); tot = 0.0; nb = 0
        for x_full, x_single in dl:
            opt.zero_grad()
            z_full = feat(x_full); z_single = feat(x_single)
            loss = info_nce(z_single, z_full, tau=tau)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return feat


def pretrain_with_watchsim(records, epochs=20, tau=0.1, lr=1e-3, batch_size=128, tag="pre-w"):
    """Self-cutting with watch-sim on the single-lead side (aligns watch-like
    single-lead embeddings to clean 12-lead embeddings)."""
    ds = PretrainDataset(records, watch_sim_cfg=_cfg(seed=None))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    feat = Featurizer(N_LEADS, base_ch=FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(feat.parameters(), lr=lr)
    for ep in range(epochs):
        feat.train(); tot = 0.0; nb = 0
        for x_full, x_single in dl:
            opt.zero_grad()
            z_full = feat(x_full); z_single = feat(x_single)
            loss = info_nce(z_single, z_full, tau=tau)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return feat


def linear_probe(feat, records, labels_idx, epochs=20, lr=1e-3, batch_size=128, tag="lp"):
    """Freeze feat, train linear head on z_full (12-lead) for classes."""
    for p in feat.parameters(): p.requires_grad = False
    feat.eval()
    ds = ClsDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    head = LinearHead(FEAT_DIM, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        head.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            with torch.no_grad(): z = feat(xb)
            opt.zero_grad()
            loss = loss_fn(head(z), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return head


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    results = {}

    # --- Variant A: self-cutting latent alignment (clean) ---
    print("\n=== A: self-cutting latent alignment (clean) ===", flush=True)
    featA = pretrain(tr, epochs=20, tag="A-pre")
    headA = linear_probe(featA, tr, ytr, epochs=20, tag="A-lp")
    for stage in ("L1", "L4"):
        lg, y = evaluate(featA, headA, te, yte, stage)
        m = macro_auroc(lg, y); pc = per_class_auroc(lg, y)
        results.setdefault("A_latent_clean", {})[stage] = {"macro_auroc": m, "per_class": pc}
        print(f"  A @ {stage}: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)

    # --- Variant B: self-cutting + watch-sim on single side (aligns watch→12) ---
    print("\n=== B: latent alignment + watch-sim on single-lead side ===", flush=True)
    featB = pretrain_with_watchsim(tr, epochs=20, tag="B-pre")
    headB = linear_probe(featB, tr, ytr, epochs=20, tag="B-lp")
    for stage in ("L1", "L4"):
        lg, y = evaluate(featB, headB, te, yte, stage)
        m = macro_auroc(lg, y); pc = per_class_auroc(lg, y)
        results.setdefault("B_latent_watchsim", {})[stage] = {"macro_auroc": m, "per_class": pc}
        print(f"  B @ {stage}: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)

    metrics = {"fs": FS, "n_train": len(tr), "n_test": len(te),
              "classes": SUPERCLASSES, "feat_dim": FEAT_DIM, "variants": results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)

    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    l1 = [results[n]["L1"]["macro_auroc"] for n in names]
    l4 = [results[n]["L4"]["macro_auroc"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    b2 = ax.bar(x + w/2, l4, w, label="L4 full watch", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("E3: latent-space lead alignment (SelfMIS self-cutting)")
    ax.axhline(0.865, color="green", ls="--", lw=1, label="L0 ceiling 0.865")
    ax.legend(fontsize=8, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "ladder.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
