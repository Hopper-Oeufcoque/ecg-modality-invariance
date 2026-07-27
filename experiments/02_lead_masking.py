"""Experiment 2 — Closing the lead-count gap (the dominant axis from E1).

E1 proved lead-count is the war (-0.338 of -0.311 AUROC) and that naive
12-lead->single-lead transfer collapses to near-chance (0.527) because the
model never saw missing leads. E2 attacks that axis directly with the proven
method (K-MERL lead-masking, C9/E2) and a single-lead reference model.

Variants (each trained 20 ep, tested on L1 clean-Lead-I and L4 full-watch):
  V1 Naive        12-lead, no aug               (E1 baseline, retrained in-run)
  V2 LeadMask     12-lead, random lead masking  (K-MERL C9)
  V3 WatchAug     12-lead, watch-sim on Lead-I  (E1 R2, in-run)
  V4 LeadMask+Aug 12-lead, both                  (the combo)
  V5 SingleLead   1-lead model, trained on Lead-I (target-domain reference)

Scientific question: can lead-masking + watch-aug close the gap WITHOUT any
target-domain (watch) labels, and how far is that combo from the single-lead
model that gets to train directly on Lead-I?
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
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "02_lead_masking"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# Dataset with configurable augmentation
# ---------------------------------------------------------------------------

class AugDataset(Dataset):
    """Training dataset with independent lead-masking and watch-sim augs.

    n_leads=12 -> multilead model (Lead-I in ch0). n_leads=1 -> single-lead
    model trained on Lead-I only.
    """

    def __init__(self, records, labels_idx,
                 lead_mask_prob=0.0,       # per-extra-lead dropout prob
                 watch_sim_cfg=None,       # if set, apply watch sim to Lead-I
                 n_leads=12, fs=FS, siglen=SIGLEN):
        self.records = records
        self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob
        self.watch_sim_cfg = watch_sim_cfg
        self.n_leads = n_leads
        self.fs = fs
        self.siglen = siglen

    def __len__(self):
        return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        if T >= self.siglen:
            return x[:self.siglen]
        pad = np.zeros((self.siglen - T, x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=0)

    def __getitem__(self, i):
        rec = self.records[i]
        x = self._fixlen(rec["ecg"])  # (T,12)
        # per-record norm
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)

        # watch-sim augmentation on Lead-I (A5)
        if self.watch_sim_cfg is not None:
            out = simulate_watch(x, self.fs, self.watch_sim_cfg, LEAD_NAMES, rng=rng)
            watch = out["watch"]  # (T,) at fs_watch (==fs here)
            x = x.copy()
            x[:, 0] = watch

        # lead masking (K-MERL C9): drop extra leads, ALWAYS keep Lead-I
        if self.lead_mask_prob > 0.0 and self.n_leads == 12:
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False  # keep Lead I
            x = x.copy()
            x[:, mask] = 0.0

        if self.n_leads == 1:
            x = x[:, 0:1]  # Lead-I only
        y = self.labels_idx[i]
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(y)


def train_model(records, labels_idx, n_leads=12,
                lead_mask_prob=0.0, watch_sim_cfg=None,
                epochs=20, lr=1e-3, batch_size=64, tag="m"):
    ds = AugDataset(records, labels_idx, lead_mask_prob, watch_sim_cfg, n_leads=n_leads)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(n_leads, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Eval (handles both 12-lead and 1-lead models)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, records, labels_idx, stage, n_leads=12, batch_size=64):
    """stage: 'L1' clean Lead-I, 'L4' full watch on Lead-I."""
    model.eval()
    cfg_L1 = _cfg(apply_bandwidth=False, apply_electrode=False,
                  apply_noise=False, apply_quantization=False, seed=0)
    cfg_L4 = _cfg(seed=0)
    cfg = cfg_L1 if stage == "L1" else cfg_L4
    all_logits = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            # simulate Lead-I (clean or full watch)
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]  # (T,)
            # renormalize the distorted Lead-I
            mu1 = leadI.mean(); sd1 = leadI.std() + 1e-6
            leadI = (leadI - mu1) / sd1
            if n_leads == 12:
                framed = np.zeros((SIGLEN, 12), dtype=np.float32)
                framed[:, 0] = leadI
            else:
                framed = leadI[:, None].astype(np.float32)
            batch.append(framed)
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        all_logits.append(model(xb).numpy())
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
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    variants = [
        ("V1_Naive",        dict(n_leads=12, lead_mask_prob=0.0, watch_sim_cfg=None)),
        ("V2_LeadMask",     dict(n_leads=12, lead_mask_prob=0.5, watch_sim_cfg=None)),
        ("V3_WatchAug",     dict(n_leads=12, lead_mask_prob=0.0, watch_sim_cfg=_cfg(seed=None))),
        ("V4_LeadMask+Aug", dict(n_leads=12, lead_mask_prob=0.5, watch_sim_cfg=_cfg(seed=None))),
        ("V5_SingleLead",   dict(n_leads=1,  lead_mask_prob=0.0, watch_sim_cfg=None)),
    ]

    results = {}
    for name, kw in variants:
        print(f"\n=== Training {name} ===", flush=True)
        model = train_model(tr, ytr, epochs=20, tag=name, **kw)
        r = {}
        for stage in ("L1", "L4"):
            logits, y = evaluate(model, te, yte, stage, n_leads=kw["n_leads"])
            m = macro_auroc(logits, y); pc = per_class_auroc(logits, y)
            r[stage] = {"macro_auroc": m, "per_class": pc}
            print(f"  {name} @ {stage}: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)
        results[name] = r

    metrics = {"fs": FS, "n_train": len(tr), "n_test": len(te),
               "classes": SUPERCLASSES, "variants": results}
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
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    b2 = ax.bar(x + w/2, l4, w, label="L4 full watch", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("E2: closing the lead-count gap (12-lead model tested on single-lead)")
    ax.axhline(0.865, color="green", ls="--", lw=1, label="L0 clinical ceiling 0.865")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance 0.5")
    ax.legend(fontsize=8, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "ladder.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
