"""Experiment 5 — Test-time BatchNorm adaptation (H3), layered on the E2 winner.

E2's winner is V2 LeadMask (lead-masking). It reaches 0.717 on full watch.
E5 asks: can cheap, label-free test-time adaptation (BN adaptation, the classic
CV TTA method adapted to 1D) push that further, by re-aligning the model's
batch statistics to the target (watch) distribution at inference?

This is the cheapest possible post-hoc invariance: no retraining, no labels,
just recompute BN running stats from the target batch.

Protocol:
  1. Train V2 LeadMask (the E2 winner) on 12-lead PTB-XL.
  2. Test on L4 full watch under:
     (a) no TTA  — BN stats from training distribution
     (b) BN-adapt — reset BN running stats, recompute from the L4 target batch
  3. Per-class breakdown.

Note: BN-adapt here is *batch-level* (whole target set). Per-clip TTA on a
single 30 s clip is under-powered for BN stats (needs a batch); a follow-up
could use entropy-min on a LoRA adapter instead (H4). Flagged as limitation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

# reuse E2's training + dataset + eval plumbing
_spec = importlib.util.spec_from_file_location(
    "e2", Path(__file__).resolve().parents[1] / "experiments" / "02_lead_masking.py")
e2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e2)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "05_tta"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# BN adaptation
# ---------------------------------------------------------------------------

def reset_bn(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()


def bn_adapt(model, target_tensors, n_passes=1):
    """Recompute BN running stats from target data (no labels, no grad)."""
    reset_bn(model)
    model.train()  # BN updates running stats in train mode
    with torch.no_grad():
        for _ in range(n_passes):
            for xb in target_tensors:
                model(xb)
    model.eval()


def make_target_tensors(records, stage="L4", batch_size=64):
    """Build the framed Lead-I (L1 or L4) tensors for a set of records."""
    cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                apply_noise=False, apply_quantization=False, seed=0)
           if stage == "L1" else _cfg(seed=0))
    tensors = []
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
        tensors.append(torch.from_numpy(np.stack(batch)).permute(0, 2, 1))
    return tensors


@torch.no_grad()
def evaluate(model, records, labels_idx, stage, batch_size=64):
    model.eval()
    cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                apply_noise=False, apply_quantization=False, seed=0)
           if stage == "L1" else _cfg(seed=0))
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


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    # 1. Train the E2 winner (V2 LeadMask)
    print("\n[1] Training V2 LeadMask (E2 winner) ...", flush=True)
    model = e2.train_model(tr, ytr, lead_mask_prob=0.5, epochs=20, tag="V2")

    results = {}
    for stage in ("L1", "L4"):
        # (a) no TTA
        lg, y = evaluate(model, te, yte, stage)
        m_no = macro_auroc(lg, y); pc_no = per_class_auroc(lg, y)
        # (b) BN adapt on the target distribution of this stage
        target = make_target_tensors(te, stage=stage)
        bn_adapt(model, target, n_passes=2)
        lg2, y2 = evaluate(model, te, yte, stage)
        m_ad = macro_auroc(lg2, y2); pc_ad = per_class_auroc(lg2, y2)
        results[stage] = {"no_tta": {"macro_auroc": m_no, "per_class": pc_no},
                          "bn_adapt": {"macro_auroc": m_ad, "per_class": pc_ad}}
        print(f"  {stage}: no-TTA={m_no:.4f}  BN-adapt={m_ad:.4f}  "
              f"Δ={m_ad-m_no:+.4f}", flush=True)

    metrics = {"fs": FS, "n_train": len(tr), "n_test": len(te),
              "classes": SUPERCLASSES, "base_model": "V2_LeadMask", "results": results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stages = list(results.keys())
    no = [results[s]["no_tta"]["macro_auroc"] for s in stages]
    ad = [results[s]["bn_adapt"]["macro_auroc"] for s in stages]
    x = np.arange(len(stages)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - w/2, no, w, label="no TTA", color="#4C72B0")
    b2 = ax.bar(x + w/2, ad, w, label="BN adapt (TTA)", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.4, 0.9)
    ax.set_title("E5: test-time BN adaptation on V2 LeadMask")
    ax.legend(fontsize=9)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "tta.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
