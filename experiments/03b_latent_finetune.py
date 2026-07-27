"""Experiment 3b — Fair latent-alignment test: pretrain THEN end-to-end.

E3 compared latent alignment (frozen encoder + linear probe) against V2
lead-masking (end-to-end) — an unfair comparison flagged in E3's REPORT.
E3b fixes it: contrastive self-cutting pretrain (variant B, the better one)
THEN end-to-end fine-tune the WHOLE network WITH lead-masking.

Question: does contrastive pretraining add value on top of lead-masking when
both are trained end-to-end? If yes -> latent alignment is a useful
pretraining step. If no -> lead-masking alone is sufficient (the simpler win).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e3", Path(__file__).resolve().parents[1] / "experiments" / "03_latent_alignment.py")
e3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e3)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "03b_latent_finetune"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
FEAT_DIM = 64
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


def finetune(feat, records, labels_idx, lead_mask_prob=0.5,
             epochs=20, lr=1e-3, batch_size=64, tag="ft"):
    """End-to-end fine-tune: feat backbone + fresh head, WITH lead-masking."""
    # build a classifier on top of the featurizer's feature dim
    head = nn.Linear(FEAT_DIM, N_CLASSES).to(DEVICE)
    model = nn.Sequential(feat, head)
    # unfreeze feat
    for p in feat.parameters(): p.requires_grad = True
    # reuse e2's AugDataset for lead-masking training
    _s2 = importlib.util.spec_from_file_location(
        "e2", ROOT / "experiments" / "02_lead_masking.py")
    e2 = importlib.util.module_from_spec(_s2); _s2.loader.exec_module(e2)
    ds = e2.AugDataset(records, labels_idx, lead_mask_prob=lead_mask_prob, n_leads=12)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
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


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    # 1. Contrastive pretrain (variant B: watch-sim self-cutting)
    print("\n[1] Contrastive pretrain (watch-sim self-cutting, 20 ep) ...", flush=True)
    feat = e3.pretrain_with_watchsim(tr, epochs=20, tag="pre")

    # 2. End-to-end fine-tune with lead-masking
    print("\n[2] End-to-end fine-tune + lead-masking (20 ep) ...", flush=True)
    model = finetune(feat, tr, ytr, lead_mask_prob=0.5, epochs=20, tag="ft")

    # 3. Eval (reuse e3.evaluate but model is Sequential(feat,head))
    results = {}
    for stage in ("L1", "L4"):
        cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                    apply_noise=False, apply_quantization=False, seed=0)
               if stage == "L1" else _cfg(seed=0))
        all_logits = []; all_y = []
        with torch.no_grad():
            model.eval()
            for i in range(0, len(te), 64):
                batch = []
                for rec in te[i:i+64]:
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
                all_y.append(np.array(yte[i:i+64]))
        lg = np.concatenate(all_logits); y = np.concatenate(all_y)
        m = e3.macro_auroc(lg, y); pc = e3.per_class_auroc(lg, y)
        results[stage] = {"macro_auroc": m, "per_class": pc}
        print(f"  pretrain+ft @ {stage}: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)

    metrics = {"fs": FS, "n_train": len(tr), "n_test": len(te),
              "classes": SUPERCLASSES, "variant": "pretrain_watchsim + finetune_leadmask",
              "results": results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
