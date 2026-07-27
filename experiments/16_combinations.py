"""Experiment 16 — Method combinations on top of the lead-masking winner.

E2 showed lead-masking (K-MERL) is the decisive winner (0.717 on full watch).
E5 showed TTA adds +0.007. The synthesis report's "recommended recipe" stacks
lead-masking + matched-filter (A2) + watch-aug (A5) + TTA (H3). E16 tests
whether combinations add value over lead-masking alone, plus a new two-stage
combination (lead-mask pretrain → sim-watch fine-tune) and a lead-masking-prob
sweep.

All tested on L4 (full watch). Key question: does any combination beat
lead-masking alone (0.717)?

Combinations:
  C1 LeadMask (baseline, prob=0.5)
  C2 LeadMask + matched-filter (A2: shared bandpass on all leads)
  C3 LeadMask + TTA (BN-adapt)                       [reuse C1]
  C4 LeadMask + matched-filter + TTA                 [reuse C2]
  C5 Two-stage: LeadMask pretrain (10ep) → sim-watch fine-tune (10ep)
  C6 LeadMask prob=0.7 (sweep)
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
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e5", Path(__file__).resolve().parents[1] / "experiments" / "05_tta.py")
e5 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "16_combinations"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# matched-filter cfg: bandpass applied to ALL leads (lead_reduction=False),
# no electrode/noise/quant — just shared bandwidth (A2).
MF_CFG = _cfg(apply_lead_reduction=False, apply_electrode=False,
              apply_noise=False, apply_quantization=False, seed=None)
# full watch-sim on Lead-I (for two-stage fine-tune target)
WATCH_CFG = _cfg(seed=None)


class ComboDataset(Dataset):
    """Supports lead-masking + matched-filter (all-lead bandpass) combo."""

    def __init__(self, records, labels_idx,
                 lead_mask_prob=0.0, matched_filter=False, n_leads=12):
        self.records = records; self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob
        self.matched_filter = matched_filter
        self.n_leads = n_leads

    def __len__(self): return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate(
            [x, np.zeros((SIGLEN - T, x.shape[1]))], 0)

    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.matched_filter:
            # apply Apple bandpass to ALL leads (shared bandwidth, A2)
            out = simulate_watch(x, FS, MF_CFG, LEAD_NAMES, rng=rng)
            x = out["watch"].copy()  # (T,12)
        if self.lead_mask_prob > 0.0:
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False
            x = x.copy(); x[:, mask] = 0.0
        if self.n_leads == 1: x = x[:, 0:1]
        y = self.labels_idx[i]
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(y)


def train(records, labels_idx, lead_mask_prob=0.5, matched_filter=False,
          epochs=20, lr=1e-3, batch_size=64, tag="c"):
    ds = ComboDataset(records, labels_idx, lead_mask_prob, matched_filter)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


def two_stage(records, labels_idx, pre_epochs=10, ft_epochs=10):
    """Stage 1: lead-masking pretrain. Stage 2: fine-tune on sim-watch Lead-I
    (domain-targeted) while keeping lead-masking."""
    # stage 1
    m = train(records, labels_idx, lead_mask_prob=0.5, epochs=pre_epochs, tag="ts-pre")
    # stage 2: fine-tune with watch-sim on Lead-I + lead-masking (lower lr)
    ds = _TwoStageFTDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(ft_epochs):
        m.train(); tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(m(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [ts-ft] ep {ep+1}/{ft_epochs} loss={tot/nb:.4f}", flush=True)
    return m


class _TwoStageFTDataset(Dataset):
    """Fine-tune: watch-sim on Lead-I (domain-targeted) + lead-masking."""
    def __init__(self, records, labels_idx):
        self.records = records; self.labels_idx = labels_idx
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate(
            [x, np.zeros((SIGLEN - T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, WATCH_CFG, LEAD_NAMES, rng=rng)
        x = x.copy(); x[:, 0] = out["watch"]
        mask = rng.random(N_LEADS) < 0.5; mask[0] = False
        x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.labels_idx[i])


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    results = {}

    # C1: LeadMask baseline
    print("\n=== C1: LeadMask (baseline) ===", flush=True)
    c1 = train(tr, ytr, lead_mask_prob=0.5, epochs=20, tag="C1")
    lg, _ = e5.evaluate(c1, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C1_LeadMask"] = {"macro_auroc": m}
    print(f"  C1 L4: {m:.4f}", flush=True)

    # C2: LeadMask + matched-filter
    print("\n=== C2: LeadMask + matched-filter (A2) ===", flush=True)
    c2 = train(tr, ytr, lead_mask_prob=0.5, matched_filter=True, epochs=20, tag="C2")
    lg, _ = e5.evaluate(c2, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C2_LeadMask+MF"] = {"macro_auroc": m}
    print(f"  C2 L4: {m:.4f}", flush=True)

    # C3: C1 + TTA
    print("\n=== C3: LeadMask + TTA (BN-adapt) ===", flush=True)
    tgt = e5.make_target_tensors(te, stage="L4")
    e5.bn_adapt(c1, tgt, n_passes=2)
    lg, _ = e5.evaluate(c1, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C3_LeadMask+TTA"] = {"macro_auroc": m}
    print(f"  C3 L4: {m:.4f}", flush=True)

    # C4: C2 + TTA (full cheap stack: lead-mask + MF + TTA)
    print("\n=== C4: LeadMask + MF + TTA ===", flush=True)
    e5.bn_adapt(c2, tgt, n_passes=2)
    lg, _ = e5.evaluate(c2, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C4_LeadMask+MF+TTA"] = {"macro_auroc": m}
    print(f"  C4 L4: {m:.4f}", flush=True)

    # C5: two-stage
    print("\n=== C5: two-stage (lead-mask pretrain → sim-watch finetune) ===", flush=True)
    c5 = two_stage(tr, ytr, pre_epochs=10, ft_epochs=10)
    lg, _ = e5.evaluate(c5, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C5_two_stage"] = {"macro_auroc": m}
    print(f"  C5 L4: {m:.4f}", flush=True)

    # C6: LeadMask prob=0.7
    print("\n=== C6: LeadMask prob=0.7 ===", flush=True)
    c6 = train(tr, ytr, lead_mask_prob=0.7, epochs=20, tag="C6")
    lg, _ = e5.evaluate(c6, te, yte, "L4")
    m = e5.macro_auroc(lg, yte); results["C6_LeadMask_p07"] = {"macro_auroc": m}
    print(f"  C6 L4: {m:.4f}", flush=True)

    metrics = {"fs": FS, "n_train": len(tr), "n_test": len(te),
              "classes": SUPERCLASSES, "combinations": results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    vals = [results[n]["macro_auroc"] for n in names]
    colors = ["#4C72B0","#4C72B0","#55A868","#55A868","#9378B6","#4C72B0"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(names))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
    ax.axhline(0.717, color="red", ls="--", lw=1, label="E2 LeadMask ref 0.717")
    ax.axhline(0.865, color="green", ls="--", lw=1, label="L0 ceiling 0.865")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=8)
    ax.set_ylabel("Macro AUROC (L4 full watch)"); ax.set_ylim(0.5, 0.9)
    ax.set_title("E16: method combinations on top of lead-masking")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig(RESULTS / "combinations.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
