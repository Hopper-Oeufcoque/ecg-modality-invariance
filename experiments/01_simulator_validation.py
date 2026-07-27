"""Experiment 1 — Forward-physics watch simulator validation.

Keystone experiment: prove the simulator (Method F10) creates a *realistic,
addressable* modality gap, decomposed by shift axis.

Protocol
--------
1. Train a 1D ResNet on 12-lead PTB-XL (5 superclasses) — the clinical model.
2. Evaluate on held-out test under a staircase of increasing watch fidelity:
     L0  clinical 12-lead                      (in-domain ceiling)
     L1  Lead-I only (lead-count axis)          zero-mask 11 leads
     L2  Lead-I + Apple bandpass                (+ bandwidth axis)
     L3  Lead-I + bandpass + electrode          (+ electrode axis)
     L4  Lead-I + full watch (bandpass+electrode+noise+quant)  (+ noise axis)
3. Per-class AUROC breakdown: spatial classes (MI/STTC/HYP/CD) should degrade
   more than NORM under lead reduction — the central sanity check.
4. Recovery ablation:
     R0  naive 12-lead model on full watch            (= L4)
     R1  + matched-filter preprocessing (A2)          train & test share bandpass
     R2  + watch-simulation augmentation (A5)         domain-randomized sim at train

Outputs: results/01_simulator_validation/{metrics.json, staircase.png,
         per_class.png, recovery.png, example_signals.png}

NOTE: Run at PTB-XL 100 Hz native geometry. The bandwidth axis is therefore
under-demonstrated (100 Hz Nyquist = 50 Hz already near the 40 Hz lowpass);
lead-count + noise axes — the dominant ones — are fully exercised. A 500 Hz
rerun would sharpen the bandwidth step.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import (WatchSimConfig, simulate_watch)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "01_simulator_validation"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000           # 10 s @ 100 Hz
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class ECGDataset(Dataset):
    def __init__(self, records, labels_idx, fs=FS, siglen=SIGLEN,
                 sim_cfg=None, augment_sim=False, lead_names=LEAD_NAMES):
        self.records = records
        self.labels_idx = labels_idx
        self.fs = fs
        self.siglen = siglen
        self.sim_cfg = sim_cfg
        self.augment_sim = augment_sim
        self.lead_names = lead_names

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
        x = rec["ecg"]  # (T,12) mV
        x = self._fixlen(x)
        # per-record normalization (A3)
        mu = x.mean(0, keepdims=True)
        sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        if self.augment_sim and self.sim_cfg is not None:
            # Domain-randomized watch augmentation (A5). Two modes:
            #  - lead_reduction=True  -> simulate Lead-I watch, embed in ch0,
            #    optionally drop extra leads (teaches lead-robustness).
            #  - lead_reduction=False -> apply only the shared stages (e.g.
            #    bandpass) to ALL leads = matched-filter preprocessing (A2).
            rng = np.random.default_rng(None)
            out = simulate_watch(x, self.fs, self.sim_cfg, self.lead_names, rng=rng)
            watch = out["watch"]
            x = x.copy()
            if watch.ndim == 1:
                x[:, 0] = watch
                if rng.random() < 0.5:
                    mask = rng.random(N_LEADS) < 0.3
                    mask[0] = False
                    x[:, mask] = 0.0
            else:
                x = watch  # all leads processed (matched-filter path)
        y = self.labels_idx[i]
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(y)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def make_staircase_input(x_12lead, cfg: WatchSimConfig):
    """Apply watch sim to Lead-I and embed back into a 12-lead frame
    (Lead-I in ch0, others zeroed) so the 12-lead model can read it."""
    out = simulate_watch(x_12lead, FS, cfg, LEAD_NAMES,
                         rng=np.random.default_rng(cfg.seed if cfg.seed is not None else 0))
    watch = out["watch"]  # (T,)
    framed = np.zeros_like(x_12lead)
    framed[:, 0] = watch
    return framed


def _cfg(**kw):
    """WatchSimConfig pinned to 100 Hz native geometry (no rate change) so the
    model frame stays 1000 samples and we isolate distortion axes, not rate."""
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


STAGE_CFGS = {
    "L0_clinical":   _cfg(apply_lead_reduction=False, apply_bandwidth=False,
                          apply_electrode=False, apply_noise=False,
                          apply_quantization=False, seed=0),
    "L1_leadI":      _cfg(apply_bandwidth=False, apply_electrode=False,
                          apply_noise=False, apply_quantization=False, seed=0),
    "L2_bandwidth":  _cfg(apply_electrode=False, apply_noise=False,
                          apply_quantization=False, seed=0),
    "L3_electrode":  _cfg(apply_noise=False, apply_quantization=False, seed=0),
    "L4_fullwatch":  _cfg(seed=0),
}


@torch.no_grad()
def evaluate(model, records, labels_idx, stage_cfg, batch_size=64):
    """Evaluate model on a staircase stage. For L0, feed full 12-lead.
    For L1-L4, feed Lead-I (possibly distorted) embedded in 12-lead frame."""
    model.eval()
    all_logits = []
    all_y = []
    is_clinical = (stage_cfg is None) or (not stage_cfg.apply_lead_reduction
                                          and not stage_cfg.apply_bandwidth
                                          and not stage_cfg.apply_electrode
                                          and not stage_cfg.apply_noise
                                          and not stage_cfg.apply_quantization)
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            if is_clinical:
                xb = x
            else:
                xb = make_staircase_input(x, stage_cfg)
                # renormalize the framed signal
                mu = xb.mean(0, keepdims=True); sd = xb.std(0, keepdims=True) + 1e-6
                xb = (xb - mu) / sd
            batch.append(xb.astype(np.float32))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        logits = model(xb)
        all_logits.append(logits.numpy())
        all_y.append(np.array(labels_idx[i:i+batch_size]))
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    return logits, y


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    """One-vs-rest AUROC per class (macro handling missing classes)."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(n_classes):
        mask = (y == c) | (np.zeros_like(y) == 0)  # one-vs-rest over all
        yb = (y == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb):
            aucs.append(float("nan")); continue
        try:
            aucs.append(float(roc_auc_score(yb, logits[:, c])))
        except Exception:
            aucs.append(float("nan"))
    return aucs


def macro_auroc(logits, y):
    aucs = per_class_auroc(logits, y)
    valid = [a for a in aucs if not np.isnan(a)]
    return float(np.mean(valid)) if valid else float("nan")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(records, labels_idx, sim_cfg=None, augment_sim=False,
                epochs=15, lr=1e-3, batch_size=64, tag="model"):
    ds = ECGDataset(records, labels_idx, sim_cfg=sim_cfg, augment_sim=augment_sim)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        tot = 0.0; nb = 0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [{tag}] epoch {ep+1}/{epochs}  loss={tot/nb:.4f}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading PTB-XL ...", flush=True)
    splits = load_all(max_per_class=1500)
    cls_to_idx = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(recs): return np.array([cls_to_idx[r["label"]] for r in recs])
    tr, va, te = splits["train"], splits["val"], splits["test"]
    ytr, yva, yte = lab(tr), lab(va), lab(te)
    print(f"  train={len(tr)} val={len(va)} test={len(te)}", flush=True)
    print(f"  test class counts: {[(SUPERCLASSES[c], int((yte==c).sum())) for c in range(N_CLASSES)]}", flush=True)

    # ---- Train clinical baseline (12-lead) ----
    print("\n[1] Training 12-lead clinical baseline ...", flush=True)
    base_model = train_model(tr, ytr, epochs=15, tag="clinical")

    # ---- Staircase evaluation ----
    print("\n[2] Staircase evaluation (modality gap by axis) ...", flush=True)
    staircase = {}
    for name, cfg in STAGE_CFGS.items():
        logits, _ = evaluate(base_model, te, yte, cfg)
        macro = macro_auroc(logits, yte)
        pc = per_class_auroc(logits, yte)
        staircase[name] = {"macro_auroc": macro, "per_class": pc}
        print(f"  {name:16s} macro AUROC = {macro:.4f}  "
              f"per-class={[round(a,3) for a in pc]}", flush=True)

    # ---- Recovery ablation ----
    print("\n[3] Recovery ablation ...", flush=True)
    recovery = {}

    # R1: matched-filter preprocessing — train on data that already went through
    # Apple bandpass (so train & test share bandwidth). Keep 12-lead.
    mf_cfg = _cfg(apply_lead_reduction=False, apply_electrode=False,
                  apply_noise=False, apply_quantization=False, seed=None)
    # We simulate matched-filter by training with the bandwidth stage applied to all leads.
    # Reuse augment path but force only bandwidth. Simpler: pre-filter training data.
    print("  R1: training matched-filter model (shared bandpass) ...", flush=True)
    r1_model = train_model(tr, ytr, epochs=15, tag="R1-mf",
                           sim_cfg=mf_cfg, augment_sim=True)
    logits, _ = evaluate(r1_model, te, yte, STAGE_CFGS["L4_fullwatch"])
    recovery["R1_matched_filter"] = {"macro_auroc": macro_auroc(logits, yte),
                                     "per_class": per_class_auroc(logits, yte)}

    # R2: full watch-sim augmentation (domain randomization) at train time
    print("  R2: training with watch-sim augmentation (A5) ...", flush=True)
    aug_cfg = _cfg(seed=None)  # randomized per sample
    r2_model = train_model(tr, ytr, epochs=15, tag="R2-aug",
                           sim_cfg=aug_cfg, augment_sim=True)
    logits, _ = evaluate(r2_model, te, yte, STAGE_CFGS["L4_fullwatch"])
    recovery["R2_watch_aug"] = {"macro_auroc": macro_auroc(logits, yte),
                                "per_class": per_class_auroc(logits, yte)}
    for k, v in recovery.items():
        print(f"  {k:20s} macro AUROC = {v['macro_auroc']:.4f}", flush=True)

    # ---- Save metrics ----
    metrics = {
        "fs": FS, "siglen": SIGLEN, "n_train": len(tr), "n_test": len(te),
        "classes": SUPERCLASSES, "staircase": staircase, "recovery": recovery,
    }
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics -> {RESULTS/'metrics.json'}", flush=True)

    # ---- Figures ----
    try:
        _plot(staircase, recovery)
    except Exception as e:
        print(f"  (plot skipped: {e})", flush=True)

    # ---- Example signal figure ----
    try:
        _plot_examples(te)
    except Exception as e:
        print(f"  (example plot skipped: {e})", flush=True)

    print("\nDONE.", flush=True)


def _plot(staircase, recovery):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(staircase.keys())
    macro = [staircase[n]["macro_auroc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(names)), macro, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Macro AUROC")
    ax.set_title("Modality gap by axis (clinical model on simulated watch)")
    ax.set_ylim(0.4, 1.0)
    for i, v in enumerate(macro):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "staircase.png", dpi=130); plt.close()

    # per-class heatmap
    pc = np.array([staircase[n]["per_class"] for n in names])
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pc, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(SUPERCLASSES))); ax.set_xticklabels(SUPERCLASSES)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
    for i in range(pc.shape[0]):
        for j in range(pc.shape[1]):
            ax.text(j, i, f"{pc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Per-class AUROC across modality staircase")
    fig.colorbar(im); plt.tight_layout()
    plt.savefig(RESULTS / "per_class.png", dpi=130); plt.close()

    # recovery
    rnames = ["L4_fullwatch"] + list(recovery.keys())
    rvals = [staircase["L4_fullwatch"]["macro_auroc"]] + [recovery[k]["macro_auroc"] for k in recovery]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(rnames)), rvals, color=["#C44E52", "#DD8452", "#55A868"])
    ax.set_xticks(range(len(rnames)))
    ax.set_xticklabels([n.replace("_", "\n") for n in rnames], fontsize=9)
    ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Recovery: closing the watch modality gap")
    for i, v in enumerate(rvals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "recovery.png", dpi=130); plt.close()


def _plot_examples(test_records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = test_records[0]
    x = rec["ecg"][:SIGLEN]
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    cfgs = [("L0 Lead-I (clean)", STAGE_CFGS["L1_leadI"]),
            ("L2 + bandwidth", STAGE_CFGS["L2_bandwidth"]),
            ("L3 + electrode", STAGE_CFGS["L3_electrode"]),
            ("L4 full watch", STAGE_CFGS["L4_fullwatch"])]
    axes[0].plot(x[:, 0], color="#4C72B0"); axes[0].set_ylabel("12-lead Lead I", fontsize=8)
    axes[0].set_title("Forward-physics simulator stages (one PTB-XL record)")
    for ax, (name, cfg) in zip(axes[1:], cfgs):
        out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
        ax.plot(out["watch"], color="#C44E52", alpha=0.85)
        ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel("samples (100 Hz)")
    plt.tight_layout(); plt.savefig(RESULTS / "example_signals.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
