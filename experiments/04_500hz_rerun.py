"""Experiment 4 — 500 Hz rerun to surface the bandwidth axis.

E1 ran at 100 Hz where the bandwidth axis was muted (Nyquist 50 Hz ≈ the 40 Hz
Apple lowpass). E4 reruns the staircase at PTB-XL's native 500 Hz, where the
Apple bandpass (0.3–40 Hz) actually removes real clinical content (the 40–250 Hz
band) — so the L2 (bandwidth) step should show a measurable drop, and matched-
filter (A2) / watch-aug (A5) should become more impactful.

Also confirms the winner (lead-masking, E2) still holds at 500 Hz, and checks
whether the per-class spatial-vs-rhythm breakdown sharpens with more bandwidth.

Setup: PTB-XL 500 Hz subset (filename_hr), FS=500, SIGLEN=5000 (10 s),
simulator fs_watch=500 (no resample; isolates the bandpass effect). Reuses the
E1 staircase (L0 clinical → L1 Lead-I → L2 +bandwidth → L3 +electrode → L4 full)
+ the E2 lead-masking winner, all at 500 Hz.
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
from src.dataset import (SUPERCLASSES, LEAD_NAMES, load_ptbxl_meta,
                         build_superclass_map, build_split)
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "04_500hz_rerun"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 500.0
SIGLEN = 5000
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)  # no resample — isolate bandpass effect
    return WatchSimConfig(FS, **kw)


STAGE_CFGS = {
    "L0_clinical":  _cfg(apply_lead_reduction=False, apply_bandwidth=False,
                         apply_electrode=False, apply_noise=False,
                         apply_quantization=False, seed=0),
    "L1_leadI":     _cfg(apply_bandwidth=False, apply_electrode=False,
                         apply_noise=False, apply_quantization=False, seed=0),
    "L2_bandwidth": _cfg(apply_electrode=False, apply_noise=False,
                         apply_quantization=False, seed=0),
    "L3_electrode": _cfg(apply_noise=False, apply_quantization=False, seed=0),
    "L4_fullwatch": _cfg(seed=0),
}


class ECGDataset(Dataset):
    def __init__(self, records, labels_idx, lead_mask_prob=0.0,
                 fs=FS, siglen=SIGLEN):
        self.records = records; self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob
        self.fs = fs; self.siglen = siglen

    def __len__(self): return len(self.records)

    def _fixlen(self, x):
        T = x.shape[0]
        return x[:self.siglen] if T >= self.siglen else np.concatenate(
            [x, np.zeros((self.siglen - T, x.shape[1]))], 0)

    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        if self.lead_mask_prob > 0.0:
            rng = np.random.default_rng(None)
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False
            x = x.copy(); x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.labels_idx[i])


def train_model(records, labels_idx, lead_mask_prob=0.0,
                epochs=15, lr=1e-3, batch_size=32, tag="m"):
    ds = ECGDataset(records, labels_idx, lead_mask_prob)
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


@torch.no_grad()
def evaluate(model, records, labels_idx, cfg, batch_size=32):
    model.eval()
    is_clinical = (not cfg.apply_lead_reduction and not cfg.apply_bandwidth
                   and not cfg.apply_electrode and not cfg.apply_noise
                   and not cfg.apply_quantization)
    all_logits = []; all_y = []
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
                out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
                watch = out["watch"]
                framed = np.zeros_like(x)
                framed[:, 0] = watch
                xb = framed
                mu = xb.mean(0, keepdims=True); sd = xb.std(0, keepdims=True) + 1e-6
                xb = (xb - mu) / sd
            batch.append(xb.astype(np.float32))
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
    root = Path.home() / "data" / "ptbxl"
    print("Loading PTB-XL 500 Hz (filename_hr, max_per_class=300) ...", flush=True)
    df = load_ptbxl_meta(root / "ptbxl_database.csv")
    smap = build_superclass_map(root / "scp_statements.csv")
    splits = build_split(df, data_root=root, smap=smap, fs_col="filename_hr",
                         max_per_class=300)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)
    if len(tr) == 0:
        print("  ERROR: no 500 Hz records loaded. Is records500/ downloaded?", flush=True)
        return

    results = {}

    # 1. Clinical baseline + staircase
    print("\n[1] Training 12-lead clinical baseline (500 Hz, 15 ep) ...", flush=True)
    base = train_model(tr, ytr, lead_mask_prob=0.0, epochs=15, tag="clinical500")
    print("\n[2] Staircase ...", flush=True)
    staircase = {}
    for name, cfg in STAGE_CFGS.items():
        lg, _ = evaluate(base, te, yte, cfg)
        m = macro_auroc(lg, yte); pc = per_class_auroc(lg, yte)
        staircase[name] = {"macro_auroc": m, "per_class": pc}
        print(f"  {name:16s} macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)
    results["staircase"] = staircase

    # 2. Lead-masking winner at 500 Hz
    print("\n[3] Lead-masking winner (500 Hz, 15 ep) ...", flush=True)
    lm = train_model(tr, ytr, lead_mask_prob=0.5, epochs=15, tag="leadmask500")
    cfg_L4 = STAGE_CFGS["L4_fullwatch"]
    lg, _ = evaluate(lm, te, yte, cfg_L4)
    m = macro_auroc(lg, yte); pc = per_class_auroc(lg, yte)
    results["lead_masking_L4"] = {"macro_auroc": m, "per_class": pc}
    print(f"  LeadMask @ L4: macro={m:.4f} per={[round(a,3) for a in pc]}", flush=True)

    metrics = {"fs": FS, "siglen": SIGLEN, "n_train": len(tr), "n_test": len(te),
              "classes": SUPERCLASSES, **results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(staircase, results["lead_masking_L4"])
    print("\nDONE.", flush=True)


def _plot(staircase, lm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(staircase.keys())
    macro = [staircase[n]["macro_auroc"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(names)), macro, color="#4C72B0")
    ax.axhline(lm["macro_auroc"], color="red", ls="--", lw=1.3,
               label=f"lead-masking @ L4 = {lm['macro_auroc']:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Macro AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("E4: 500 Hz staircase — bandwidth axis now measurable")
    for i, v in enumerate(macro):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.savefig(RESULTS / "staircase_500hz.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
