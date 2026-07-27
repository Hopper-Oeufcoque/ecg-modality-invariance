"""Experiment 17 — Lead-masking prob sweep + single-lead-with-sim-aug.

E16's prob sweep only went up (0.5→0.7, worse). E17 sweeps downward too to find
the optimum, and tests the backlog question: does a single-lead model trained
on *simulated watch* Lead-I (no 12-lead training at all) match the 12-lead
lead-masking winner? If yes, the simulator alone suffices; if no, the 12-lead
prior genuinely matters.

A. Lead-masking prob sweep: [0.2, 0.3, 0.4, 0.5, 0.6] on L4 full watch.
B. Single-lead model trained on sim-watch Lead-I (domain-targeted) vs 12-lead
   lead-masking (E2 V2). Tests whether the simulator can replace 12-lead data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e2", Path(__file__).resolve().parents[1] / "experiments" / "02_lead_masking.py")
e2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e2)
_spec5 = importlib.util.spec_from_file_location(
    "e5", Path(__file__).resolve().parents[1] / "experiments" / "05_tta.py")
e5 = importlib.util.module_from_spec(_spec5); _spec5.loader.exec_module(e5)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "17_prob_sweep_simlead"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)

WATCH_CFG = _cfg(seed=None)


class SimSingleLeadDataset(torch.utils.data.Dataset):
    """Single-lead model trained on simulated-watch Lead-I (domain-targeted)."""
    def __init__(self, records, labels_idx):
        self.records = records; self.labels_idx = labels_idx
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:1000] if T >= 1000 else np.concatenate([x, np.zeros((1000-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, WATCH_CFG, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        m1 = leadI.mean(); s1 = leadI.std() + 1e-6
        leadI = (leadI - m1) / s1
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.labels_idx[i]))


def train_singlelead_sim(records, labels_idx, epochs=20, tag="sl-sim"):
    ds = SimSingleLeadDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    model = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def eval_singlelead(model, records, labels_idx, stage, batch_size=64):
    model.eval()
    cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                apply_noise=False, apply_quantization=False, seed=0)
           if stage == "L1" else _cfg(seed=0))
    all_l = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:1000]
            if x.shape[0] < 1000:
                x = np.concatenate([x, np.zeros((1000-x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]
            m1 = leadI.mean(); s1 = leadI.std() + 1e-6
            leadI = (leadI - m1) / s1
            batch.append(leadI[:, None].astype(np.float32))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        all_l.append(model(xb).numpy()); all_y.append(np.array(labels_idx[i:i+batch_size]))
    return np.concatenate(all_l), np.concatenate(all_y)


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r["label"]] for r in rs])
    tr, te = splits["train"], splits["test"]
    ytr, yte = lab(tr), lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    results = {}

    # A. prob sweep
    print("\n=== A: lead-masking prob sweep ===", flush=True)
    for p in [0.2, 0.3, 0.4, 0.5, 0.6]:
        m = e2.train_model(tr, ytr, lead_mask_prob=p, epochs=20, tag=f"p{p}")
        lg, _ = e5.evaluate(m, te, yte, "L4")
        a = e5.macro_auroc(lg, yte)
        results[f"prob_{p}"] = {"macro_auroc_L4": a}
        print(f"  prob={p}: L4={a:.4f}", flush=True)

    # B. single-lead model trained on sim-watch
    print("\n=== B: single-lead model trained on sim-watch Lead-I ===", flush=True)
    sl = train_singlelead_sim(tr, ytr, epochs=20, tag="sl-sim")
    for st in ("L1", "L4"):
        lg, _ = eval_singlelead(sl, te, yte, st)
        a = e5.macro_auroc(lg, yte)
        results[f"singlelead_sim_{st}"] = {"macro_auroc": a}
        print(f"  singlelead-sim @ {st}: {a:.4f}", flush=True)

    metrics = {"n_train": len(tr), "n_test": len(te), "classes": SUPERCLASSES, **results}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    probs = [0.2, 0.3, 0.4, 0.5, 0.6]
    vals = [results[f"prob_{p}"]["macro_auroc_L4"] for p in probs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(probs, vals, "o-", color="#4C72B0", lw=2, markersize=10)
    ax.axhline(0.717, color="red", ls="--", lw=1, label="E2 LeadMask ref 0.717")
    ax.axhline(results["singlelead_sim_L4"]["macro_auroc"], color="#9378B6", ls=":",
               lw=1.3, label=f"single-lead+sim @ L4 = {results['singlelead_sim_L4']['macro_auroc']:.3f}")
    for p, v in zip(probs, vals):
        ax.text(p, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("lead-masking dropout probability"); ax.set_ylabel("Macro AUROC (L4)")
    ax.set_title("E17: lead-masking prob sweep + single-lead+sim reference")
    ax.set_ylim(0.65, 0.78); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "prob_sweep.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
