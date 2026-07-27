"""Experiment 32 — TENT: test-time entropy minimization (the zero-shot lever).

USER GOAL: train on abundant clinical, deploy on watch with NO/MINIMAL tuning,
reach the watch-oracle (~0.93). Augmentation alone plateaus ~0.80 (E27b); few-
shot needs labels (E30); unsupervised feature alignment failed (E29 — but AdaBN
only RECOMPUTED BN stats, a weak mechanism). TENT (Wang et al. ICLR 2021) is the
strongest zero-label test-time method: it adapts ONLY the BN affine params
(gamma/beta) by minimizing prediction ENTROPY on the unlabeled target batch —
gradient-based, no labels, no source data needed at deployment.

This is the most on-target method for "minimal tuning": the model is trained on
clinical (+augmentation), then at test time it self-adjusts to the watch batch
using only the unlabeled watch signals.

Arms (5 seeds, tested on real CinC AF/NORM):
  clean                          clean Lead-I, no adaptation
  augment                        E27b recipe (s1.5, 5x), no adaptation (~0.80)
  augment + TENT                 <-- the test: entropy-min BN-affine on unlabeled target
  augment + TENT (full)          adapt ALL params via entropy (more aggressive)
  oracle                         real->real ceiling (~0.93)

TENT uses cinc_test SIGNALS ONLY at test time (transductive, no labels) — this
is legitimate: it's exactly the deployment setting (adapt to the batch you're
scoring). Reported honestly as transductive test-time adaptation.

Honesty: 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c AW proxy. TENT is
transductive (adapts on the test batch); that's the intended use but must be
labeled as such.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "32_tent"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"


def _to_tensor(sigs):
    arr = []
    for x in sigs:
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.shape[0])])
        x = x[:SIGLEN]; x = (x - x.mean()) / (x.std() + 1e-6)
        arr.append(x[None, :].astype(np.float32))
    return torch.from_numpy(np.stack(arr))


def entropy(logits):
    p = torch.softmax(logits, 1)
    return -(p * torch.log(p + 1e-9)).sum(1).mean()


def configure_tent(model, mode="affine"):
    """Set up which params TENT adapts. affine = BN gamma/beta only; full = all."""
    model.train()
    if mode == "affine":
        params = []
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.requires_grad_(True)
                # use batch stats at test time (TENT default)
                m.track_running_stats = False; m.running_mean = None; m.running_var = None
                params += [m.weight, m.bias]
            else:
                for p in m.parameters(recurse=False):
                    p.requires_grad_(False)
        return params
    else:  # full
        for p in model.parameters(): p.requires_grad_(True)
        return list(model.parameters())


@torch.enable_grad()
def tent_adapt(model, tgt_sigs, params, steps=10, lr=1e-3, bs=64):
    opt = torch.optim.Adam(params, lr=lr)
    X = _to_tensor(tgt_sigs)
    for step in range(steps):
        perm = torch.randperm(X.size(0))
        tot = 0.0; nb = 0
        for i in range(0, X.size(0), bs):
            xb = X[perm[i:i + bs]]
            opt.zero_grad()
            loss = entropy(model(xb))
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"    [tent] step {step+1}/{steps} entropy={tot/nb:.4f}", flush=True)
    return model


def run_seed(seed, tr_leadI, tr_y, cinc):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    out = {}

    # clean baseline
    m0 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(m0, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"] = e25.evaluate(m0, test_sigs, test_y)[0]

    # augment recipe (s1.5, 5x)
    aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.5, seed=seed)
    src = list(tr_leadI); sy = list(tr_y)
    for _ in range(5):
        src += [aug.generate(s) for s in tr_leadI]; sy += list(tr_y)
    mA = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mA, e25.SigDataset(src, sy), epochs=20, tag=f"s{seed}-aug")
    out["augment"] = e25.evaluate(mA, test_sigs, test_y)[0]

    # augment + TENT (affine only) — adapt on unlabeled test batch
    mT = copy.deepcopy(mA)
    params = configure_tent(mT, mode="affine")
    tent_adapt(mT, test_sigs, params, steps=10, lr=1e-3)
    mT.eval()
    out["augment_tent_affine"] = e25.evaluate(mT, test_sigs, test_y)[0]

    # augment + TENT (full params) — more aggressive
    mTf = copy.deepcopy(mA)
    paramsf = configure_tent(mTf, mode="full")
    tent_adapt(mTf, test_sigs, paramsf, steps=10, lr=1e-4)  # lower lr for full
    mTf.eval()
    out["augment_tent_full"] = e25.evaluate(mTf, test_sigs, test_y)[0]

    # oracle
    mO = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mO, e25.SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(mO, test_sigs, test_y)[0]

    for k, v in out.items(): print(f"  seed {seed} {k}: {v:.4f}", flush=True)
    return out


def main():
    print("Loading PTB-XL binary AF/NORM ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]; tr_leadI = [r["leadI"] for r in tr]; tr_y = [r["y"] for r in tr]
    print(f"  train={len(tr)}", flush=True)
    print("Loading REAL CinC ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC={len(cinc)}", flush=True)

    seeds = [0, 1, 2, 3, 4]
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc)

    keys = ["clean", "augment", "augment_tent_affine", "augment_tent_full", "oracle"]
    agg = {}
    augv = np.array([per_seed[s]["augment"] for s in seeds])
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
        if k.startswith("augment_tent"):
            agg[k]["delta_vs_augment_mean"] = float((vals - augv).mean())
            agg[k]["delta_positive_in_seeds"] = int(((vals - augv) > 0).sum())

    print("\n===== AGGREGATE (mean±std, 5 seeds) =====", flush=True)
    for k in keys:
        extra = ""
        if "delta_vs_augment_mean" in agg[k]:
            extra = f"  Δvs_aug={agg[k]['delta_vs_augment_mean']:+.4f} ({agg[k]['delta_positive_in_seeds']}/5)"
        print(f"  {k:20s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds,
        "aggregate": agg, "per_seed": per_seed,
        "question": "Does TENT (test-time entropy min, no labels) close the augment->oracle (0.80->0.93) gap?",
        "honesty": ["5 seeds", "n=700 CinC", "AF/NORM only", "CinC = E6c AW proxy",
                    "TENT is TRANSDUCTIVE (adapts on the unlabeled test batch) — intended deployment mode, but flagged"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["clean", "augment\n(s1.5,5x)", "aug+TENT\naffine", "aug+TENT\nfull", "oracle\nreal"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#937860"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(means[1], color="#55A868", ls="--", lw=1, label=f"augment bar ({means[1]:.3f})")
    ax.axhline(means[-1], color="#937860", ls=":", lw=1, label=f"oracle ({means[-1]:.3f})")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC on real CinC (5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E32: does TENT (zero-label test-time adapt) close the gap?")
    ax.legend(fontsize=8, loc="upper left")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "tent.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
