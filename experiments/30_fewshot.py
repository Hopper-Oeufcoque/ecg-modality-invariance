"""Experiment 30 — Semi-supervised few-shot: how many REAL labels close the gap?

E29 showed unsupervised alignment can't close the 0.73->0.93 gap; E27 showed
augmentation recovers ~46% with zero labels. The honest remaining question for a
practitioner: how many LABELED real Apple Watch samples does it take to close
the rest? This directly quantifies the data-collection cost.

Method: pretrain on the augmentation recipe (clean Lead-I + stochastic aug,
strength 1.5), then FINE-TUNE on k labeled real target samples
(k in {0, 10, 25, 50, 100, 200}). Compare against:
  - augment-only (k=0)           the E27 zero-label recipe
  - from-scratch on k real only   (no clinical pretrain — is pretrain worth it?)
  - oracle (all ~700 real)        ceiling

The gap between "augment+finetune" and "scratch" at each k = the value of the
clinical-data + augmentation pretraining. The k where augment+finetune reaches
~oracle = the real-world labeled-data budget needed.

5 seeds. k labeled samples are drawn from cinc_ref (disjoint from cinc_test).

Honesty: 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c AW proxy. Few-shot k is
small so high variance expected at low k; report std.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "30_fewshot"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"
KS = [0, 10, 25, 50, 100, 200]


def finetune(model, sigs, ys, epochs=15, lr=5e-4, tag="ft"):
    if len(sigs) == 0:
        return model
    return e25.train(model, e25.SigDataset(sigs, ys), epochs=epochs, lr=lr, batch_size=min(32, len(sigs)), tag=tag)


def balanced_draw(ref, k, rng):
    """Draw k label-balanced samples from ref list of dicts."""
    if k == 0: return [], []
    pos = [r for r in ref if r["y"] == 1]; neg = [r for r in ref if r["y"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    kp = k // 2; kn = k - kp
    sel = pos[:kp] + neg[:kn]
    return [r["sig"] for r in sel], [r["y"] for r in sel]


def run_seed(seed, tr_leadI, tr_y, cinc):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    # pretrain once on the augmentation recipe (strength 1.5, 3x expansion ~ E27 best)
    aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.5, seed=seed)
    src = list(tr_leadI); sy = list(tr_y)
    for _ in range(3):
        src += [aug.generate(s) for s in tr_leadI]; sy += list(tr_y)
    base = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(base, e25.SigDataset(src, sy), epochs=20, tag=f"s{seed}-pretrain")

    py_rng = np.random.default_rng(1000 + seed)  # noqa: F841 (reserved)
    out = {"pretrain_augonly": e25.evaluate(base, test_sigs, test_y)[0]}

    for k in KS:
        # draw the SAME k for both arms for fairness
        rr = list(cinc_ref)
        import random; random.Random(seed * 100 + k).shuffle(rr)
        ks_sigs, ks_y = balanced_draw(rr, k, random.Random(seed * 100 + k))

        # arm A: augment-pretrain + finetune on k
        mA = copy.deepcopy(base)
        finetune(mA, ks_sigs, ks_y, tag=f"s{seed}-ftk{k}")
        out[f"finetune_k{k}"] = e25.evaluate(mA, test_sigs, test_y)[0]

        # arm B: from scratch on k real only (k=0 -> chance)
        if k == 0:
            out[f"scratch_k{k}"] = 0.5
        else:
            mB = ECGResNet1d(1, N_CLASSES).to(DEVICE)
            finetune(mB, ks_sigs, ks_y, epochs=30, lr=1e-3, tag=f"s{seed}-scratchk{k}")
            out[f"scratch_k{k}"] = e25.evaluate(mB, test_sigs, test_y)[0]

    # oracle: all ref
    mO = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mO, e25.SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), epochs=20, tag=f"s{seed}-oracle")
    out["oracle_all"] = e25.evaluate(mO, test_sigs, test_y)[0]

    for kk, vv in out.items():
        print(f"  seed {seed} {kk}: {vv:.4f}", flush=True)
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

    keys = ["pretrain_augonly"] + [f"finetune_k{k}" for k in KS] + [f"scratch_k{k}" for k in KS] + ["oracle_all"]
    agg = {}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}

    print("\n===== AGGREGATE (mean±std over 5 seeds) =====", flush=True)
    oracle = agg["oracle_all"]["mean"]
    print(f"  oracle(all real) = {oracle:.4f}", flush=True)
    print("  k | augment+finetune | scratch(real only) | %gap-closed(ft)", flush=True)
    base = agg["pretrain_augonly"]["mean"]
    for k in KS:
        ft = agg[f"finetune_k{k}"]["mean"]; sc = agg[f"scratch_k{k}"]["mean"]
        pct = 100 * (ft - base) / (oracle - base) if oracle > base else 0
        print(f"  {k:4d} | {ft:.4f}±{agg[f'finetune_k{k}']['std']:.3f} | {sc:.4f}±{agg[f'scratch_k{k}']['std']:.3f} | {pct:+.0f}%", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds, "ks": KS,
        "aggregate": agg, "per_seed": per_seed, "oracle": oracle, "augonly_base": base,
        "question": "how many labeled real samples (k) does augment-pretrain + finetune need to close the gap, and does pretrain beat from-scratch?",
        "honesty": ["5 seeds", "n=700 CinC", "AF/NORM only", "CinC = E6c AW proxy",
                    "low-k high variance expected"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, oracle, base)
    print("\nDONE.", flush=True)


def _plot(agg, oracle, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ft = [agg[f"finetune_k{k}"]["mean"] for k in KS]; fte = [agg[f"finetune_k{k}"]["std"] for k in KS]
    sc = [agg[f"scratch_k{k}"]["mean"] for k in KS]; sce = [agg[f"scratch_k{k}"]["std"] for k in KS]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(KS, ft, yerr=fte, marker="o", capsize=4, color="#55A868", label="augment-pretrain + finetune")
    ax.errorbar(KS, sc, yerr=sce, marker="s", capsize=4, color="#C44E52", label="from scratch (real only)")
    ax.axhline(oracle, color="#937860", ls=":", lw=1.5, label=f"oracle all real ({oracle:.3f})")
    ax.axhline(base, color="#4C72B0", ls="--", lw=1, label=f"augment-only k=0 ({base:.3f})")
    ax.set_xlabel("k = number of LABELED real target samples")
    ax.set_ylabel("AUROC on real CinC (mean±std, 5 seeds)")
    ax.set_title("E30: how many real labels to close the modality gap?")
    ax.legend(fontsize=8, loc="lower right"); ax.set_ylim(0.5, 1.0); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(RESULTS / "fewshot_curve.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
