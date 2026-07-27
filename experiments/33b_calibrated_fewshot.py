"""Experiment 33b — Calibrated DR + few-shot: the best-of-both recipe.

E33 (calibrated domain randomization) is the new zero-shot best (0.824, closes
~57% of the gap with NO labels). E30 showed ~50 real labels closes most of the
rest when stacked on hand-tuned augmentation. E33b combines the two strongest
levers: calibrated-DR pretrain + fine-tune on k labeled real samples.

Question: does calibrated-DR pretrain (better zero-shot start) + few-shot reach
oracle (~0.93) with fewer labels than hand-tuned pretrain did? i.e. does a better
zero-shot base lower the labeled-data budget?

Arms (5 seeds, real CinC AF/NORM):
  calibrated (k=0)                     zero-shot base (E33 = 0.824)
  calibrated + finetune k=25/50/100
  hand-tuned + finetune k=50           (E30 reference point on this seed set)
  oracle

Honesty: 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c AW proxy. Calibration
uses unlabeled target stats (no labels); k samples are labeled real. Morphology
validated in E33 via QRS-band corr (0.966) + R-peak match (0.976).
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
_spec33 = importlib.util.spec_from_file_location(
    "e33", Path(__file__).resolve().parents[1] / "experiments" / "33_calibrated_dr.py")
e33 = importlib.util.module_from_spec(_spec33); _spec33.loader.exec_module(e33)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "33b_calibrated_fewshot"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"
KS = [25, 50, 100]


def finetune(model, sigs, ys, epochs=15, lr=5e-4, tag="ft"):
    if len(sigs) == 0: return model
    return e25.train(model, e25.SigDataset(sigs, ys), epochs=epochs, lr=lr, batch_size=min(32, len(sigs)), tag=tag)


def balanced_draw(ref, k, seed):
    import random
    rr = list(ref); random.Random(seed).shuffle(rr)
    pos = [r for r in rr if r["y"] == 1][:k // 2]; neg = [r for r in rr if r["y"] == 0][:k - k // 2]
    return [r["sig"] for r in pos + neg], [r["y"] for r in pos + neg]


def run_seed(seed, tr_leadI, tr_y, cinc, clin_stats):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]
    tgt_stats = e33.dist_stats([r["sig"] for r in cinc_ref])

    out = {}

    # calibrated-DR pretrain (E33 recipe)
    cal = e33.CalibratedAugmenter(clin_stats, tgt_stats, seed=seed, cover=1.3)
    src = list(tr_leadI); sy = list(tr_y)
    for _ in range(5): src += [cal.generate(s) for s in tr_leadI]; sy += list(tr_y)
    base = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(base, e25.SigDataset(src, sy), epochs=20, tag=f"s{seed}-calib")
    out["calibrated_k0"] = e25.evaluate(base, test_sigs, test_y)[0]

    for k in KS:
        ks_sigs, ks_y = balanced_draw(cinc_ref, k, seed * 100 + k)
        mF = copy.deepcopy(base)
        finetune(mF, ks_sigs, ks_y, tag=f"s{seed}-calib_ft{k}")
        out[f"calibrated_ft_k{k}"] = e25.evaluate(mF, test_sigs, test_y)[0]

    # hand-tuned + finetune k50 reference
    hand = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.5, seed=seed)
    hsrc = list(tr_leadI); hsy = list(tr_y)
    for _ in range(5): hsrc += [hand.generate(s) for s in tr_leadI]; hsy += list(tr_y)
    mh = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mh, e25.SigDataset(hsrc, hsy), epochs=20, tag=f"s{seed}-hand")
    ks_sigs, ks_y = balanced_draw(cinc_ref, 50, seed * 100 + 50)
    finetune(mh, ks_sigs, ks_y, tag=f"s{seed}-hand_ft50")
    out["handtuned_ft_k50"] = e25.evaluate(mh, test_sigs, test_y)[0]

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
    clin_stats = e33.dist_stats(tr_leadI)

    seeds = [0, 1, 2, 3, 4]
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc, clin_stats)

    keys = ["calibrated_k0"] + [f"calibrated_ft_k{k}" for k in KS] + ["handtuned_ft_k50", "oracle"]
    agg = {}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}

    print("\n===== AGGREGATE (mean±std, 5 seeds) =====", flush=True)
    orc = agg["oracle"]["mean"]
    for k in keys:
        pct = 100 * (agg[k]["mean"] - agg["calibrated_k0"]["mean"]) / (orc - agg["calibrated_k0"]["mean"]) if orc > agg["calibrated_k0"]["mean"] else 0
        print(f"  {k:20s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}  ({pct:+.0f}% of remaining gap)", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds, "ks": KS,
        "aggregate": agg, "per_seed": per_seed,
        "question": "does calibrated-DR pretrain + few-shot reach oracle with fewer labels than hand-tuned?",
        "honesty": ["5 seeds", "n=700 CinC", "AF/NORM only", "CinC = E6c AW proxy",
                    "calibration = unlabeled target stats; k = labeled real",
                    "morphology validated in E33 (QRS corr 0.966, R-peak 0.976)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["calib\nk=0", "calib\n+k25", "calib\n+k50", "calib\n+k100", "hand\n+k50", "oracle"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#DD8452", "#C44E52", "#8172B2", "#937860", "#55A868", "#333333"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(agg["oracle"]["mean"], color="#333333", ls=":", lw=1.5, label=f"oracle ({agg['oracle']['mean']:.3f})")
    ax.axhline(agg["calibrated_k0"]["mean"], color="#DD8452", ls="--", lw=1, label=f"calibrated zero-shot ({agg['calibrated_k0']['mean']:.3f})")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AUROC on real CinC (5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E33b: calibrated DR + few-shot — how close to oracle?")
    ax.legend(fontsize=8, loc="lower right")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "calibrated_fewshot.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
