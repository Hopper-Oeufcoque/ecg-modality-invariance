"""Experiment 27b — Push augmentation to its plateau, WITH a morphology guard.

E27 found augmentation scales monotonically and the best config sat at the GRID
CORNER (strength 1.5, 3x) = 0.791 — i.e. we had not found the ceiling. E27b
extends UP: strength {1.5, 2.0, 2.5} x expansion {3, 5, 7}, 5 seeds, to locate
the true plateau.

CRITICAL GUARD: higher strength risks corrupting QRS morphology -> silently
invalidating the label (buying AUROC with broken data). So for every config we
measure the mean source-correlation (augmented vs source R-peak structure) and
FLAG any config whose morphology preservation drops below 0.85. A config only
"counts" as a real win if it's both high-AUROC AND morphology-safe.

Baselines: clean-only + the E26/E27 reference (s1.0_x1, s1.5_x3).

Honesty: 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c AW proxy. Morphology
guard is a correlation proxy for label validity, not cardiologist-verified.
"""

from __future__ import annotations

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
RESULTS = ROOT / "results" / "27b_augment_plateau"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"
STRENGTHS = [1.5, 2.0, 2.5]
EXPANSIONS = [3, 5, 7]
MORPH_GUARD = 0.85


def morph_corr(aug, sigs, n=100):
    """Mean correlation between augmented and source signals (label-validity proxy)."""
    cs = []
    for s in sigs[:n]:
        s = np.asarray(s, dtype=np.float64)
        a = aug.generate(s)
        m = min(len(s), len(a))
        c = np.corrcoef(s[:m], a[:m])[0, 1]
        if np.isfinite(c): cs.append(c)
    return float(np.mean(cs))


def run_seed(seed, tr_leadI, tr_y, cinc, morph_cache):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    out = {}
    def evaluate_config(sigs, ys, tag):
        m = ECGResNet1d(1, N_CLASSES).to(DEVICE)
        e25.train(m, e25.SigDataset(sigs, ys), epochs=20, tag=f"s{seed}-{tag}")
        auc = e25.evaluate(m, test_sigs, test_y)[0]
        out[tag] = auc
        print(f"  seed {seed} {tag}: {auc:.4f}", flush=True)

    evaluate_config(tr_leadI, tr_y, "clean_only")
    for st in STRENGTHS:
        aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=st, seed=seed)
        if st not in morph_cache:
            morph_cache[st] = morph_corr(aug, tr_leadI)
        for ex in EXPANSIONS:
            sigs = list(tr_leadI); ys = list(tr_y)
            for _ in range(ex):
                sigs += [aug.generate(s) for s in tr_leadI]; ys += list(tr_y)
            evaluate_config(sigs, ys, f"s{st}_x{ex}")
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
    morph_cache = {}
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc, morph_cache)

    keys = ["clean_only"] + [f"s{st}_x{ex}" for st in STRENGTHS for ex in EXPANSIONS]
    agg = {}
    clean = np.array([per_seed[s]["clean_only"] for s in seeds])
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
        if k != "clean_only":
            d = vals - clean
            agg[k]["delta_vs_clean_mean"] = float(d.mean())
            agg[k]["delta_positive_in_seeds"] = int((d > 0).sum())
            st = float(k.split("_")[0][1:])
            agg[k]["morph_corr"] = morph_cache.get(st)
            agg[k]["morph_safe"] = bool(morph_cache.get(st, 0) >= MORPH_GUARD)

    print("\n===== AGGREGATE (mean±std over 5 seeds) =====", flush=True)
    print("  morphology preservation (source corr) by strength:", flush=True)
    for st in STRENGTHS:
        safe = "SAFE" if morph_cache[st] >= MORPH_GUARD else "*** UNSAFE (label risk) ***"
        print(f"    strength {st}: corr={morph_cache[st]:.3f}  {safe}", flush=True)
    for k in keys:
        extra = ""
        if "delta_vs_clean_mean" in agg[k]:
            flag = "" if agg[k]["morph_safe"] else "  ⚠UNSAFE"
            extra = f"  Δ={agg[k]['delta_vs_clean_mean']:+.4f} ({agg[k]['delta_positive_in_seeds']}/5 +){flag}"
        print(f"  {k:12s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)

    safe_keys = [k for k in keys if k != "clean_only" and agg[k]["morph_safe"]]
    best_safe = max(safe_keys, key=lambda k: agg[k]["mean"]) if safe_keys else None
    best_any = max([k for k in keys if k != "clean_only"], key=lambda k: agg[k]["mean"])
    print(f"\n  BEST morphology-safe config: {best_safe} = {agg[best_safe]['mean']:.4f}" if best_safe else "\n  no morphology-safe config", flush=True)
    print(f"  BEST any config: {best_any} = {agg[best_any]['mean']:.4f} (morph_safe={agg[best_any]['morph_safe']})", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds,
        "strengths": STRENGTHS, "expansions": EXPANSIONS, "morph_guard": MORPH_GUARD,
        "morph_corr_by_strength": morph_cache,
        "aggregate": agg, "per_seed": per_seed,
        "best_morph_safe": best_safe, "best_any": best_any,
        "question": "where does augmentation plateau, and is the best config morphology-safe (label-valid)?",
        "honesty": ["5 seeds", "n=700 CinC", "AF/NORM only", "CinC = E6c AW proxy",
                    "morph guard is a correlation proxy for label validity, not clinician-verified"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, morph_cache)
    print("\nDONE.", flush=True)


def _plot(agg, morph_cache):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.zeros((len(STRENGTHS), len(EXPANSIONS)))
    for i, st in enumerate(STRENGTHS):
        for j, ex in enumerate(EXPANSIONS):
            grid[i, j] = agg[f"s{st}_x{ex}"]["mean"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 2]})
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(EXPANSIONS))); ax.set_xticklabels([f"{e}x" for e in EXPANSIONS])
    ax.set_yticks(range(len(STRENGTHS))); ax.set_yticklabels([f"{s}" for s in STRENGTHS])
    ax.set_xlabel("expansion"); ax.set_ylabel("strength")
    ax.set_title(f"E27b AUROC (clean={agg['clean_only']['mean']:.3f})")
    for i, st in enumerate(STRENGTHS):
        for j in range(len(EXPANSIONS)):
            unsafe = morph_cache[st] < MORPH_GUARD
            txt = f"{grid[i,j]:.3f}" + ("\n⚠" if unsafe else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="red" if unsafe else ("white" if grid[i, j] < grid.mean() else "black"), fontsize=9)
    fig.colorbar(im, ax=ax, label="AUROC")
    sts = list(morph_cache.keys()); corrs = [morph_cache[s] for s in sts]
    ax2.plot(sts, corrs, "o-", color="#C44E52")
    ax2.axhline(MORPH_GUARD, color="gray", ls="--", label=f"guard {MORPH_GUARD}")
    ax2.set_xlabel("augmentation strength"); ax2.set_ylabel("source correlation (label validity)")
    ax2.set_title("Morphology preservation vs strength"); ax2.legend(fontsize=8); ax2.set_ylim(0.5, 1.0)
    plt.tight_layout(); plt.savefig(RESULTS / "augment_plateau.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
