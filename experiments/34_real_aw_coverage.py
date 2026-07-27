"""Experiment 34 — Does calibrated-DR COVER the REAL Apple Watch distribution?

THE DEEPEST CAVEAT: every result (E26-E33b) uses CinC 2017 as an Apple Watch
PROXY (E6c validated it at distance 0.247). But our winning zero-shot method
(E33 calibrated domain randomization) calibrates the augmenter toward CinC. The
load-bearing question: does an augmenter built from CinC stats actually COVER
the REAL Apple Watch distribution? If yes, the whole calibrated-DR result is
validated on true hardware. If no, we learn exactly what's missing.

We have 20 REAL Apple Watch Lead-I ECGs (HOME dataset, 500 Hz, 30s, 0.01 mV).
LICENSE-COMPLIANT: we measure DISTRIBUTION STATISTICS only — NO training,
fine-tuning, or model adaptation on HOME data (eval/analysis use only).

Analysis (all on the modality-stat axes from src.aw_generator):
  1. Profile 4 distributions: clinical PTB-XL Lead-I, CinC (proxy), REAL AW
     (HOME), and calibrated-DR-augmented-clinical.
  2. Distance matrix between distributions (standardized per-axis).
  3. COVERAGE test: for each axis, does the calibrated-augmented range envelope
     the real-AW mean? (the property that makes zero-shot DR work)
  4. Is CinC a good proxy for real AW (validates E26-E33b) on these axes?

Honesty: n=20 real AW (small — coarse profile only), stats-based coverage (not
full-manifold), HOME is eval-only (no training). Different sampling rate (500Hz)
resampled to 100Hz to match the pipeline.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aw_generator import (measure_distribution, signal_modality_stats,
                              CalibratedAWAugmenter)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "34_real_aw_coverage"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000
AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]


def load_home_aw(fs_home=500):
    """Load real Apple Watch Lead-I from HOME (eval-only). Resample 500->100 Hz."""
    import csv
    files = sorted(glob.glob(str(Path.home() / "data" / "HOME" / "data" / "ecg" / "*.csv")))
    from math import gcd
    g = gcd(int(fs_home), int(FS)); up = int(FS) // g; down = int(fs_home) // g
    out = []
    for f in files:
        with open(f) as fh:
            vals = [float(x[0]) for x in list(csv.reader(fh))[1:] if x and x[0]]
        v = np.asarray(vals, dtype=np.float64)  # 0.01 mV units, irrelevant after z-norm
        if v.size < 1000: continue
        v = resample_poly(v, up, down)
        # take multiple 10s windows to boost sample count from 20 recordings
        for start in range(0, max(1, len(v) - SIGLEN), SIGLEN):
            seg = v[start:start + SIGLEN]
            if seg.size < SIGLEN: continue
            seg = (seg - seg.mean()) / (seg.std() + 1e-9)
            out.append(seg.astype(np.float32))
    return out


def main():
    print("Loading clinical PTB-XL Lead-I ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=350)
    clin = [r["leadI"] for r in ptb["train"]]
    print(f"  clinical n={len(clin)}", flush=True)

    print("Loading CinC (proxy) ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=350); cinc = [r["sig"] for r in (cA + cN)]
    print(f"  CinC n={len(cinc)}", flush=True)

    print("Loading REAL Apple Watch (HOME, eval-only) ...", flush=True)
    aw = load_home_aw()
    print(f"  real AW n={len(aw)} windows from 20 recordings", flush=True)

    # profiles
    clin_stats = measure_distribution(clin, n=len(clin))
    cinc_stats = measure_distribution(cinc, n=len(cinc))
    aw_stats = measure_distribution(aw, n=len(aw))

    # calibrated-DR augmented clinical (calibrated toward CinC, as in E33)
    cal = CalibratedAWAugmenter(clin_stats, cinc_stats, seed=0, cover=1.3)
    aug = [cal.generate(c) for c in clin]
    aug_stats = measure_distribution(aug, n=len(aug))

    profiles = {"clinical": clin_stats, "cinc_proxy": cinc_stats,
                "real_aw": aw_stats, "calibrated_aug": aug_stats}

    print("\n===== MODALITY PROFILES (mean per axis) =====", flush=True)
    print(f"  {'axis':12s} {'clinical':>10s} {'cinc':>10s} {'real_AW':>10s} {'calib_aug':>10s}", flush=True)
    for ax in AXES:
        print(f"  {ax:12s} {clin_stats[ax][0]:10.3f} {cinc_stats[ax][0]:10.3f} "
              f"{aw_stats[ax][0]:10.3f} {aug_stats[ax][0]:10.3f}", flush=True)

    # standardized distances (z by clinical axis std across the 4 means)
    def dist(a, b):
        ds = []
        for ax in AXES:
            scale = clin_stats[ax][1] + 1e-9
            ds.append(abs(a[ax][0] - b[ax][0]) / scale)
        return float(np.mean(ds))
    D = {}
    names = list(profiles.keys())
    for i in names:
        for j in names:
            if i < j: D[f"{i}__{j}"] = dist(profiles[i], profiles[j])

    print("\n===== KEY DISTANCES (mean standardized axis gap) =====", flush=True)
    print(f"  clinical  -> real_AW : {dist(clin_stats, aw_stats):.3f}  (the raw gap)", flush=True)
    print(f"  cinc      -> real_AW : {dist(cinc_stats, aw_stats):.3f}  (is proxy good?)", flush=True)
    print(f"  calib_aug -> real_AW : {dist(aug_stats, aw_stats):.3f}  (does calibrated DR reach real AW?)", flush=True)

    # COVERAGE: does calibrated-aug distribution envelope the real-AW mean per axis?
    print("\n===== COVERAGE of real AW by calibrated augmentation =====", flush=True)
    coverage = {}
    for ax in AXES:
        aug_lo = aug_stats[ax][0] - aug_stats[ax][1]
        aug_hi = aug_stats[ax][0] + aug_stats[ax][1]
        covered = bool(aug_lo <= aw_stats[ax][0] <= aug_hi)
        # directional: did aug move FROM clinical TOWARD real AW?
        moved_right_way = bool(np.sign(aug_stats[ax][0] - clin_stats[ax][0]) ==
                               np.sign(aw_stats[ax][0] - clin_stats[ax][0])) if abs(aw_stats[ax][0]-clin_stats[ax][0])>1e-6 else True
        coverage[ax] = {"covered_within_1std": covered, "moved_toward_aw": moved_right_way,
                        "clinical": clin_stats[ax][0], "aug": aug_stats[ax][0], "real_aw": aw_stats[ax][0]}
        flag = "COVERED" if covered else ("toward" if moved_right_way else "*** WRONG WAY ***")
        print(f"  {ax:12s} clin {clin_stats[ax][0]:7.3f} -> aug {aug_stats[ax][0]:7.3f} "
              f"(±{aug_stats[ax][1]:.3f}) vs real_AW {aw_stats[ax][0]:7.3f}  [{flag}]", flush=True)

    n_covered = sum(1 for a in coverage.values() if a["covered_within_1std"])
    n_toward = sum(1 for a in coverage.values() if a["moved_toward_aw"])
    print(f"\n  {n_covered}/{len(AXES)} axes COVERED within 1std; {n_toward}/{len(AXES)} moved toward real AW", flush=True)

    metrics = {
        "fs": FS, "n_clinical": len(clin), "n_cinc": len(cinc), "n_real_aw": len(aw),
        "profiles": profiles, "distances": D, "coverage": coverage,
        "key_distances": {
            "clinical_to_real_aw": dist(clin_stats, aw_stats),
            "cinc_to_real_aw": dist(cinc_stats, aw_stats),
            "calibrated_aug_to_real_aw": dist(aug_stats, aw_stats),
        },
        "question": "Does CinC-calibrated DR cover the REAL Apple Watch distribution? Is CinC a good AW proxy?",
        "honesty": ["n=20 real AW recordings (windowed) — coarse profile only",
                    "stats-based coverage, not full-manifold", "HOME eval-only (NO training)",
                    "500Hz resampled to 100Hz", "license-compliant: distribution analysis only"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(profiles, coverage)
    print("\nDONE.", flush=True)


def _plot(profiles, coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    # grouped bars per axis
    x = np.arange(len(AXES)); w = 0.2
    for i, (name, col) in enumerate(zip(["clinical", "cinc_proxy", "real_aw", "calibrated_aug"],
                                        ["#4C72B0", "#DD8452", "#333333", "#55A868"])):
        vals = [profiles[name][ax][0] for ax in AXES]
        ax1.bar(x + (i - 1.5) * w, vals, w, label=name, color=col)
    ax1.set_xticks(x); ax1.set_xticklabels(AXES, fontsize=8, rotation=20)
    ax1.set_ylabel("stat value (mean)"); ax1.set_title("E34: modality profiles — clinical / CinC / REAL AW / calibrated-aug")
    ax1.legend(fontsize=8)
    # coverage: clinical->aug->real_aw per axis (normalized arrows)
    for i, ax in enumerate(AXES):
        c = coverage[ax]
        ax2.plot([c["clinical"], c["aug"], c["real_aw"]], [i, i, i], "-o", markersize=4,
                 color="#55A868" if c["covered_within_1std"] else "#C44E52")
        ax2.annotate("", xy=(c["real_aw"], i), xytext=(c["clinical"], i),
                     arrowprops=dict(arrowstyle="->", alpha=0.3))
    ax2.set_yticks(range(len(AXES))); ax2.set_yticklabels(AXES, fontsize=8)
    ax2.set_xlabel("stat value"); ax2.set_title("Coverage: clinical → calib-aug → real AW\n(green=covered within 1std)")
    plt.tight_layout(); plt.savefig(RESULTS / "real_aw_coverage.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
