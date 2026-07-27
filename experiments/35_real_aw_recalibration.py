"""Experiment 35 — Recalibrate domain randomization toward REAL Apple Watch.

E34 (n=20) found the calibrated-DR method was calibrated to the WRONG target:
CinC is a mediocre AW proxy (dist 0.477) — worse than raw clinical (0.253) — and
calibrating toward CinC's inflated baseline-wander pushed the augmenter AWAY from
real AW (1.136). E35 fixes this with (a) the FULL 1000 real Apple Watch waveforms
(HOME data-for-predicting/Apple_Watch_waveform.csv, 1000 patients x 6000 samples
@ 500Hz) for a trustworthy profile, and (b) recalibrating DR toward REAL AW stats
instead of CinC.

LICENSE-COMPLIANT: distribution statistics only. NO training/fine-tuning on HOME
(these are the unlabeled prediction cohort — we only measure signal stats).

Compares coverage of real AW by:
  - clinical (no aug)
  - CinC-calibrated DR      (E33 recipe — the mis-calibrated one)
  - REAL-AW-calibrated DR    (new: calibrate toward measured real AW)
  - light DR                 (minimal perturbation, since AW is clean)

Question: does calibrating toward real AW (or going light) cover real AW far
better than the CinC-calibrated recipe? Confirms the corrected deployment target.

Honesty: n=1000 real AW (robust profile now), stats-based coverage on 5 axes,
HOME eval-only, 500->100Hz resample. This is a DISTRIBUTION study (no AUROC —
would need labels/training on HOME, which we don't do).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.aw_generator import (measure_distribution, CalibratedAWAugmenter,
                              StochasticAWAugmenter, qrs_morphology_preserved)
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "35_real_aw_recalibration"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000
AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]


def load_full_home_aw(max_patients=1000, fs_home=500):
    """Full HOME Apple Watch cohort: 1000 patients x 6000 samples @ 500Hz.
    Columns = patients (UIDs), rows = samples. Eval-only (stats)."""
    path = Path.home() / "data" / "HOME" / "data-for-predicting" / "Apple_Watch_waveform.csv"
    from math import gcd
    g = gcd(int(fs_home), int(FS)); up = int(FS) // g; down = int(fs_home) // g
    with open(path) as f:
        r = csv.reader(f); header = next(r)
        cols = [[] for _ in header]
        for row in r:
            for i, v in enumerate(row):
                if i < len(cols):
                    try: cols[i].append(float(v))
                    except: cols[i].append(np.nan)
    out = []
    for c in cols[:max_patients]:
        v = np.asarray(c, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 500: continue
        v = resample_poly(v, up, down)
        # one 10s window per patient (SIGLEN samples)
        if v.size < SIGLEN: continue
        seg = v[:SIGLEN]; seg = (seg - seg.mean()) / (seg.std() + 1e-9)
        out.append(seg.astype(np.float32))
    return out


def main():
    print("Loading clinical PTB-XL Lead-I ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=350)
    clin = [r["leadI"] for r in ptb["train"]]
    print(f"  clinical n={len(clin)}", flush=True)

    print("Loading CinC ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=350); cinc = [r["sig"] for r in (cA + cN)]

    print("Loading FULL real Apple Watch cohort (HOME, eval-only) ...", flush=True)
    aw = load_full_home_aw(max_patients=1000)
    print(f"  real AW n={len(aw)} patients", flush=True)

    clin_stats = measure_distribution(clin, n=len(clin))
    cinc_stats = measure_distribution(cinc, n=len(cinc))
    aw_stats = measure_distribution(aw, n=len(aw))

    # Four augmentation strategies
    cinc_cal = CalibratedAWAugmenter(clin_stats, cinc_stats, seed=0, cover=1.3)   # E33 (mis-cal)
    aw_cal = CalibratedAWAugmenter(clin_stats, aw_stats, seed=0, cover=1.3)        # NEW: toward real AW
    light = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=0.5, seed=0)      # light (AW is clean)

    aug_cinc = [cinc_cal.generate(c) for c in clin]
    aug_aw = [aw_cal.generate(c) for c in clin]
    aug_light = [light.generate(c) for c in clin]

    profiles = {
        "clinical": clin_stats,
        "cinc": cinc_stats,
        "real_aw": aw_stats,
        "cinc_calibrated_DR": measure_distribution(aug_cinc, n=len(aug_cinc)),
        "aw_calibrated_DR": measure_distribution(aug_aw, n=len(aug_aw)),
        "light_DR": measure_distribution(aug_light, n=len(aug_light)),
    }

    def dist(a, b):
        return float(np.mean([abs(a[ax][0] - b[ax][0]) / (clin_stats[ax][1] + 1e-9) for ax in AXES]))

    key = {
        "clinical_to_aw": dist(clin_stats, aw_stats),
        "cinc_to_aw": dist(cinc_stats, aw_stats),
        "cinc_calDR_to_aw": dist(profiles["cinc_calibrated_DR"], aw_stats),
        "aw_calDR_to_aw": dist(profiles["aw_calibrated_DR"], aw_stats),
        "light_DR_to_aw": dist(profiles["light_DR"], aw_stats),
    }

    # morphology preservation (QRS-band + R-peak) for each strategy
    morph = {}
    for name, augset in [("cinc_calibrated_DR", aug_cinc), ("aw_calibrated_DR", aug_aw), ("light_DR", aug_light)]:
        qs, rs = [], []
        for o, a in zip(clin[:80], augset[:80]):
            q, rp = qrs_morphology_preserved(o, a)
            if np.isfinite(q): qs.append(q)
            if np.isfinite(rp): rs.append(rp)
        morph[name] = {"qrs_corr": float(np.mean(qs)), "rpeak_match": float(np.mean(rs))}

    print("\n===== MODALITY PROFILES (mean per axis, n_AW=%d) =====" % len(aw), flush=True)
    print(f"  {'axis':11s} {'clin':>8s} {'cinc':>8s} {'REAL_AW':>8s} {'cinc_cal':>9s} {'aw_cal':>8s} {'light':>8s}", flush=True)
    for ax in AXES:
        print(f"  {ax:11s} {clin_stats[ax][0]:8.3f} {cinc_stats[ax][0]:8.3f} {aw_stats[ax][0]:8.3f} "
              f"{profiles['cinc_calibrated_DR'][ax][0]:9.3f} {profiles['aw_calibrated_DR'][ax][0]:8.3f} "
              f"{profiles['light_DR'][ax][0]:8.3f}", flush=True)

    print("\n===== DISTANCE TO REAL AW (lower=better) =====", flush=True)
    for k, v in key.items():
        print(f"  {k:22s} {v:.3f}", flush=True)

    print("\n===== MORPHOLOGY PRESERVATION (QRS-band corr / R-peak) =====", flush=True)
    for k, v in morph.items():
        print(f"  {k:22s} QRS {v['qrs_corr']:.3f}  R-peak {v['rpeak_match']:.3f}", flush=True)

    best = min(key, key=lambda k: key[k])
    print(f"\n  CLOSEST to real AW: {best} ({key[best]:.3f})", flush=True)

    metrics = {
        "fs": FS, "n_clinical": len(clin), "n_cinc": len(cinc), "n_real_aw": len(aw),
        "profiles": profiles, "distances_to_real_aw": key, "morphology": morph, "closest": best,
        "question": "recalibrate DR toward REAL AW (n=1000) — does it cover real AW better than CinC-calibrated?",
        "honesty": ["n=1000 real AW (robust profile)", "stats-based coverage 5 axes",
                    "HOME eval-only, NO training", "500->100Hz resample", "distribution study, no AUROC"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(profiles, key)
    print("\nDONE.", flush=True)


def _plot(profiles, key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 2]})
    order = ["clinical", "cinc", "real_aw", "cinc_calibrated_DR", "aw_calibrated_DR", "light_DR"]
    cols = ["#4C72B0", "#DD8452", "#333333", "#C44E52", "#55A868", "#8172B2"]
    x = np.arange(len(AXES)); w = 0.14
    for i, (name, col) in enumerate(zip(order, cols)):
        vals = [profiles[name][ax][0] for ax in AXES]
        ax1.bar(x + (i - 2.5) * w, vals, w, label=name, color=col)
    ax1.set_xticks(x); ax1.set_xticklabels(AXES, fontsize=8, rotation=20)
    ax1.set_ylabel("stat value"); ax1.set_title("E35: profiles vs REAL AW (n=1000)")
    ax1.legend(fontsize=7)
    # distance bars
    dk = ["clinical_to_aw", "cinc_to_aw", "cinc_calDR_to_aw", "aw_calDR_to_aw", "light_DR_to_aw"]
    dlabels = ["clinical", "CinC", "CinC-cal DR\n(E33)", "AW-cal DR\n(new)", "light DR"]
    dvals = [key[k] for k in dk]
    dcols = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B2"]
    bars = ax2.bar(range(len(dk)), dvals, color=dcols)
    ax2.set_xticks(range(len(dk))); ax2.set_xticklabels(dlabels, fontsize=8)
    ax2.set_ylabel("distance to REAL AW (lower=better)")
    ax2.set_title("Which strategy is closest to real Apple Watch?")
    for b, v in zip(bars, dvals): ax2.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "real_aw_recalibration.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
