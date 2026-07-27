#!/usr/bin/env python3
"""
E39 — Recalibrate the augmenter to the E38-measured REAL Apple profile, then
test coverage + morphology against real SJLIFE Apple as ground truth.

Question: does calibrating CalibratedAWAugmenter to the REAL paired target
(E38: baseline-wander UP to ~0.20, HF flat) produce augmented clinical Lead-I
whose modality distribution matches real Apple BETTER than:
   (a) light-DR (StochasticAWAugmenter, strength 0.5) — the E37 recommendation,
   (b) CinC-calibrated DR — the old E33 target (over-injected wander/HF),
   (c) clean clinical (no augmentation) — floor,
while PRESERVING morphology (QRS-band corr + R-peak match, the E33 guard)?

Ground truth target = real SJLIFE Apple modality profile (E38).
Source = SJLIFE clinical Lead-I (same cohort, so the transform is the honest one).

HONEST CAVEATS:
- Coverage/fidelity is a NECESSARY-not-sufficient check (E25b: fidelity != utility).
  This tells us the augmenter now aims at the right target; the downstream-AUROC
  proof still needs labels (SJLIFE has none).
- Distribution distance is population-level (recordings not beat-aligned).
- Morphology guard runs per-signal on the augmented clinical (source known).
"""
import os, sys, glob, json
import numpy as np
from scipy.signal import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/projects/ecg-modality-invariance/src"))
from aw_generator import (signal_modality_stats, measure_distribution,       # noqa
                          qrs_morphology_preserved, CalibratedAWAugmenter,
                          StochasticAWAugmenter)

ROOT = os.path.expanduser("~/projects/ecg-modality-invariance")
SJ = f"{ROOT}/data/sjlife"
OUT = f"{ROOT}/results/39_recalibrated_augment"
os.makedirs(OUT, exist_ok=True)
FS_APPLE, FS_CLIN, FS_COMMON, SIGLEN = 512.0, 500.0, 100.0, 1000
rng = np.random.default_rng(0)

AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]

def to_common(x, fs_in, n_target=None):
    n_out = int(round(len(x) * FS_COMMON / fs_in))
    y = resample(np.asarray(x, float), n_out)
    if n_target:
        if len(y) >= n_target: y = y[:n_target]
        else: y = np.concatenate([y, np.zeros(n_target - len(y))])
    return y

def clinical_leadI(path):
    c = np.squeeze(np.load(path).astype(np.float64))
    if c.ndim == 2 and c.shape[0] in (12, 15): return c[0]
    if c.ndim == 2 and c.shape[1] in (12, 15): return c[:, 0]
    return c.ravel()

def profile_dist(prof_a_mean, tgt_mean, tgt_std):
    v = np.array([(prof_a_mean[k] - tgt_mean[k]) / max(tgt_std[k], 1e-6) for k in AXES])
    return float(np.sqrt(np.mean(v ** 2)))

def main():
    # ---- load real paired signals (common 100 Hz, fixed length) ----
    ap = sorted(glob.glob(f"{SJ}/apple/apple_ecg_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    apple_c, clin_c = [], []
    for a in ap:
        n = a.split("_")[-1].split(".")[0]
        cp = f"{SJ}/clinical/clinical_ecg_{n}.npy"
        if not os.path.exists(cp): continue
        apple_c.append(to_common(np.load(a).astype(np.float64).ravel(), FS_APPLE, SIGLEN))
        clin_c.append(to_common(clinical_leadI(cp), FS_CLIN, SIGLEN))
    print(f"paired: {len(apple_c)}")

    # ---- target = real Apple profile; source clin profile ----
    apple_prof = measure_distribution(apple_c, FS_COMMON, n=len(apple_c))
    clin_prof  = measure_distribution(clin_c,  FS_COMMON, n=len(clin_c))
    tgt_mean = {k: apple_prof[k][0] for k in AXES}
    tgt_std  = {k: apple_prof[k][1] for k in AXES}
    print("real Apple target:", {k: round(tgt_mean[k],3) for k in AXES})

    # ---- the OLD CinC target that E33 calibrated to (from E34 report / known values) ----
    # E34 documented CinC bw_energy ~0.256, hf ~0.004, kurt ~8.3 (the over-injection source).
    cinc_prof = {"kurtosis": (8.3, 4.0), "bw_energy": (0.256, 0.10),
                 "qrs_energy": (0.264, 0.10), "hf_energy": (0.004, 0.01),
                 "mid_energy": (0.384, 0.10)}

    # ---- build the four variants ----
    variants = {}
    # (0) clean
    variants["clean"] = clin_c
    # (a) light-DR
    light = StochasticAWAugmenter(fs=FS_COMMON, siglen=SIGLEN, strength=0.5, seed=1)
    variants["light_DR"] = [light.generate(x) for x in clin_c]
    # (b) CinC-calibrated (old target)
    cinc_cal = CalibratedAWAugmenter(clin_prof, cinc_prof, fs=FS_COMMON, siglen=SIGLEN, seed=2, cover=1.3)
    variants["cinc_calDR"] = [cinc_cal.generate(x) for x in clin_c]
    # (c) E38-real-calibrated (NEW)
    real_cal = CalibratedAWAugmenter(clin_prof, apple_prof, fs=FS_COMMON, siglen=SIGLEN, seed=3, cover=1.3)
    variants["real_calDR"] = [real_cal.generate(x) for x in clin_c]

    # ---- evaluate coverage + morphology ----
    print("\n===== COVERAGE (distance to REAL Apple, lower=better) + MORPHOLOGY =====")
    rows = {}
    for name, sigs in variants.items():
        prof = measure_distribution(sigs, FS_COMMON, n=len(sigs))
        pm = {k: prof[k][0] for k in AXES}
        d = profile_dist(pm, tgt_mean, tgt_std)
        # morphology vs the clean source (only meaningful for augmented variants)
        if name == "clean":
            qrs, rpk = 1.0, 1.0
        else:
            qs, rs = [], []
            for i in range(0, len(sigs), 5):  # subsample for speed
                q, r = qrs_morphology_preserved(clin_c[i], sigs[i], FS_COMMON)
                if not np.isnan(q): qs.append(q)
                if not np.isnan(r): rs.append(r)
            qrs, rpk = float(np.mean(qs)), float(np.mean(rs))
        rows[name] = dict(dist=d, qrs=qrs, rpeak=rpk, profile=pm)
        print(f"  {name:12s} dist={d:6.3f}   QRS-corr={qrs:.3f}  R-peak={rpk:.3f}   "
              f"bw={pm['bw_energy']:.3f} hf={pm['hf_energy']:.3f} kurt={pm['kurtosis']:.2f}")

    # ---- verdict ----
    aug = {k: v for k, v in rows.items() if k != "clean"}
    # label valid only if QRS>=0.85 and R-peak>=0.9
    valid = {k: v for k, v in aug.items() if v["qrs"] >= 0.85 and v["rpeak"] >= 0.90}
    best = min(valid, key=lambda k: valid[k]["dist"]) if valid else None
    print(f"\n  clean floor dist   : {rows['clean']['dist']:.3f}")
    print(f"  label-valid variants: {list(valid.keys())}")
    print(f"  BEST label-valid    : {best} (dist {valid[best]['dist']:.3f})" if best else "  none label-valid")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = list(rows.keys())
    dists = [rows[n]["dist"] for n in names]
    cols = ["#999999", "#66aa66", "#cc7744", "#d4a017"]
    ax[0].bar(names, dists, color=cols)
    ax[0].axhline(rows["clean"]["dist"], color="k", ls=":", lw=1, label="clean floor")
    ax[0].set_ylabel("distance to REAL Apple (norm)"); ax[0].set_title("Coverage: which augmenter aims at real Apple?")
    for i, d in enumerate(dists): ax[0].text(i, d+0.02, f"{d:.2f}", ha="center", fontsize=9)
    ax[0].tick_params(axis="x", rotation=20)

    idx = np.arange(len(AXES)); w = 0.2
    ax[1].bar(idx-1.5*w, [tgt_mean[k] for k in AXES], w, label="REAL Apple (target)", color="#d4a017", edgecolor="k")
    ax[1].bar(idx-0.5*w, [rows["clean"]["profile"][k] for k in AXES], w, label="clean", color="#999999")
    ax[1].bar(idx+0.5*w, [rows["cinc_calDR"]["profile"][k] for k in AXES], w, label="cinc_calDR", color="#cc7744")
    ax[1].bar(idx+1.5*w, [rows["real_calDR"]["profile"][k] for k in AXES], w, label="real_calDR", color="#66aa66")
    ax[1].set_xticks(idx); ax[1].set_xticklabels(AXES, rotation=30, ha="right")
    ax[1].set_title("Profile match to real Apple"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/recalibration.png", dpi=110)
    print(f"\nSaved figure -> {OUT}/recalibration.png")

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump({"n": len(apple_c),
                   "real_apple_target": tgt_mean,
                   "clinical_source": {k: clin_prof[k][0] for k in AXES},
                   "variants": {k: {"dist": v["dist"], "qrs": v["qrs"], "rpeak": v["rpeak"],
                                    "profile": v["profile"]} for k, v in rows.items()},
                   "best_label_valid": best,
                   "caveats": ["coverage is necessary-not-sufficient (fidelity != utility, E25b)",
                               "population-level distance; not beat-aligned",
                               "downstream AUROC still needs labels (SJLIFE has none)"]},
                  f, indent=2)
    print(f"Saved metrics -> {OUT}/metrics.json\nDONE.")

if __name__ == "__main__":
    main()
