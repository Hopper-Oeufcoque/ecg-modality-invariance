#!/usr/bin/env python3
"""
E38 — Real paired clinical Lead-I vs Apple Watch: measured modality profile.

FIRST experiment on REAL paired hardware (SJLIFE, 243 patients, same person
recorded both ways). Purpose: replace the simulated/guessed clinical->watch
target profile (used by CalibratedAWAugmenter / light-DR in E33-E37) with the
TRUE measured one, and characterize the amplitude offset between modalities.

HONEST CAVEATS (load-bearing):
- Paired = same patient, but the two recordings are ~64 min apart (metadata
  Minutes_ABS_Btw_Clinical_Apple). They are NOT the same cardiac cycles, so
  per-beat / sample-wise morphology regression is INVALID. Only distributional
  and population-spectral characterization is valid here.
- Clinical = 12-lead 500 Hz x 10 s; we use Lead I (index 0) only.
- Apple = single lead 512 Hz x 30 s.
- Both resampled to a common 100 Hz for stats parity with prior experiments.
- n=243, young SJLIFE survivor cohort -> modality-transform lab, not a disease
  training set. No CIs beyond simple spread.
"""
import os, glob, json
import numpy as np
from scipy.signal import resample
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.expanduser("~/projects/ecg-modality-invariance/src"))
from aw_generator import signal_modality_stats, measure_distribution  # noqa

ROOT = os.path.expanduser("~/projects/ecg-modality-invariance")
SJ = f"{ROOT}/data/sjlife"
OUT = f"{ROOT}/results/38_paired_transform"
os.makedirs(OUT, exist_ok=True)

FS_APPLE, FS_CLIN, FS_COMMON = 512.0, 500.0, 100.0

def load_pairs():
    ap = sorted(glob.glob(f"{SJ}/apple/apple_ecg_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    pairs = []
    for a in ap:
        n = a.split("_")[-1].split(".")[0]
        c = f"{SJ}/clinical/clinical_ecg_{n}.npy"
        if os.path.exists(c):
            pairs.append((n, a, c))
    return pairs

def to_common(x, fs_in):
    n_out = int(round(len(x) * FS_COMMON / fs_in))
    return resample(np.asarray(x, float), n_out)

def clinical_leadI(path):
    c = np.load(path).astype(np.float64)
    c = np.squeeze(c)            # (12, 5000)
    if c.ndim == 2 and c.shape[0] in (12, 15):
        return c[0]              # Lead I
    if c.ndim == 2 and c.shape[1] in (12, 15):
        return c[:, 0]
    return c.ravel()

def main():
    pairs = load_pairs()
    print(f"paired records: {len(pairs)}")

    apple_amp, clin_amp, ratio = [], [], []
    apple_stats, clin_stats = [], []
    apple_sigs_c, clin_sigs_c = [], []

    for n, ap, cp in pairs:
        a = np.load(ap).astype(np.float64).ravel()
        ci = clinical_leadI(cp)
        # amplitude on native scale (robust p99 of |x|, and peak-to-peak)
        a_amp = np.percentile(np.abs(a), 99)
        c_amp = np.percentile(np.abs(ci), 99)
        apple_amp.append(a_amp); clin_amp.append(c_amp)
        if c_amp > 1e-6:
            ratio.append(a_amp / c_amp)
        # resample to common fs, then modality stats (stats internally z-score)
        a_c = to_common(a, FS_APPLE)
        c_c = to_common(ci, FS_CLIN)
        apple_sigs_c.append(a_c); clin_sigs_c.append(c_c)
        apple_stats.append(signal_modality_stats(a_c, FS_COMMON))
        clin_stats.append(signal_modality_stats(c_c, FS_COMMON))

    apple_amp = np.array(apple_amp); clin_amp = np.array(clin_amp); ratio = np.array(ratio)

    # ---- amplitude offset ----
    print("\n===== AMPLITUDE (native stored units, p99 of |x|) =====")
    print(f"  Apple    median={np.median(apple_amp):8.1f}  IQR[{np.percentile(apple_amp,25):.0f},{np.percentile(apple_amp,75):.0f}]")
    print(f"  Clinical median={np.median(clin_amp):8.1f}  IQR[{np.percentile(clin_amp,25):.0f},{np.percentile(clin_amp,75):.0f}]")
    print(f"  Apple/Clin gain ratio: median={np.median(ratio):.2f}  IQR[{np.percentile(ratio,25):.2f},{np.percentile(ratio,75):.2f}]  spread(cv)={ratio.std()/ratio.mean():.2f}")

    # ---- modality profiles (post z-score, shape only) ----
    def agg(stats):
        keys = stats[0].keys()
        return {k: (float(np.mean([s[k] for s in stats])),
                    float(np.std([s[k] for s in stats]))) for k in keys}
    ap_prof = agg(apple_stats); cl_prof = agg(clin_stats)

    print("\n===== MEASURED MODALITY PROFILE (z-scored, common 100 Hz) =====")
    print(f"  {'axis':12s}{'clinical(realLeadI)':>22s}{'appleWatch(real)':>20s}")
    for k in ap_prof:
        print(f"  {k:12s}{cl_prof[k][0]:>12.3f}±{cl_prof[k][1]:<7.3f}{ap_prof[k][0]:>10.3f}±{ap_prof[k][1]:<7.3f}")

    # ---- what did our prior recipe ASSUME? (compare to CinC-derived target used in E33) ----
    # E33/E34 flagged proxy bw_energy target ~0.12 (clinical) and over-injected to ~0.5.
    # Here we report the REAL target so future calibration uses it.
    real_target = {k: ap_prof[k][0] for k in ap_prof}
    with open(f"{OUT}/real_apple_target_profile.json", "w") as f:
        json.dump({"apple_profile": ap_prof, "clinical_leadI_profile": cl_prof,
                   "amplitude_gain_ratio_median": float(np.median(ratio)),
                   "n": len(pairs)}, f, indent=2)

    # ---- normalized-distance clinical->apple (per-axis std-normalized, like E34/E37) ----
    keys = list(ap_prof.keys())
    sd = np.array([max(ap_prof[k][1], 1e-6) for k in keys])
    cvec = np.array([cl_prof[k][0] for k in keys])
    avec = np.array([ap_prof[k][0] for k in keys])
    dist = float(np.sqrt(np.mean(((cvec - avec) / sd) ** 2)))
    print(f"\n  clinical-LeadI -> real-Apple normalized distance: {dist:.3f}")
    print("  (compare: E37 clinical->HOME-AW 0.25-0.50; this is the DIRECT paired read)")

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0,0].hist(np.log10(ratio), bins=30, color="#d4a017", edgecolor="k")
    ax[0,0].axvline(np.log10(np.median(ratio)), color="r", ls="--", label=f"median {np.median(ratio):.1f}x")
    ax[0,0].set_title("Apple/Clinical amplitude gain ratio (log10)"); ax[0,0].legend()
    ax[0,0].set_xlabel("log10(gain ratio)")

    idx = np.arange(len(keys))
    w = 0.35
    ax[0,1].bar(idx-w/2, [cl_prof[k][0] for k in keys], w, yerr=[cl_prof[k][1] for k in keys],
               label="clinical Lead-I", color="#4477aa", capsize=3)
    ax[0,1].bar(idx+w/2, [ap_prof[k][0] for k in keys], w, yerr=[ap_prof[k][1] for k in keys],
               label="Apple Watch", color="#d4a017", capsize=3)
    ax[0,1].set_xticks(idx); ax[0,1].set_xticklabels(keys, rotation=30, ha="right")
    ax[0,1].set_title("Measured modality profile (z-scored)"); ax[0,1].legend()

    # example waveforms (2 s), z-scored for shape comparison
    t = np.arange(200) / FS_COMMON
    az = (apple_sigs_c[0][:200] - np.mean(apple_sigs_c[0][:200])) / (np.std(apple_sigs_c[0][:200])+1e-9)
    cz = (clin_sigs_c[0][:200] - np.mean(clin_sigs_c[0][:200])) / (np.std(clin_sigs_c[0][:200])+1e-9)
    ax[1,0].plot(t, cz, color="#4477aa", label="clinical Lead-I"); ax[1,0].plot(t, az, color="#d4a017", alpha=0.8, label="Apple Watch")
    ax[1,0].set_title("Example z-scored waveforms (2 s, NOT beat-aligned)"); ax[1,0].legend(); ax[1,0].set_xlabel("s")

    # amplitude scatter
    ax[1,1].scatter(clin_amp, apple_amp, s=12, alpha=0.5, color="#aa3377")
    ax[1,1].set_xlabel("clinical Lead-I p99|x| (native)"); ax[1,1].set_ylabel("Apple p99|x| (native)")
    ax[1,1].set_title(f"Per-patient amplitude (median gain {np.median(ratio):.1f}x)")
    fig.tight_layout(); fig.savefig(f"{OUT}/paired_profile.png", dpi=110)
    print(f"\nSaved figure -> {OUT}/paired_profile.png")

    metrics = dict(
        n=len(pairs),
        amplitude={"apple_median_p99": float(np.median(apple_amp)),
                   "clinical_median_p99": float(np.median(clin_amp)),
                   "gain_ratio_median": float(np.median(ratio)),
                   "gain_ratio_iqr": [float(np.percentile(ratio,25)), float(np.percentile(ratio,75))],
                   "gain_ratio_cv": float(ratio.std()/ratio.mean())},
        apple_profile=ap_prof, clinical_leadI_profile=cl_prof,
        clinical_to_apple_norm_distance=dist,
        real_apple_target_profile=real_target,
        caveats=["recordings ~64min apart -> not beat-aligned; distributional only",
                 "n=243 young SJLIFE survivor cohort", "Lead I only from 12-lead clinical",
                 "resampled to common 100Hz"],
    )
    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {OUT}/metrics.json")
    print("DONE.")

if __name__ == "__main__":
    main()
