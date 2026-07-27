"""Experiment 37 — CORRECTED sampling-rate real-AW analysis (retracts E35/E36).

BUG FOUND: the HOME 'data-for-predicting/Apple_Watch_waveform.csv' (1000-patient
prediction cohort) is sampled at 200 Hz (README + heart-rate check: 66 bpm @200Hz
vs 100 @500Hz; 6000 samples/200Hz = 30s). E35 and E36 wrongly treated it as
500 Hz, stretching the frequency axis 2.5x. Their "real AW has high HF energy /
bandwidth gap" conclusion is a sampling-rate ARTIFACT and is RETRACTED.

E37 redoes it correctly:
  - Real AW (Source B, wide file): TRUE 200 Hz
  - Real AW (Source A, data/ecg/, 20 recordings): TRUE 500 Hz (cross-check)
  - Clinical PTB-XL Lead-I: 500 Hz
  All resampled to a COMMON 100 Hz (Nyquist 50 Hz, within both sources' bands)
  using the CORRECT source rates. Then compare band-energy profiles honestly.

Question: with correct sampling rates, is there ANY real clinical->AW modality
gap, and of what kind? Or was the whole E35/E36 gap an artifact?

Honesty: distribution study only (no AUROC), HOME eval-only. Both AW sources
profiled to check internal consistency. 100 Hz common rate loses >50 Hz content
(fine — clinical is bandlimited there anyway).
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "37_corrected_sampling"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000
AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]


def resample_to_100(v, fs_src):
    from math import gcd
    g = gcd(int(fs_src), int(FS)); up = int(FS) // g; down = int(fs_src) // g
    return resample_poly(v, up, down)


def stats_at(sig):
    x = np.asarray(sig, float); x = (x - x.mean()) / (x.std() + 1e-9)
    f = np.fft.rfftfreq(len(x), 1 / FS); P = np.abs(np.fft.rfft(x)) ** 2; T = P.sum() + 1e-9
    def band(lo, hi): return P[(f >= lo) & (f < hi)].sum() / T
    return {"kurtosis": float(sstats.kurtosis(x)), "bw_energy": float(band(0, 1)),
            "qrs_energy": float(band(5, 15)), "hf_energy": float(band(30, 50)),
            "mid_energy": float(band(1, 5))}


def profile(sigs):
    ks = [stats_at(s) for s in sigs]
    return {k: (float(np.mean([d[k] for d in ks])), float(np.std([d[k] for d in ks]))) for k in AXES}


def load_wide_200hz(max_patients=1000):
    """Source B: 1000-patient prediction cohort, TRUE 200 Hz."""
    path = Path.home() / "data" / "HOME" / "data-for-predicting" / "Apple_Watch_waveform.csv"
    with open(path) as f:
        r = csv.reader(f); next(r); cols = None
        for row in r:
            if cols is None: cols = [[] for _ in row]
            for i, v in enumerate(row):
                if i < len(cols):
                    try: cols[i].append(float(v))
                    except: cols[i].append(np.nan)
    out = []
    for c in cols[:max_patients]:
        v = np.asarray(c, float); v = v[np.isfinite(v)]
        if v.size < 2000: continue
        v = resample_to_100(v, 200)                     # CORRECT: 200 Hz
        if v.size < SIGLEN: continue
        seg = v[:SIGLEN]; out.append(((seg - seg.mean()) / (seg.std() + 1e-9)).astype(np.float32))
    return out


def load_ecg_500hz():
    """Source A: data/ecg/ 20 recordings, TRUE 500 Hz."""
    files = sorted(glob.glob(str(Path.home() / "data" / "HOME" / "data" / "ecg" / "*.csv")))
    out = []
    for fpath in files:
        with open(fpath) as fh:
            v = np.array([float(x[0]) for x in list(csv.reader(fh))[1:] if x and x[0]], dtype=float)
        if v.size < 5000: continue
        v = resample_to_100(v, 500)                     # CORRECT: 500 Hz
        for start in range(0, max(1, len(v) - SIGLEN), SIGLEN):
            seg = v[start:start + SIGLEN]
            if seg.size < SIGLEN: continue
            out.append(((seg - seg.mean()) / (seg.std() + 1e-9)).astype(np.float32))
    return out


def main():
    print("Loading clinical PTB-XL Lead-I (500Hz->100) ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=350)
    clin = [r["leadI"] for r in ptb["train"]]  # already 100 Hz in the loader
    print(f"  clinical n={len(clin)}", flush=True)

    print("Loading real AW Source B (wide, TRUE 200Hz->100) ...", flush=True)
    awB = load_wide_200hz(1000)
    print(f"  AW-B n={len(awB)}", flush=True)
    print("Loading real AW Source A (data/ecg, TRUE 500Hz->100) ...", flush=True)
    awA = load_ecg_500hz()
    print(f"  AW-A n={len(awA)}", flush=True)

    profiles = {"clinical": profile(clin), "aw_B_200hz": profile(awB), "aw_A_500hz": profile(awA)}

    def dist(a, b):
        return float(np.mean([abs(a[ax][0] - b[ax][0]) / (a[ax][1] + 1e-9) for ax in AXES]))

    print("\n===== CORRECTED PROFILES (all at common 100 Hz) =====", flush=True)
    print(f"  {'axis':11s} {'clinical':>10s} {'AW_B(200)':>10s} {'AW_A(500)':>10s}", flush=True)
    for ax in AXES:
        print(f"  {ax:11s} {profiles['clinical'][ax][0]:10.3f} {profiles['aw_B_200hz'][ax][0]:10.3f} "
              f"{profiles['aw_A_500hz'][ax][0]:10.3f}", flush=True)

    print("\n===== CORRECTED DISTANCES =====", flush=True)
    dB = dist(profiles["clinical"], profiles["aw_B_200hz"])
    dA = dist(profiles["clinical"], profiles["aw_A_500hz"])
    dAB = dist(profiles["aw_A_500hz"], profiles["aw_B_200hz"])
    print(f"  clinical -> AW_B(200Hz): {dB:.3f}", flush=True)
    print(f"  clinical -> AW_A(500Hz): {dA:.3f}", flush=True)
    print(f"  AW_A     -> AW_B        : {dAB:.3f}  (internal consistency of the two AW sources)", flush=True)

    # E36 retraction check: the old (wrong-fs) hf_energy gap
    print("\n===== E36 RETRACTION: HF energy, correct vs wrong fs =====", flush=True)
    print(f"  clinical hf_energy      : {profiles['clinical']['hf_energy'][0]:.3f}", flush=True)
    print(f"  AW_B hf CORRECT (200Hz) : {profiles['aw_B_200hz']['hf_energy'][0]:.3f}", flush=True)
    print(f"  (E36 WRONGLY reported AW hf ~0.16 by treating 200Hz as 500Hz)", flush=True)

    metrics = {
        "fs_common": FS, "n_clin": len(clin), "n_awB": len(awB), "n_awA": len(awA),
        "profiles": profiles,
        "distances": {"clinical_to_awB": dB, "clinical_to_awA": dA, "awA_to_awB": dAB},
        "bug": "E35/E36 treated the 200Hz wide file as 500Hz (2.5x frequency-axis error); HF/bandwidth-gap conclusion RETRACTED",
        "question": "with correct sampling rates, what/whether is the real clinical->AW gap?",
        "honesty": ["distribution study, no AUROC", "HOME eval-only",
                    "two AW sources at genuinely different native rates (200 vs 500 Hz)",
                    "common 100 Hz (Nyquist 50)", "AW_A only 20 recordings"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(profiles)
    print("\nDONE.", flush=True)


def _plot(profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    order = ["clinical", "aw_B_200hz", "aw_A_500hz"]
    cols = ["#4C72B0", "#333333", "#55A868"]
    x = np.arange(len(AXES)); w = 0.25
    for i, (n, c) in enumerate(zip(order, cols)):
        ax.bar(x + (i - 1) * w, [profiles[n][ax][0] for ax in AXES], w, label=n, color=c)
    ax.set_xticks(x); ax.set_xticklabels(AXES, fontsize=9, rotation=15)
    ax.set_ylabel("stat value (mean)")
    ax.set_title("E37: CORRECTED sampling-rate profiles (retracts E35/E36 bandwidth-gap)")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "corrected_profiles.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
