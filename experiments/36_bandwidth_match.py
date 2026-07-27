"""Experiment 36 — Morphology-preserving bandwidth match to REAL Apple Watch.

E35 (n=1000) showed the real clinical->AW gap is a BANDWIDTH/FILTERING mismatch:
clinical PTB-XL is over-filtered (~0.5-40 Hz), real AW keeps high-freq content
(hf_energy 0.165 vs clinical 0.019). E35 also showed you CANNOT close it by
injecting broadband noise (aw-calibrated DR hit the HF stat but destroyed QRS,
corr 0.594).

E36 does the mechanistically-correct fix: a ZERO-PHASE SPECTRAL TRANSFER (the
E33 build_transfer_function idea — magnitude reshaping, phase preserved) that
moves clinical's frequency envelope toward the REAL AW envelope WITHOUT touching
phase (so QRS morphology/timing survive). Measured at NATIVE 500 Hz so we don't
resample away the HF band we're matching.

LICENSE-COMPLIANT: measures the real-AW magnitude spectrum for the transfer
target; NO training/fine-tuning on HOME (a transfer function is a signal-stats
object, not a trained model on their labels).

Compares (profile + morphology, at 500 Hz):
  clinical (raw)
  spectral-transfer -> AW      (magnitude reshaping toward real AW, phase kept)
  light DR
  broadband-noise (E35 aw-cal) reference (the morphology-wrecking one)

Question: can a phase-preserving filter match real AW's HF content while keeping
QRS corr > 0.95 (label valid)? That would be the correct real-AW zero-shot recipe.

Honesty: stats + morphology study (no AUROC — needs labels/training on HOME which
we don't do). Real AW HF beyond 100 Hz still lost (watch native ~512 Hz; HOME is
500 Hz, we keep full band). Distance dominated by HF axis — read per-axis + QRS.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "36_bandwidth_match"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 500.0          # NATIVE — keep the HF band
SIGLEN = 5000       # 10 s at 500 Hz
AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]


def stats_at(sig, fs=FS):
    from scipy import stats as sstats
    x = np.asarray(sig, float); x = (x - x.mean()) / (x.std() + 1e-9)
    f = np.fft.rfftfreq(len(x), 1 / fs); P = np.abs(np.fft.rfft(x)) ** 2; T = P.sum() + 1e-9
    def band(lo, hi): return P[(f >= lo) & (f < hi)].sum() / T
    return {"kurtosis": float(sstats.kurtosis(x)), "bw_energy": float(band(0, 1)),
            "qrs_energy": float(band(5, 15)), "hf_energy": float(band(30, 50)),
            "mid_energy": float(band(1, 5))}


def profile(sigs):
    ks = [stats_at(s) for s in sigs]
    return {k: (float(np.mean([d[k] for d in ks])), float(np.std([d[k] for d in ks]))) for k in AXES}


def load_home_500(max_patients=1000):
    path = Path.home() / "data" / "HOME" / "data-for-predicting" / "Apple_Watch_waveform.csv"
    with open(path) as f:
        r = csv.reader(f); next(r)
        cols = None
        for row in r:
            if cols is None: cols = [[] for _ in row]
            for i, v in enumerate(row):
                if i < len(cols):
                    try: cols[i].append(float(v))
                    except: cols[i].append(np.nan)
    out = []
    for c in cols[:max_patients]:
        v = np.asarray(c, float); v = v[np.isfinite(v)]
        if v.size < SIGLEN: continue
        seg = v[:SIGLEN]; out.append(((seg - seg.mean()) / (seg.std() + 1e-9)).astype(np.float32))
    return out


def load_clinical_500(max_per_class=250):
    """PTB-XL 500 Hz Lead-I (filename_hr)."""
    import pandas as pd, ast, wfdb
    d = Path.home() / "data" / "ptbxl"
    df = pd.read_csv(d / "ptbxl_database.csv")
    def has(c, s):
        try: return c in ast.literal_eval(s)
        except: return False
    norm = df[df["scp_codes"].apply(lambda s: has("NORM", s))].head(max_per_class)
    out = []
    for _, row in norm.iterrows():
        try: sig, _ = wfdb.rdsamp(str(d / row["filename_hr"]))
        except: continue
        x = sig[:SIGLEN, 0].astype(np.float64)
        if x.size < SIGLEN: continue
        out.append(((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32))
    return out


def build_transfer(clin_sigs, aw_sigs, smooth=15, clip=(0.3, 8.0)):
    def mean_mag(sigs):
        acc = None
        for s in sigs:
            s = np.asarray(s, float)[:SIGLEN]
            if s.size < SIGLEN: s = np.concatenate([s, np.zeros(SIGLEN - s.size)])
            s = (s - s.mean()) / (s.std() + 1e-9)
            m = np.abs(np.fft.rfft(s))
            acc = m if acc is None else acc + m
        return acc / len(sigs)
    Hc = mean_mag(clin_sigs); Ha = mean_mag(aw_sigs)
    H = Ha / (Hc + 1e-9)
    k = np.ones(smooth) / smooth
    H = np.convolve(H, k, mode="same")
    return np.clip(H, clip[0], clip[1])


def apply_transfer(x, H):
    x = np.asarray(x, float)[:SIGLEN]
    if x.size < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.size)])
    x = (x - x.mean()) / (x.std() + 1e-9)
    X = np.fft.rfft(x)
    Y = np.abs(X) * H[:X.size] * np.exp(1j * np.angle(X))  # magnitude reshape, PHASE KEPT
    y = np.fft.irfft(Y, n=SIGLEN)
    return ((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)


def qrs_morph(o, a):
    o = np.asarray(o, float)[:SIGLEN]; a = np.asarray(a, float)[:SIGLEN]
    b, c = butter(2, 1.0 / (FS / 2), "high")
    oh = filtfilt(b, c, o); ah = filtfilt(b, c, a)
    qc = float(np.corrcoef(oh, ah)[0, 1])
    po, _ = find_peaks(oh, height=np.std(oh) * 1.5, distance=100)
    pa, _ = find_peaks(ah, height=np.std(ah) * 1.5, distance=100)
    rp = float(np.mean([np.min(np.abs(pa - p)) <= 25 for p in po])) if len(po) and len(pa) else (0.0 if len(po) else float("nan"))
    return qc, rp


def main():
    print("Loading clinical PTB-XL 500 Hz ...", flush=True)
    clin = load_clinical_500(max_per_class=250)
    print(f"  clinical n={len(clin)}", flush=True)
    print("Loading REAL AW 500 Hz (HOME, eval-only) ...", flush=True)
    aw = load_home_500(max_patients=1000)
    print(f"  real AW n={len(aw)}", flush=True)

    # split AW: half to BUILD the transfer (stats only), half to profile against (no leakage)
    half = len(aw) // 2
    aw_ref = aw[:half]; aw_prof = aw[half:]

    H = build_transfer(clin, aw_ref)
    xfer = [apply_transfer(c, H) for c in clin]

    # light DR + broadband ref (import from module)
    from src.aw_generator import StochasticAWAugmenter
    light = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=0.5, seed=0)
    light_sigs = [light.generate(c) for c in clin]

    profiles = {
        "clinical": profile(clin),
        "real_aw": profile(aw_prof),
        "spectral_transfer": profile(xfer),
        "light_DR": profile(light_sigs),
    }

    def dist(a, b):
        return float(np.mean([abs(a[ax][0] - b[ax][0]) / (a[ax][1] + 1e-9) for ax in AXES]))

    # morphology
    morph = {}
    for name, sset_ in [("spectral_transfer", xfer), ("light_DR", light_sigs)]:
        qs, rs = [], []
        for o, a in zip(clin[:60], sset_[:60]):
            q, rp = qrs_morph(o, a)
            if np.isfinite(q): qs.append(q)
            if np.isfinite(rp): rs.append(rp)
        morph[name] = {"qrs_corr": float(np.mean(qs)), "rpeak_match": float(np.mean(rs))}

    print("\n===== PROFILES at 500 Hz (mean per axis) =====", flush=True)
    print(f"  {'axis':11s} {'clinical':>9s} {'REAL_AW':>9s} {'xfer':>9s} {'light':>9s}", flush=True)
    for ax in AXES:
        print(f"  {ax:11s} {profiles['clinical'][ax][0]:9.3f} {profiles['real_aw'][ax][0]:9.3f} "
              f"{profiles['spectral_transfer'][ax][0]:9.3f} {profiles['light_DR'][ax][0]:9.3f}", flush=True)

    print("\n===== DISTANCE TO REAL AW (per-axis std normalized) =====", flush=True)
    print(f"  clinical          -> AW: {dist(profiles['clinical'], profiles['real_aw']):.3f}", flush=True)
    print(f"  spectral_transfer -> AW: {dist(profiles['spectral_transfer'], profiles['real_aw']):.3f}", flush=True)
    print(f"  light_DR          -> AW: {dist(profiles['light_DR'], profiles['real_aw']):.3f}", flush=True)

    print("\n===== MORPHOLOGY (QRS-band corr / R-peak) =====", flush=True)
    for k, v in morph.items():
        print(f"  {k:18s} QRS {v['qrs_corr']:.3f}  R-peak {v['rpeak_match']:.3f}", flush=True)

    hf_clin = profiles["clinical"]["hf_energy"][0]; hf_aw = profiles["real_aw"]["hf_energy"][0]
    hf_xfer = profiles["spectral_transfer"]["hf_energy"][0]
    hf_closed = (hf_xfer - hf_clin) / (hf_aw - hf_clin) if abs(hf_aw - hf_clin) > 1e-6 else 0
    print(f"\n  HF-gap closed by spectral transfer: {100*hf_closed:.0f}%  (QRS corr {morph['spectral_transfer']['qrs_corr']:.3f})", flush=True)

    metrics = {
        "fs_native": FS, "n_clinical": len(clin), "n_aw": len(aw),
        "profiles": profiles,
        "distances_to_aw": {k: dist(profiles[k], profiles["real_aw"]) for k in ["clinical", "spectral_transfer", "light_DR"]},
        "morphology": morph, "hf_gap_closed_frac": hf_closed,
        "question": "can a phase-preserving spectral transfer match real-AW HF content while keeping QRS morphology?",
        "honesty": ["500 Hz native", "stats+morphology (no AUROC)", "HOME eval-only, transfer built on held-out AW half",
                    "distance dominated by HF axis — read per-axis + QRS", "HF>250Hz beyond Nyquist not modeled"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(profiles, H, morph)
    print("\nDONE.", flush=True)


def _plot(profiles, H, morph):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    order = ["clinical", "real_aw", "spectral_transfer", "light_DR"]
    cols = ["#4C72B0", "#333333", "#55A868", "#8172B2"]
    x = np.arange(len(AXES)); w = 0.2
    for i, (n, c) in enumerate(zip(order, cols)):
        ax1.bar(x + (i - 1.5) * w, [profiles[n][ax][0] for ax in AXES], w, label=n, color=c)
    ax1.set_xticks(x); ax1.set_xticklabels(AXES, fontsize=8, rotation=20)
    ax1.set_ylabel("stat value"); ax1.set_title("E36: profiles at 500Hz — spectral transfer vs real AW")
    ax1.legend(fontsize=8)
    freqs = np.fft.rfftfreq(SIGLEN, 1 / FS)
    ax2.plot(freqs, H, color="#55A868"); ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("gain H(f)")
    ax2.set_title(f"Learned clinical->AW transfer\n(QRS corr {morph['spectral_transfer']['qrs_corr']:.3f})")
    ax2.set_xlim(0, 60)
    plt.tight_layout(); plt.savefig(RESULTS / "bandwidth_match.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
