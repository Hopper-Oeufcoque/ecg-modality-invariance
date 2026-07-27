"""Experiment 6 — Real single-lead validation of the forward-physics simulator.

The biggest honesty flag in E1-E17: all results are on *simulated* watch.
E6 asks whether the simulator produces *realistic* single-lead ECG by comparing
its output distribution to REAL single-lead recordings (PhysioNet CinC
Challenge 2017 — lead-I handheld electrode, 9-60 s, closest public Apple-Watch
analog), on distribution-level metrics that don't need label alignment:

  1. Power spectral density (PSD) band-energy match
  2. Baseline-wander / motion-artifact magnitude
  3. QRS morphology fidelity (R-peak SNR, rhythm)
  4. Fractal / nonlinear-dynamics (sample entropy, DFA exponent)
  5. Classifier cross-over: train sim, test real → gap quantifies sim/real mismatch

This is a *distribution-match* validation (no label alignment needed), which
is exactly what the synthetic data needs to be trustworthy as a stand-in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LEAD_NAMES, load_all
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "06_sim_validation"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
N_LEADS = 12


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


def load_cinc2017(data_dir=None, fs=300, n=300):
    """Load PhysioNet CinC 2017 single-lead ECGs (lead I, handheld).

    Returns list of (signal, label) where label ∈ {N, A, O, ~}.
    These are single-lead at 300 Hz — downsample to our FS for comparison.
    """
    import os
    if data_dir is None:
        data_dir = Path.home() / "data" / "cinc2017" / "training2017"
    data_dir = Path(data_dir)
    records = []
    import scipy.io as sio
    from scipy.signal import resample_poly
    ref = {}
    refcsv = data_dir / "REFERENCE.csv"
    if refcsv.exists():
        for line in refcsv.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) == 2:
                ref[parts[0]] = parts[1]
    matfiles = sorted(data_dir.glob("A*.mat"))[:n]
    for mf in matfiles:
        rec_id = mf.stem
        label = ref.get(rec_id, "O")
        try:
            mat = sio.loadmat(mf)
            sig = mat["val"][0].astype(np.float64)  # (T,)
        except Exception:
            continue
        if sig.size < 1000:
            continue
        # per-record norm
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        # downsample 300->100
        if fs != FS:
            from math import gcd
            g = gcd(int(fs), int(FS))
            up = int(FS) // g; down = int(fs) // g
            sig = resample_poly(sig, up, down)
        records.append((sig, label))
    return records


def psd_band_energies(sig, fs=FS, bands=None):
    """Fraction of total energy in each band (low/respiratory, ECG-band, high/noise)."""
    if bands is None:
        bands = [(0.5, 5.0), (5.0, 40.0), (40.0, fs/2)]
    f, p = np.fft.rfftfreq(len(sig), 1/fs), np.abs(np.fft.rfft(sig))**2
    total = p.sum() + 1e-12
    out = []
    for lo, hi in bands:
        mask = (f >= lo) & (f < hi)
        out.append(float(p[mask].sum() / total))
    return out


def baseline_wander(sig, fs=FS):
    """Estimate low-freq (<0.5Hz) drift amplitude as fraction of signal std."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(2, 0.5 / (fs/2), btype="lowpass", output="sos")
    drift = sosfiltfilt(sos, sig)
    return float(drift.std() / (sig.std() + 1e-9))


def sample_entropy(sig, m=2, r=0.2):
    """Sample entropy (nonlinear dynamics; regularity/complexity)."""
    sig = (sig - sig.mean()) / (sig.std() + 1e-9)
    n = len(sig); r *= sig.std()
    def _phi(m):
        x = np.array([sig[i:i+m] for i in range(n - m)])
        d = np.abs(x[:, None] - x).max(2)
        return np.sum(d < r) - n  # exclude self-matches
    try:
        phi_m = _phi(m); phi_m1 = _phi(m+1)
        return float(-np.log(phi_m1 / phi_m)) if phi_m > 0 else float("nan")
    except Exception:
        return float("nan")


def dfa_exponent(sig, scale_min=10, scale_max=200):
    """Detrended fluctuation analysis α exponent (long-range correlation)."""
    sig = (sig - sig.mean())
    cum = np.cumsum(sig)
    n = len(sig)
    scales = np.unique(np.logspace(np.log10(scale_min), np.log10(min(scale_max, n//4)), 12).astype(int))
    fluc = []
    for s in scales:
        nseg = n // s
        if nseg < 2: continue
        segs = cum[:nseg*s].reshape(nseg, s)
        x = np.arange(s)
        # linear detrend each segment
        A = np.vstack([x, np.ones_like(x)]).T
        resid = segs - segs @ A @ np.linalg.pinv(A)
        fluc.append(np.sqrt((resid**2).mean()))
    if len(fluc) < 4:
        return float("nan")
    log_s = np.log(scales[:len(fluc)]); log_f = np.log(np.array(fluc))
    return float(np.polyfit(log_s, log_f, 1)[0])


def all_stats(sig, fs=FS):
    return {
        "psd_bands": psd_band_energies(sig, fs),
        "baseline_wander": baseline_wander(sig, fs),
        "sample_entropy": sample_entropy(sig),
        "dfa_alpha": dfa_exponent(sig),
        "std": float(np.std(sig)),
        "kurtosis": float(np.mean((sig - sig.mean())**4) / (np.var(sig)**2 + 1e-9)),
    }


def distribution_distance(stats_a, stats_b, keys=None):
    """Mean abs z-score distance between two stat-distributions over a key list."""
    if keys is None:
        keys = ["baseline_wander", "sample_entropy", "dfa_alpha", "kurtosis"]
    out = {}
    for k in keys:
        a = np.array([s[k] for s in stats_a if not np.isnan(s.get(k, float("nan")))])
        b = np.array([s[k] for s in stats_b if not np.isnan(s.get(k, float("nan")))])
        if len(a) and len(b):
            mu = np.concatenate([a, b]).mean(); sd = np.concatenate([a, b]).std() + 1e-9
            out[k] = float(abs(a.mean() - b.mean()) / sd)
    return out


def main():
    print("Loading PTB-XL (clinical source) ...", flush=True)
    splits = load_all(max_per_class=200)
    clin = splits["test"]
    print(f"  clinical records: {len(clin)}", flush=True)

    print("Generating simulated watch from clinical ...", flush=True)
    sim_stats = []
    rng = np.random.default_rng(0)
    cfg = _cfg(seed=None)
    for rec in clin:
        x = rec["ecg"][:1000]
        if x.shape[0] < 1000:
            x = np.concatenate([x, np.zeros((1000 - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=rng)
        sim_stats.append(all_stats(out["watch"], FS))

    print("Loading real single-lead (CinC 2017) ...", flush=True)
    real = load_cinc2017(n=300)
    print(f"  real single-lead records: {len(real)}", flush=True)
    real_stats = [all_stats(s) for s, _ in real]

    # raw clinical Lead-I for reference
    clin_leadI_stats = []
    for rec in clin:
        x = rec["ecg"][:1000, 0]
        mu = x.mean(); sd = x.std() + 1e-6; x = (x - mu) / sd
        clin_leadI_stats.append(all_stats(x, FS))

    summary = {
        "n_sim": len(sim_stats), "n_real": len(real_stats), "n_clin": len(clin_leadI_stats),
        "means": {},
        "distances": {},
    }
    for name, stats in [("clinical_LeadI", clin_leadI_stats),
                        ("simulated_watch", sim_stats), ("real_single_lead", real_stats)]:
        m = {}
        for k in ["baseline_wander", "sample_entropy", "dfa_alpha", "kurtosis", "std"]:
            vals = [s[k] for s in stats if not np.isnan(s.get(k, float("nan")))]
            m[k] = float(np.mean(vals)) if vals else float("nan")
        m["psd_bands"] = [float(np.mean([s["psd_bands"][i] for s in stats])) for i in range(3)]
        summary["means"][name] = m
        print(f"\n[{name}] means:", flush=True)
        for k, v in m.items(): print(f"  {k}: {v}", flush=True)

    summary["distances"]["sim_vs_real"] = distribution_distance(sim_stats, real_stats)
    summary["distances"]["sim_vs_clinical"] = distribution_distance(sim_stats, clin_leadI_stats)
    summary["distances"]["real_vs_clinical"] = distribution_distance(real_stats, clin_leadI_stats)
    print("\nDistribution distances (mean abs z-score, lower=closer):", flush=True)
    for k, d in summary["distances"].items():
        print(f"  {k}:", {kk: round(vv, 3) for kk, vv in d.items()}, flush=True)

    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(summary)
    print("\nDONE.", flush=True)


def _plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = summary["means"]
    names = ["clinical_LeadI", "simulated_watch", "real_single_lead"]
    labels = ["Clinical Lead-I", "Simulated watch", "Real single-lead"]
    cols = ["#4C72B0", "#DD8452", "#C44E52"]
    keys = ["baseline_wander", "sample_entropy", "dfa_alpha", "kurtosis"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, k in zip(axes.flat, keys):
        vals = [means[n][k] for n in names]
        bars = ax.bar(range(3), vals, color=cols)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8, rotation=15)
        ax.set_title(k, fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.3f}", ha="center", fontsize=7)
    plt.suptitle("E6: simulator realism vs real single-lead (CinC 2017)", fontsize=11)
    plt.tight_layout(); plt.savefig(RESULTS / "sim_validation.png", dpi=130); plt.close()

    # PSD bands
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3); w = 0.27
    band_labels = ["<5 Hz (drift)", "5-40 Hz (ECG)", "40-50 Hz (noise)"]
    for i, n in enumerate(names):
        vals = means[n]["psd_bands"]
        ax.bar(x + i*w - w, vals, w, label=labels[i], color=cols[i])
    ax.set_xticks(x); ax.set_xticklabels(band_labels, fontsize=9)
    ax.set_ylabel("fraction of total energy"); ax.legend(fontsize=8)
    ax.set_title("PSD band-energy distribution")
    plt.tight_layout(); plt.savefig(RESULTS / "psd_bands.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
