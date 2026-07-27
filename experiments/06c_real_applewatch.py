"""Experiment 6c — Simulator realism vs REAL Apple Watch (HOME dataset).

THE DEFINITIVE REALISM TEST. E6 used CinC 2017 (handheld lead-I, finger contact —
cleaner than wrist dry-electrode) as the real reference and found the simulator
over-degrades. But CinC is a PROXY, not the true target. E6c uses REAL Apple Watch
wrist dry-electrode ECGs (HOME benchmark, 1000 subjects, 200 Hz, Lead I) — the
actual target device.

Two outcomes:
  - If sim still over-degrades vs REAL Apple Watch → E6's finding holds on the true
    target; the simulator is fundamentally miscalibrated for wrist dry-electrode.
  - If sim matches real Apple Watch BETTER than it matched CinC → the simulator's
    aggressive noise was justified for wrist dry-electrode (which IS noisier than
    handheld); CinC was the wrong reference, and the sim is more defensible.

Also computes the same triangle (sim vs real-Apple-Watch vs clinical Lead-I) and
the key stats (kurtosis, entropy, baseline_wander, PSD) at each. The kurtosis axis
is the sharpest discriminator (E6: sim 4.8 vs CinC 17.7; what is it vs real AW?).

HOME data: data-for-predicting/Apple_Watch_waveform.csv — 1000 cols (subjects),
6000 rows (samples @ 200 Hz = 30 s), unit 0.01 mV, Lead I. Evaluation-only:
waveforms used for distribution analysis (no labels, no training) — compliant
with the benchmark's evaluation-only license.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import LEAD_NAMES, load_all
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

_spec6 = importlib.util.spec_from_file_location(
    "e6", Path(__file__).resolve().parents[1] / "experiments" / "06_sim_validation.py")
e6 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(e6)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "06c_real_applewatch"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0  # target rate (resample HOME 200->100 for comparison to sim@100)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


def load_home_applewatch(path=None, n=300, fs_in=200):
    """Load real Apple Watch single-lead ECGs from HOME benchmark.

    Returns list of (signal @ FS Hz, None). Each col = one subject, 6000 rows @ 200 Hz.
    """
    if path is None:
        path = Path.home() / "data" / "HOME" / "data-for-predicting" / "Apple_Watch_waveform.csv"
    df = pd.read_csv(path)  # 6000 rows x 1000 cols
    cols = list(df.columns)
    rng = np.random.default_rng(0)
    rng.shuffle(cols)
    records = []
    from math import gcd
    g = gcd(int(fs_in), int(FS)); up = int(FS) // g; down = int(fs_in) // g
    for col in cols[:n]:
        sig = df[col].to_numpy(dtype=np.float64)  # 6000 samples @ 200 Hz, unit 0.01 mV
        # convert to mV then per-record normalize
        sig = sig * 0.01
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        # resample 200 -> 100
        sig = resample_poly(sig, up, down)
        # take a 1000-sample (10s) window for consistency with PTB-XL analysis
        if sig.size < 1000:
            continue
        records.append(sig[:1000])
    return records


def main():
    print("Loading PTB-XL clinical (max_per_class=200) ...", flush=True)
    splits = load_all(max_per_class=200)
    clin = splits["test"]
    print(f"  clinical records: {len(clin)}", flush=True)

    print("Generating simulated watch from clinical (default + recalibrated m=0.05) ...", flush=True)
    rng = np.random.default_rng(0)
    sim_stats_default = []; sim_stats_recalib = []
    cfg_default = _cfg(seed=0)
    cfg_recalib = _cfg(baseline_wander_sigma=0.05*0.05, motion_sigma=0.10*0.05,
                       emg_sigma=0.05*0.05, seed=0)  # m=0.05 from E22
    for rec in clin:
        x = rec["ecg"][:1000]
        if x.shape[0] < 1000:
            x = np.concatenate([x, np.zeros((1000 - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        out_d = simulate_watch(x, FS, cfg_default, LEAD_NAMES, rng=np.random.default_rng(0))
        sim_stats_default.append(e6.all_stats(out_d["watch"], FS))
        out_r = simulate_watch(x, FS, cfg_recalib, LEAD_NAMES, rng=np.random.default_rng(0))
        sim_stats_recalib.append(e6.all_stats(out_r["watch"], FS))

    print("Loading real CinC 2017 (n=300) ...", flush=True)
    real_cinc = e6.load_cinc2017(n=300)
    cinc_stats = [e6.all_stats(s) for s, _ in real_cinc]

    print("Loading REAL Apple Watch (HOME, n=300) ...", flush=True)
    aw_sigs = load_home_applewatch(n=300)
    aw_stats = [e6.all_stats(s) for s in aw_sigs]
    print(f"  real Apple Watch records: {len(aw_stats)}", flush=True)

    # raw clinical Lead-I for reference
    clin_leadI_stats = []
    for rec in clin:
        x = rec["ecg"][:1000, 0]
        mu = x.mean(); sd = x.std() + 1e-6; x = (x - mu) / sd
        clin_leadI_stats.append(e6.all_stats(x, FS))

    # means per distribution
    dists = {
        "sim_default": sim_stats_default,
        "sim_recalib_m0.05": sim_stats_recalib,
        "clinical_LeadI": clin_leadI_stats,
        "real_CinC_handheld": cinc_stats,
        "real_AppleWatch": aw_stats,
    }
    means = {}
    for name, stats in dists.items():
        m = {k: float(np.mean([s[k] for s in stats if not np.isnan(s.get(k, float("nan")))]))
             for k in ["baseline_wander", "sample_entropy", "kurtosis", "dfa_alpha"]}
        m["psd_bands"] = [float(np.mean([s["psd_bands"][i] for s in stats])) for i in range(3)]
        means[name] = m
        print(f"\n[{name}] kurt={m['kurtosis']:.2f} entropy={m['sample_entropy']:.3f} "
              f"bw={m['baseline_wander']:.3f} dfa={m['dfa_alpha']:.3f} psd={[round(p,3) for p in m['psd_bands']]}", flush=True)

    # pairwise distances to REAL Apple Watch (the true target)
    print("\n=== Distances to REAL Apple Watch (mean abs z-score, lower=closer) ===", flush=True)
    dist_to_aw = {}
    for name in ["sim_default", "sim_recalib_m0.05", "clinical_LeadI", "real_CinC_handheld"]:
        d = e6.distribution_distance(dists[name], aw_stats)
        md = float(np.mean(list(d.values())))
        dist_to_aw[name] = {"per_metric": d, "mean": md}
        print(f"  {name} -> AppleWatch: {md:.3f}  {dict((k, round(v,3)) for k,v in d.items())}", flush=True)

    # also sim_default vs each real reference
    print("\n=== sim_default vs each real reference ===", flush=True)
    for ref_name in ["real_CinC_handheld", "real_AppleWatch"]:
        d = e6.distribution_distance(sim_stats_default, dists[ref_name])
        print(f"  sim_default -> {ref_name}: {float(np.mean(list(d.values()))):.3f}", flush=True)

    summary = {
        "n_sim": len(sim_stats_default), "n_cinc": len(cinc_stats), "n_applewatch": len(aw_stats),
        "means": means,
        "distances_to_real_AppleWatch": dist_to_aw,
        "headline": "sim vs REAL Apple Watch (true target) — does over-degradation hold on wrist dry-electrode?",
        "fs": FS,
        "honesty": ["HOME is evaluation-only — waveforms used for distribution analysis only, no labels, no training (license-compliant)",
                    "single seed, 300 records per distribution",
                    "HOME Apple Watch @ 200 Hz resampled to 100 Hz for comparison",
                    "HOME subjects are a clinical cohort linked to health records — may differ from general Apple Watch users"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(means, dist_to_aw)
    print("\nDONE.", flush=True)


def _plot(means, dist_to_aw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["clinical_LeadI", "sim_default", "sim_recalib_m0.05", "real_CinC_handheld", "real_AppleWatch"]
    labels = ["Clinical Lead-I", "Sim (default)", "Sim (recalib m=0.05)", "Real CinC (handheld)", "REAL Apple Watch"]
    cols = ["#4C72B0", "#DD8452", "#FFA500", "#8172B2", "#C44E52"]
    keys = ["baseline_wander", "sample_entropy", "kurtosis", "dfa_alpha"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, k in zip(axes.flat, keys):
        vals = [means[n][k] for n in names]
        bars = ax.bar(range(5), vals, color=cols)
        ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=7, rotation=20)
        ax.set_title(k, fontsize=11)
        for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", fontsize=7)
    plt.suptitle("E6c: simulator vs REAL Apple Watch (HOME) — the true wrist dry-electrode target", fontsize=12)
    plt.tight_layout(); plt.savefig(RESULTS / "real_applewatch.png", dpi=130); plt.close()
    # distances to Apple Watch
    fig, ax = plt.subplots(figsize=(9, 5))
    dn = list(dist_to_aw.keys())
    dlabels = ["Sim default", "Sim recalib m=0.05", "Clinical Lead-I", "Real CinC (handheld)"]
    dvals = [dist_to_aw[n]["mean"] for n in dn]
    bars = ax.bar(range(4), dvals, color=["#DD8452", "#FFA500", "#4C72B0", "#8172B2"])
    ax.set_xticks(range(4)); ax.set_xticklabels(dlabels, fontsize=9, rotation=15)
    ax.set_ylabel("mean abs z-score distance to REAL Apple Watch (lower=closer)"); ax.set_ylim(0, max(dvals)*1.2)
    ax.set_title("Which reference is closest to REAL Apple Watch?")
    for b, v in zip(bars, dvals): ax.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "dist_to_applewatch.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
