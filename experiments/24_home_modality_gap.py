"""Experiment 24 — Modality-gap meta-analysis from HOME's published baselines.

FULLY LICENSE-COMPLIANT: HOME is evaluation-only (no training/fine-tuning/domain-
adaptation on their data). This experiment does NONE of that — it only ANALYZES
HOME's own published baseline predictions to quantify the clinical->watch modality
gap on REAL Apple Watch, per clinical task.

HOME provides, for each of 9 tasks, predictions from two models on the same 1000
real Apple Watch ECGs:
  - "12-lead model": trained on Lead-I from resting 12-lead ECGs (THE clinical->
    watch transfer scenario this whole project studies)
  - "fine-tuning model": trained on Lead-I from the actual device (Apple Watch)

The AGREEMENT between these two on real Apple Watch data directly measures the
modality gap at the task level on the TRUE target device:
  - High agreement (12-lead ~ fine-tuned) => the clinical model already transfers
    well to watch for that task; the modality gap is small; fine-tuning barely helps.
  - Low agreement => fine-tuning substantially changes predictions => large
    modality gap => clinical-only transfer is insufficient for that task.

This is the REAL-DEVICE, task-level analog of our sim experiments (E1 axis
decomposition, E6b transfer). It tells us WHICH clinical tasks survive the Apple
Watch modality shift — grounded in real data, from the benchmark authors' own
validated models, with zero training on our part.

Metrics per task (12-lead vs fine-tuning predictions, 1000 real AW subjects):
  - Pearson r (linear agreement)
  - Spearman rho (rank agreement — matters for AUROC-style ranking tasks)
  - mean abs difference (normalized by prediction range)
  - for binary tasks (prob in [0,1]): agreement of thresholded decisions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "24_home_modality_gap"
RESULTS.mkdir(parents=True, exist_ok=True)

HOME_PRED = Path.home() / "data" / "HOME" / "data-for-predicting" / "baseline-prediction"

# task name -> (is_binary_prob, human label)
TASKS = {
    "1-Gender":        (True,  "Gender (binary)"),
    "2-Age":           (False, "Age (continuous)"),
    "3-Death":         (True,  "Death (binary)"),
    "4-Low_EF":        (True,  "Low EF (binary)"),
    "5-High_PASP":     (True,  "High PASP (binary)"),
    "6-High_LA":       (True,  "High LA (binary)"),
    "7-High_NT-proBNP":(True,  "High NT-proBNP (binary)"),
    "8-Low_Hb":        (True,  "Low Hb (binary)"),
    "9-Low_eGFR":      (True,  "Low eGFR (binary)"),
}


def load_pair(task, device="Apple"):
    lead12 = pd.read_csv(HOME_PRED / f"{task} 12-lead model ({device}).csv")
    ft = pd.read_csv(HOME_PRED / f"{task} fine-tuning model ({device}).csv")
    merged = lead12.merge(ft, on="UID", suffixes=("_12lead", "_ft"))
    return merged


def analyze(task, is_binary, device="Apple"):
    from scipy.stats import pearsonr, spearmanr
    m = load_pair(task, device)
    a = m["pred_12lead"].to_numpy()
    b = m["pred_ft"].to_numpy()
    r, _ = pearsonr(a, b)
    rho, _ = spearmanr(a, b)
    rng = max(a.max(), b.max()) - min(a.min(), b.min())
    mad = float(np.mean(np.abs(a - b)))
    mad_norm = float(mad / (rng + 1e-9))
    out = {
        "n": len(m), "pearson_r": float(r), "spearman_rho": float(rho),
        "mean_abs_diff": mad, "mad_normalized": mad_norm,
        "mean_12lead": float(a.mean()), "mean_ft": float(b.mean()),
    }
    if is_binary:
        # decision agreement at 0.5 threshold
        dec_a = (a >= 0.5).astype(int); dec_b = (b >= 0.5).astype(int)
        out["decision_agreement_@0.5"] = float(np.mean(dec_a == dec_b))
        out["positive_rate_12lead"] = float(dec_a.mean())
        out["positive_rate_ft"] = float(dec_b.mean())
    return out


def main():
    print("HOME modality-gap meta-analysis (Apple Watch, 12-lead vs fine-tuning) ...", flush=True)
    print("License-compliant: analyzing HOME's published predictions only, NO training.\n", flush=True)
    results = {}
    for task, (is_binary, label) in TASKS.items():
        try:
            r = analyze(task, is_binary, "Apple")
        except FileNotFoundError as e:
            print(f"  {task}: MISSING ({e})", flush=True); continue
        results[task] = {"label": label, "is_binary": is_binary, **r}
        extra = f" dec_agree={r.get('decision_agreement_@0.5', float('nan')):.3f}" if is_binary else ""
        print(f"  {label:28s} r={r['pearson_r']:.3f} rho={r['spearman_rho']:.3f} "
              f"mad_norm={r['mad_normalized']:.3f}{extra}", flush=True)

    # summary: rank tasks by transferability (high r = clinical model transfers well)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["spearman_rho"], reverse=True)
    print("\n=== Tasks ranked by clinical->watch transferability (Spearman rho) ===", flush=True)
    for task, r in ranked:
        verdict = "TRANSFERS WELL" if r["spearman_rho"] > 0.8 else ("MODERATE" if r["spearman_rho"] > 0.5 else "LARGE GAP")
        print(f"  {r['label']:28s} rho={r['spearman_rho']:.3f}  [{verdict}]", flush=True)

    rhos = [r["spearman_rho"] for r in results.values()]
    summary = {
        "device": "Apple Watch",
        "n_tasks": len(results),
        "per_task": results,
        "mean_spearman_rho": float(np.mean(rhos)),
        "median_spearman_rho": float(np.median(rhos)),
        "interpretation": {
            "high_rho": "clinical 12-lead model transfers well to real Apple Watch; small modality gap",
            "low_rho": "fine-tuning substantially changes predictions; large modality gap; clinical-only insufficient",
        },
        "honesty": [
            "This analyzes AGREEMENT between HOME's two models, NOT accuracy (labels withheld).",
            "High 12-lead/fine-tuned agreement does not prove either is CORRECT — only that fine-tuning changed little.",
            "A task where both models are equally wrong would show high agreement but poor real utility.",
            "License-compliant: no training/fine-tuning/domain-adaptation on HOME data; meta-analysis of published predictions only.",
        ],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = list(results.keys())
    labels = [results[t]["label"].replace(" (binary)", "").replace(" (continuous)", "") for t in tasks]
    rhos = [results[t]["spearman_rho"] for t in tasks]
    rs = [results[t]["pearson_r"] for t in tasks]
    order = np.argsort(rhos)[::-1]
    labels = [labels[i] for i in order]; rhos = [rhos[i] for i in order]; rs = [rs[i] for i in order]
    x = np.arange(len(tasks)); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#55A868" if v > 0.8 else ("#DD8452" if v > 0.5 else "#C44E52") for v in rhos]
    b1 = ax.bar(x - w/2, rhos, w, label="Spearman rho (rank agreement)", color=colors)
    b2 = ax.bar(x + w/2, rs, w, label="Pearson r (linear agreement)", color="#4C72B0", alpha=0.6)
    ax.axhline(0.8, color="green", ls=":", lw=1, label="transfers well (>0.8)")
    ax.axhline(0.5, color="orange", ls=":", lw=1, label="large gap (<0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
    ax.set_ylabel("12-lead vs fine-tuning agreement on REAL Apple Watch")
    ax.set_ylim(0, 1.05)
    ax.set_title("E24: clinical->watch modality gap per task (HOME real Apple Watch baselines)")
    ax.legend(fontsize=8, loc="lower left")
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{b.get_height():.2f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "modality_gap_per_task.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
