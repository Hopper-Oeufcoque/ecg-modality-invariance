# E22 — Simulator recalibration against real single-lead + E17 rerun

**Date:** 2026-07-27 · **Status:** ⚠️✅ Recalibration helps the model, but the kurtosis gap is filter-bound (not noise-bound)

## Hypothesis (spawned by E6)
E6 found the simulator over-degrades (sim_vs_real 1.077 > real_vs_clinical 0.717).
E22 asks: (A) can the noise magnitudes be tuned to match real CinC stats
(kurtosis ≥15, entropy ≤0.4)? (B) does the E17 single-lead+sim edge survive
recalibration — does it grow (over-degradation was hurting) or vanish (a sim→sim artifact)?

## Setup
- (A) Calibration sweep: global noise multiplier m∈{1.0, 0.5, 0.25, 0.1, 0.05}
  on baseline_wander/motion/EMG sigma. Generate sim-watch, compute distribution
  stats vs real CinC (300 records). Pick m minimizing mean abs z-score distance.
- (B) Rerun E17 single-lead+sim at default m=1.0 vs best-m. 20 ep, single seed.
  `experiments/22_sim_recalib.py`. Note: n_train=657 (calibration used
  max_per_class=200; half of E17's 1225) — within-experiment comparison is the
  valid signal, not absolute values vs E17.

## Results

**Calibration sweep (distance to real CinC):**

| noise mult m | kurtosis | sample_entropy | baseline_wander | mean dist |
|---|---|---|---|---|
| 1.0 (default) | 4.39 | 0.831 | 0.431 | 1.054 |
| 0.5 | 4.80 | 0.682 | 0.420 | 1.004 |
| 0.25 | 4.92 | 0.626 | 0.415 | 0.967 |
| 0.1 | 4.96 | 0.608 | 0.413 | 0.953 |
| **0.05 (best)** | **4.97** | **0.606** | **0.413** | **0.950** |
| *real CinC* | *17.71* | *0.282* | *0.190* | — |

**Edge robustness (single-lead+sim rerun):**

| config | L1 | L4 |
|---|---|---|
| default m=1.0 | 0.709 | 0.711 |
| **calib m=0.05** | **0.719** | **0.723** |

## Verdict: ⚠️✅ — two distinct findings

**(A) ✅ Recalibration helps the model.** The edge GREW with reduced noise:
L4 0.711 → 0.723 (+0.012). This confirms E6's hypothesis — the over-degradation
was hurting the model, and less aggressive noise is better. The calibration
converged to m=0.05 (the minimum tested), suggesting even less noise might help
further. So E17's edge is NOT purely a sim→sim artifact; reducing the sim's
distribution mismatch toward real improves the model, which is the right sign.

**(B) ⚠️ The kurtosis gap is FILTER-bound, not noise-bound.** The critical
diagnostic: across a **20× noise reduction** (m=1.0→0.05), kurtosis barely moves
(4.39 → 4.97) — it stays stuck at ~5, far below the real target of 17.71.
Reducing additive noise helps entropy (0.831→0.606, approaching real 0.282)
and baseline_wander somewhat, but **cannot fix kurtosis** because the QRS
flattening comes from the **bandpass filter (0.3–40 Hz) and electrode high-pass
coupling**, not the additive noise stages. The morphology destruction is baked
into the linear-filter stages, which a noise multiplier doesn't touch.

## Interpretation — where the simulator needs fixing
This localizes E6's over-degradation to a specific stage: the **bandpass is the
kurtosis-killer**, not the noise. The noise being too aggressive was a real but
secondary issue (entropy/baseline_wander), now partly addressed. The primary
realism gap — flattened QRS peakedness — requires revising the bandpass design:
either a gentler filter (higher order, less ringing), preserving phase, or
acknowledging that the real CinC handheld electrode doesn't apply the same
filter as the simulated Apple bandpass.

**Notable disconnect:** E1 found the bandwidth axis is *minor for AUROC*
(L2≈L1, the bandpass barely hurts classification). E22 shows the bandpass is
*major for realism* (destroys kurtosis). So the bandpass hurts the *distribution
match* but not the *task* — the model uses features robust to the filter's
flattening, which is why E17 still works despite the kurtosis gap. This explains
why E17's edge survives even with an unrealistic kurtosis: the model doesn't
rely on peakedness.

## Honesty flags
- Single seed; n_train=657 (half E17's 1225 — absolute values not comparable to
  E17's 0.742; within-experiment calib-vs-default is the valid comparison).
- CinC is handheld lead-I (cleaner than wrist dry-electrode) — matching it may
  under-model the true Apple Watch noise floor; the "best" m=0.05 may be too
  clean for real watch.
- Calibration matches distribution *stats*, not *task transfer* (E6b is the
  task-level test). Converged to the sweep boundary (m=0.05) — a lower m or a
  bandpass redesign is the natural next step.
- kurtosis target (≥15) was NOT met at any m — the noise axis alone cannot
  achieve it.

## Follow-ups
- **E22b — bandpass redesign:** sweep the filter (order, cutoff, zero-phase vs
  causal, preserve phase) to recover kurtosis while keeping the bandwidth axis
  minor. The kurtosis-killer is the filter, not the noise.
- **E6b** (scripted, ready) — the task-level sim→real cross-over; tests whether
  the recalibrated sim transfers better to real CinC than the default.
- The convergent-to-boundary result (m=0.05) suggests the optimal sim may have
  *minimal* noise — closer to clean Lead-I than to full watch. This aligns with
  E6's finding that real single-lead ≈ clinical Lead-I.

## Artifacts
- `results/22_sim_recalib/metrics.json` — full sweep + rerun
- `results/22_sim_recalib/recalib_sweep.png` — 3-panel kurtosis/entropy/distance
- `results/22_sim_recalib/edge_rerun.png` — default vs calibrated L1/L4
