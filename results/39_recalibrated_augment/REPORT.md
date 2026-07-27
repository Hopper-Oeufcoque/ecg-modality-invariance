# E39 — Recalibrate augmenter to real Apple profile: coverage + morphology

**Date:** 2026-07-27
**Data:** SJLIFE paired, n=243. Source = clinical Lead-I; ground-truth target =
real Apple modality profile (E38). Common 100 Hz, 1000 samples, per-record
z-scored.

## Question
Does calibrating `CalibratedAWAugmenter` to the **real** E38 target (vs the old
CinC target, vs light-DR, vs clean) produce augmented clinical Lead-I whose
modality distribution matches real Apple better, while preserving morphology?

## Result — informative NEGATIVE

| variant | dist→realApple ↓ | QRS-corr | R-peak | bw_energy | kurtosis |
|---|---|---|---|---|---|
| clean (floor) | 1.059 | 1.000 | 1.000 | 0.033 | 11.94 |
| light_DR (str 0.5) | **1.045** | 0.985 | 0.961 | 0.042 | 12.71 |
| cinc_calDR | 1.217 | 0.951 | 0.920 | 0.558 | 3.49 |
| real_calDR | 1.130 | 0.957 | 0.924 | 0.528 | 3.59 |
| **target (real Apple)** | — | — | — | **0.230** | **9.46** |

- **light-DR is too gentle:** it barely perturbs the profile (bw 0.033→0.042),
  so it neither helps nor hurts coverage. Distance ≈ clean.
- **calibrated-DR OVERSHOOTS:** real Apple wants bw≈0.23; the augmenter injects
  bw≈0.53 (2.3×) AND collapses kurtosis (11.9→3.6) because heavy multi-tone
  sinusoidal wander dominates the z-scored signal. Net distance gets **worse**
  than clean.
- Recalibrating to the real target (real_calDR) is better than the CinC target
  (1.130 < 1.217) — right direction — but still overshoots and still loses.
- **No augmenter closes the gap:** the coverage floor is stuck at ~1.05.

## Root cause
`CalibratedAWAugmenter` is **open-loop**: it maps a target energy gap to an
injection amplitude via a fixed heuristic (`sqrt(gap)·4·cover`) and never checks
the resulting measured profile. The heuristic massively over-injects on real
data and destroys kurtosis as a side effect. Calibration must be **closed-loop**
— tune the injection until the *measured* output profile hits the target.

## Honest flags
- Coverage is **necessary-not-sufficient** (E25b: fidelity ≠ utility). Even a
  perfectly-matched profile isn't guaranteed to help downstream; but a badly
  overshooting one is a clear miss. This is a diagnostic, not the endpoint.
- Population-level distance (not beat-aligned).
- Downstream AUROC still needs labels — SJLIFE has none.

## Verdict ❌ (open-loop calibration) → spawns E40
The open-loop augmenter cannot hit the real target. Next: **E40 closed-loop
calibrator** that binary-searches the wander/HF amplitudes until the measured
output profile matches the E38 target on each axis, with kurtosis preserved.

## Artifacts
`metrics.json`, `recalibration.png` (coverage bars + per-axis profile match).
