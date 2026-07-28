# E43 — Multi-axis closed-loop calibration: HF axis is a no-op (no gap to close)

**Date:** 2026-07-27
**Setup:** identical to E42 (train PTB-XL Lead-I AFIB/NORM → test real held-out
CinC AF/N, 20 seeds). Adds a second closed-loop axis: high-frequency noise tuned
to target hf_energy, on top of the E42 baseline-wander axis.

## Question
E40's single-axis (bw-only) calibration left residual spectral gaps. Does closing
a second axis (hf_energy) beat E42's +0.041 single-axis lift?

## Result

| arm | AUROC (20 seeds) |
|---|---|
| clean | 0.701 ± 0.048 |
| closed_bw (single-axis, E42) | 0.742 ± 0.047 |
| closed_bw_hf (multi-axis) | 0.735 ± 0.045 |
| oracle | 0.932 ± 0.010 |

- **closed_bw − clean:** +0.041, 15/20, p=0.0088, dz=0.65 (reproduces E42 exactly)
- **closed_bw_hf − clean:** +0.035, 13/20, p=0.0091, dz=0.65
- **closed_bw_hf − closed_bw (does HF axis help?):** **−0.007, 10/20, p=0.67, dz=−0.10 → NO.**

## Why — and it confirms our physics
The closed-loop search set **hf_amp = 0.000**. Reason: the measured target
**hf_energy = 0.005** — CinC has almost NO high-frequency content (it is a
low-pass, HF-clean signal). There is no HF gap to close, so the second axis adds
nothing; the tiny residual noise from the search slightly hurt (n.s.).

This independently reconfirms E37/E38: **real single-lead ECG (CinC here, real AW
there) is a CLEAN, low-pass signal — the modality gap is baseline-wander, NOT
high-frequency noise.** A calibrator should close the wander axis and leave HF
alone. E43 is the end-to-end (AUROC-level) confirmation of that spectral finding.

## Verdict ❌ (multi-axis) / ✅ (reconfirms single-axis + clean-signal physics)
Single-axis baseline-wander closed-loop calibration remains the winner (+0.041).
Adding an HF axis is neutral-to-slightly-negative because the target has no HF
gap. `MultiAxisClosedLoopCalibrator` retained in src/ (it correctly zeroes
unused axes — useful if a future target DOES have an HF gap), but for
CinC/AW-like clean single-lead, single-axis is the recipe.

## Honest flags
- CinC finger ≠ AW wrist; AF/NORM easy task; single fixed clinical train set
  across seeds (CI omits cohort variance). Same as E42.
- Sequential 2-axis search; qrs/mid axes still not addressed (but those are
  in-band and can't be perturbed without touching morphology — likely a hard
  ceiling, not a lever).

## Artifacts
`metrics.json`, `multiaxis.png`.
