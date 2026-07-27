# E40 — Closed-loop calibrated augmenter hits the real Apple profile

**Date:** 2026-07-27
**Data:** SJLIFE paired, n=243. Source = clinical Lead-I; ground-truth target =
real Apple profile (E38). Common 100 Hz, per-record z-scored.

## Question
E39 showed the open-loop `CalibratedAWAugmenter` overshoots (injects bw≈0.53–0.82
when real Apple wants 0.23) and collapses kurtosis. Can a **closed-loop**
calibrator — one that tunes injection until the *measured* output matches the
target — hit the real profile while preserving morphology?

## Method
`ClosedLoopCalibrator`:
- wander model = **<1 Hz coloured (1/f-ish) noise**, not multi-tone sinusoids
  (sinusoids crush kurtosis; coloured wander preserves peak structure).
- **binary-search the wander amplitude (18 iters)** on a 40-signal probe until
  measured `bw_energy(clinical + a·wander) == target bw_energy`.
- QRS band left untouched → morphology preserved.
- mild global gain wander for wearable realism.

## Result — WIN ✅

| variant | dist→realApple ↓ | QRS-corr | R-peak | bw_energy | kurtosis | qrs_energy |
|---|---|---|---|---|---|---|
| clean (floor) | 1.059 | 1.000 | 1.000 | 0.033 | 11.94 | 0.383 |
| light_DR | 1.063 | 0.993 | 0.965 | 0.035 | 12.62 | 0.382 |
| open_loop_calDR | 1.979 | 0.971 | 0.938 | 0.816 | 0.08 | 0.071 |
| **closed_loop** | **0.659** | **0.988** | **0.963** | **0.217** | **7.50** | **0.293** |
| target (real Apple) | — | — | — | 0.230 | 9.46 | 0.242 |

- **Distance 1.059 → 0.659** — first augmenter to beat the clean floor
  (**38% closer** to real Apple). Calibrated wander amplitude = 0.586.
- **bw_energy 0.217 vs target 0.230** — hits the dominant axis (open-loop
  overshot to 0.816; light-DR barely moved 0.035).
- **kurtosis 7.50 vs 9.46** — preserved (open-loop destroyed it → 0.08).
- **Morphology intact:** QRS-corr 0.988, R-peak 0.963 → label valid.

## Why it works
The open-loop heuristic (`sqrt(gap)·4·cover`) is uncalibrated and uses
kurtosis-destroying multi-tone sinusoids. Closing the loop (measure → adjust)
plus a spectrally-correct wander model fixes both the overshoot and the kurtosis
collapse simultaneously. Residual gap (0.659, not 0) is mostly qrs_energy /
mid_energy the wander model doesn't touch — a future axis to add.

## Honest flags
- Coverage is **necessary-not-sufficient** (E25b: fidelity ≠ utility). This
  proves we can now AIM at the real target precisely; it does NOT prove
  downstream AUROC gain. That test needs labels (SJLIFE has none).
- Population-level distance; recordings not beat-aligned.
- Single calibration seed; probe n=40.

## Verdict ✅ → promote + spawn
`ClosedLoopCalibrator` promoted to `src/aw_generator.py`. Residual axes
(qrs_energy, mid_energy) logged for a future multi-axis closed-loop.

## Artifacts
`metrics.json`, `closed_loop.png` (coverage bars + per-axis profile match).
