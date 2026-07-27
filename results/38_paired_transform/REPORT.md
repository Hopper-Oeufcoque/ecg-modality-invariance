# E38 — Real Paired Clinical Lead-I ↔ Apple Watch Modality Profile

**Date:** 2026-07-27
**Data:** SJLIFE Paired repo (public, trainable), n=243 patients, each with a
12-lead clinical ECG (500 Hz × 10 s) and their own Apple Watch single-lead ECG
(512 Hz × 30 s). Same individual, both devices.

## Why this matters
Every prior modality experiment (E1–E37) had to *guess* the clinical→watch
transform via a forward-physics simulator (over-degraded, E6/E22) or a
distribution proxy (CinC, E33; HOME cleaned AW, E34/E37). SJLIFE gives the
**real per-patient pairing** — the first time we can measure the true transform
directly. This retires the project's biggest standing caveat ("simulated,
not real").

## Method
- Loaded all 243 pairs (0 download failures). Verified sampling rates with a
  heart-rate sanity check (AGENTS.md mandate): Apple 69.8 bpm (meta 69 ✓),
  clinical Lead-I 71.6 bpm (meta VR 71 ✓).
- Amplitude audit on native stored units (p99 of |x|, peak-to-peak).
- Modality profile via `signal_modality_stats` (z-scored, resampled to common
  100 Hz) — same axes used in E33/E34/E37 for comparability.

## Results

### 1. Voltage scale (the audit Hop asked for)
- **All 243 Apple files are on ONE consistent scale** — 0 outliers beyond 5× /
  ⅕ of median peak-to-peak; single unimodal order-of-magnitude spread.
- Unit = **microvolts (µV)**. Median R-wave ≈ 700–1200 µV; max ≈ 6 mV.
- **Cross-modality gain:** Apple median p99|x| = 718 vs clinical Lead-I = 77 →
  **~8.25× gain** (IQR 6.6–11.5×, cv 0.59). Clinical stored in a coarser unit.
- **Consequence:** per-record z-score normalization is mandatory before any
  cross-modality model; otherwise the 8× gain leaks in as fake signal.

### 2. Modality profile (z-scored, 100 Hz)
| axis | clinical Lead-I | Apple Watch | direction |
|---|---|---|---|
| kurtosis | 11.94 ± 7.74 | 15.31 ± 17.11 | AW peakier/noisier |
| bw_energy (<1 Hz) | 0.033 ± 0.065 | 0.202 ± 0.159 | **AW 6× more baseline wander** |
| qrs_energy (5–15 Hz) | 0.383 ± 0.105 | 0.270 ± 0.132 | AW relatively lower QRS band |
| hf_energy (30–50 Hz) | 0.030 ± 0.025 | 0.011 ± 0.010 | AW LESS HF (confirms E37) |
| mid_energy (1–5 Hz) | 0.319 | 0.390 | — |

Normalized clinical→Apple distance = **1.065**.

## Interpretation
- The real modality gap is **baseline wander + mild noise**, NOT a high-frequency
  gap. This is the third independent confirmation (E37, E34, now E38) that the
  retracted E35/E36 "HF/bandwidth gap" was an artifact.
- The paired gap (1.065) is **larger** than HOME's (0.25–0.50) because HOME's
  `data-for-predicting` AW is pre-cleaned/filtered, whereas SJLIFE Apple is
  closer to raw wrist output (heavy baseline wander = expected for a dry-
  electrode single-lead). **SJLIFE is the more honest raw-AW target.**
- Recipe implication: keep LIGHT-DR direction but **calibrate baseline-wander UP
  toward ~0.20**, not CinC's over-injection (0.5) and not near-zero.

## Honesty flags
- Recordings ~64 min apart → **not beat-aligned**; only distributional /
  population-spectral characterization is valid (no per-beat morphology
  regression, no sample-wise transfer function).
- n=243, young SJLIFE survivor cohort — skews amplitude/HR; not representative
  of an older cardiac population.
- Lead I only from the 12-lead clinical.
- **No disease labels in this repo** (metadata = age/sex/race/HR/time-gap). This
  establishes the *transform*, not a diagnostic endpoint. Downstream AUROC still
  needs a labeled single-lead set.

## Artifacts
- `metrics.json` — full numbers.
- `real_apple_target_profile.json` — measured target profile for
  CalibratedAWAugmenter (drives E39).
- `paired_profile.png` — gain-ratio histogram, profile bars, example z-scored
  waveforms, per-patient amplitude scatter.

## Next (E39)
Re-calibrate `CalibratedAWAugmenter` to the E38 measured profile (wander↑, HF
flat), verify morphology preserved (QRS-band corr + R-peak match), then test
whether train-on-clinical + real-calibrated-DR reproduces the SJLIFE Apple
distribution better than the CinC-calibrated and light-DR variants.
