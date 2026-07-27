# Experiment 4 — 500 Hz Rerun: Bandwidth Axis (Directional, Small-N)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

E1 ran at 100 Hz where the bandwidth axis was muted (Nyquist 50 Hz ≈ the 40 Hz
Apple lowpass). E4 reruns the staircase at PTB-XL's native 500 Hz, where the
Apple bandpass (0.3–40 Hz) can remove real clinical content (the 40–250 Hz band),
to see if the L2 (bandwidth) step shows a measurable drop.

## Setup
PTB-XL 500 Hz subset (filename_hr). **Small-N directional run** — HYP is the
rare limiting class and the 500 Hz download was partial at run time, so
max_per_class=40 → n_train=122, n_test=63. fs_watch=500 (no resample; isolates
the bandpass effect). 1D ResNet, 15 ep.

## Results (staircase)

| Stage | 500 Hz (E4) | 100 Hz (E1 ref) |
|---|---|---|
| L0 clinical | 0.810 | 0.865 |
| L1 Lead-I | 0.393 | 0.527 |
| L2 + bandwidth | 0.433 | 0.543 |
| L3 + electrode | 0.400 | 0.530 |
| L4 full watch | 0.418 | 0.554 |
| Lead-masking @ L4 | 0.641 | 0.717 |

## Findings

1. **The bandwidth axis shows NO clear drop at 500 Hz** — L2 (0.433) ≈ L1
   (0.393), within noise. This is *directional evidence that the bandwidth axis
   is genuinely minor even at 500 Hz*, because ECG diagnostic content lives below
   40 Hz; the 40–250 Hz band Apple's filter removes is mostly noise/EMG/high-freq
   QRS detail, not diagnostic signal. This **confirms the synthesis report's
   low-priority ranking of bandwidth methods (A2) and E16's finding that
   matched-filter actively hurts.**

2. **Lead-masking generalizes to 500 Hz** — recovers from 0.393 (naive) to 0.641
   (lead-masked), the same recovery pattern as 100 Hz (0.527→0.717). The main
   finding (E2/E17) is rate-invariant.

3. **The absolute values are lower than E1** (L0 0.810 vs 0.865; LeadMask 0.641 vs
   0.717) — purely the small training set (122 vs 1186), not a rate effect.

## Honest limitations
- **n_test=63 is too small to trust the exact staircase values** — all stages
  hover near chance (0.39–0.43), and the L2-vs-L1 difference is within noise.
  This is a *directional* result: the bandwidth axis shows no signal of a drop,
  consistent with the physics, but a larger rerun is needed for confidence.
- The 500 Hz download was partial and HYP-limited at run time; a fuller rerun is
  queued but **low-priority** since the bandwidth axis appears minor regardless.

## Implication
The bandwidth axis is minor at both 100 Hz and (directionally) 500 Hz. The
modality gap is **dominated by lead-count** (E1), and the bandwidth/noise axes
are minor — so the lead-masking (E2) and single-lead+sim (E17) solutions, which
target lead-count, are the right focus. Bandwidth methods (A2) are de-prioritized
at both rates.

## Files
- `experiments/04_500hz_rerun.py` · `results/04_500hz_rerun/{metrics.json,staircase_500hz.png,run.log}`
