# E59 — Frequency-band "modality dropout" augmentation

**Branch:** `explore/novel-methods` · **Idea:** N4 (`docs/NOVEL_IDEATION.md`)
**Date:** 2026-07-27 · **Seeds:** 20 · **Task:** real CinC 2017 AF vs NORM

## Question
The modality gap is dominated by the low-frequency out-of-band component
(baseline wander, <~1.5 Hz; E38/E43). Calibration (E42) *injects* matched wander.
N4 tries the opposite mechanism — **remove/randomize the low band during training**
so the model cannot build any dependence on it — forcing reliance on in-band
morphology by construction. Crucially, this uses **no watch data at all** (no paired
set, no target profile) — it's a pure structured augmentation on clinical Lead-I.

Three variants (low band = 4th-order Butterworth < 1.5 Hz; in-band QRS/T preserved,
verified corr 0.9997):
- **band_drop** — attenuate the low band by a random gain ∈ [0,1] per sample.
- **band_scramble** — phase-randomize the low band (keep its power spectrum,
  destroy wander *structure*) per sample.
- **band_both** — each sample randomly drops OR scrambles OR stays clean.

## Result (AUROC on real CinC, 20 seeds)

| Arm | AUROC | vs clean | vs calibration | vs paired |
|-----|-------|----------|----------------|-----------|
| clean | 0.701 ± 0.048 | — | — | — |
| closed_aug (calibration) | 0.742 ± 0.047 | +0.041 | — | — |
| **band_drop** | **0.666 ± 0.052** | **−0.035 (p=0.042) HURTS** | −0.076 | — |
| **band_scramble** | **0.739 ± 0.046** | **+0.038 (p=0.006, 13/20) WIN** | −0.003 (≈tie) | — |
| **band_both** | **0.723 ± 0.047** | +0.022 (p=0.072, borderline) | −0.019 (p=0.19, n.s.) | −0.084 (p<1e-4) |
| joint_paired (E51) | 0.807 ± 0.023 | +0.106 | +0.065 | — |

## Verdict — ✅ PARTIAL WIN (scramble ties calibration with NO watch data)
- **band_scramble matches calibration** (0.739 vs 0.742, Δ−0.003, statistical tie)
  and beats clean by a real, significant +0.038 (13/20, p=0.006) — **while using
  zero watch data and requiring no target-profile measurement.** Calibration needs
  an unlabeled target set to measure the baseline-wander level to hit; scramble
  needs *nothing but the clinical training data*. That makes it the cheapest lever
  we have to date, at calibration-tier benefit.
- **band_drop HURTS** (−0.035, p=0.042). Attenuating the low band with a random
  gain sometimes strips genuine low-frequency discriminative content (P-wave energy,
  slow ST-segment shifts matter for rhythm/morphology) → destroys signal, doesn't
  teach invariance. A cautionary echo of the E50/E58 information-destruction motif:
  *removing* the band is destructive; *randomizing its structure* is not.
- **band_both is diluted** (0.723) — mixing the harmful drop in with the helpful
  scramble drags the mean down. Scramble alone is the right recipe.
- All band variants stay far below paired (0.807) — consistent with the two-lever
  picture: augmentation-family methods (calibration, scramble) share the same
  information-bound ceiling (~0.74, E48), and only same-patient correspondence
  (paired alignment) breaks past it.

## Why scramble works where drop fails (mechanism)
Baseline wander is a *structured* low-frequency waveform. Phase-scrambling keeps
the **amount** of low-frequency energy realistic (so the network still sees
wander-like power) but destroys its **temporal structure** every epoch — so the
network cannot latch onto any particular wander morphology as a feature. It learns
a representation invariant to wander shape while never being starved of the band.
Dropping, by contrast, *removes* energy (variably to zero), which both (a) creates
an unrealistic clean-signal distribution the real watch never matches and (b)
discards real low-frequency diagnostic content. Structure-randomization ≫
amplitude-removal for modality robustness.

## Consequence for the goal
Adds a **third, cheapest lever** to the toolbox: a no-data augmentation that reaches
calibration-tier transfer. Deployment implication — even when you have *no* unlabeled
target data to profile (calibration's one requirement), you can still buy the ~+0.04
robustness gain purely from clinical data via band-phase-scrambling. Natural next
step: **does scramble stack with calibration** (they may share mechanism → redundant,
like E2-V4 aug+lead-masking) **and does it stack under paired alignment** (different
mechanism → possibly additive)? Logged as follow-up.

## Honesty flags
- CinC dry-finger ≠ AW wrist; AF-vs-NORM easy axis.
- Cutoff 1.5 Hz chosen a priori (not swept); single clinical train set; 20 seeds.
- Scramble ties calibration *on CinC* — the dose-response law (E44) means both are
  gap-proportional; the tie may not hold at a different modality gap.
- No watch data used is a feature, but also means scramble can't *target* a specific
  device profile the way calibration can — it's device-agnostic robustness.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/59_band_dropout && \
  python3 experiments/59_band_dropout.py > results/59_band_dropout/run.log 2>&1
```
Figure: `results/59_band_dropout/band_dropout.png`
