# Experiment 18 — Scattering Transform Features (Novel, Deformation-Stable)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

Wavelet scattering coefficients (Mallat) are **provably Lipschitz-stable to
time-warp and amplitude deformations, and translation-invariant** — exactly the
perturbations modality shift induces. Unlike the speech cepstral approach (E8),
scattering preserves the time-frequency structure that ECG diagnostics need. A
training-free, deformation-stable front-end is the strongest "invariant-by-
construction" candidate (synthesis report I4).

## Setup
- Hand-rolled 1st + 2nd order Morlet scattering (J=6 octaves, Q=4 per octave).
- 44-dim feature (24 first-order + 20 second-order, capped).
- MLP classifier (30 ep) on sim-watch Lead-I.
- No learned front-end — the deformation-stability is *by construction*, not
  trained, so it can't overfit the clinical distribution.

## Result
- **Scattering @ L4 (full watch): macro AUROC = 0.661**
- Per-class: NORM 0.756 · MI 0.594 · STTC 0.654 · CD 0.652 · HYP 0.650
- feat_dim = 44 (compact)

## Verdict: ⚠️ — works (beats naive 0.521, competitive) but underperforms learned

Scattering (0.661) beats naive transfer (0.521) by a wide margin and is
competitive with learned approaches, *without any training of the front-end*.
But it underperforms the learned single-lead+sim winner (E17, 0.742) and
lead-masking (E2, 0.718). The deformation-stability guarantee buys real
modality-robustness for free, but the 44-dim compression loses discriminative
detail the end-to-end CNN retains.

## Why it's still interesting
- **Training-free modality robustness** — the only method here that needs no
  training data to be modality-invariant; the invariance is mathematical, not
  empirical. Useful as a frozen feature extractor or complement.
- **Complementarity (the F6 thesis)** — per SignalMC-MED (arXiv:2603.09940),
  hand-crafted features are *complementary* to FM embeddings. Scattering + a
  learned embedding ensemble may beat either alone (queued as E18b).
- **Strongest positive novel-method result so far** — unlike E8 (which failed),
  scattering validates a genuinely novel deformation-stable front-end for ECG
  modality invariance.

## Lesson
The right adjacent field for ECG modality invariance is **time-warp-stable
signal processing (scattering)**, not speech channel-robustness (E8). Both
target deformation invariance, but scattering preserves the time-frequency
structure ECG needs; cepstral features (E8) destroy it. The contrast E8 vs E18
is a clean positive-vs-negative pair on *which* cross-domain method transfers.

## Files
- `experiments/18_scattering.py` · `results/18_scattering/{metrics.json,run.log}`
