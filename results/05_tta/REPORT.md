# Experiment 5 — Test-Time BN Adaptation (H3) on the E2 Winner

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

Layers the cheapest label-free post-hoc invariance (BN adaptation) on top of
V2 LeadMask, the E2 winner. Question: does re-aligning BN statistics to the
target (watch) distribution at inference add anything once lead-masking already
handles the dominant axis?

## Setup
- Retrain V2 LeadMask (20 ep). Test on L1 / L4 with and without BN-adapt
  (reset BN running stats, recompute from the target batch, 2 passes).

## Results

| Stage | no TTA | BN adapt | Δ |
|---|---|---|---|
| L1 clean Lead-I | 0.747 | 0.737 | −0.011 |
| L4 full watch | 0.701 | 0.708 | +0.007 |

## Finding
BN-adapt helps marginally under genuine shift (L4 +0.007) and slightly hurts
when shift is small (L1 −0.011) — exactly what TTA theory predicts. The effect
is small because **lead-masking already closed the dominant (lead-count) axis**,
leaving only the minor noise axis for TTA to nibble. TTA is a finishing move,
not a main lever.

## What this means for the method ladder
- TTA stacks cleanly on top of lead-masking for a small extra gain under shift.
- Per-clip TTA (single 30 s watch clip) is under-powered for BN stats; a
  follow-up using entropy-min on a LoRA adapter (H4) would be the stronger
  per-clip variant — flagged as future work.

## Files
- `experiments/05_tta.py` · `results/05_tta/{metrics.json,tta.png,run.log}`
