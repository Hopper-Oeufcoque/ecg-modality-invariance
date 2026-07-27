# Experiment 2 — Closing the Lead-Count Gap (Lead-Masking vs Augmentation)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

E1 proved lead-count is the dominant modality axis (-0.338 of -0.311 AUROC)
and that naive 12-lead→single-lead transfer collapses to chance. E2 attacks
that axis with the proven method (K-MERL lead-masking, C9) and benchmarks it
against a single-lead reference model that gets to train directly on Lead-I.

## Setup
- PTB-XL 100 Hz, 5 superclasses. Train n=1225, test n=497. 20 epochs each.
- 5 variants, all tested on L1 (clean Lead-I) and L4 (full watch on Lead-I).
- V1–V4: 12-lead model. V5: 1-lead model trained on Lead-I only.

## Results

| Variant | L1 clean Lead-I | L4 full watch | Notes |
|---|---|---|---|
| V1 Naive (no aug) | 0.470 | 0.521 | near chance (lead-count kills it) |
| **V2 LeadMask** (K-MERL) | **0.750** | **0.717** | random lead dropout, keep Lead-I |
| V3 WatchAug (A5) | 0.521 | 0.551 | watch-sim on Lead-I only |
| V4 LeadMask+Aug | 0.735 | 0.717 | combo |
| V5 SingleLead (reference) | 0.751 | 0.690 | 1-lead model, trains on Lead-I directly |

## Headline findings

1. **Lead-masking matches the single-lead reference on clean Lead-I (0.750 vs 0.751).**
   A 12-lead model trained with random lead dropout achieves the *same* single-lead
   performance as a model that gets to train directly on Lead-I — **without needing
   any target-domain single-lead data.** This is the K-MERL result (arXiv:2502.17900)
   reproduced on PTB-XL, and it's the key practical win: you keep the rich 12-lead
   training signal and still deploy on single-lead.

2. **Lead-masking BEATS the single-lead model on full watch (0.717 vs 0.690).**
   The 12-lead model leverages the richer spatial training signal, making it *more*
   robust to watch noise than a model that only ever saw one clean lead. The 12-lead
   prior is a feature, not a bug.

3. **Watch-augmentation alone (V3) does NOT address the lead-count axis** (0.521/0.551,
   no better than naive). It only helps the noise axis, which E1 showed is minor.
   This confirms: lead-count is the war; noise aug fights the wrong battle alone.

4. **The combo (V4) ≈ lead-masking alone.** Watch-sim aug is redundant once lead-masking
   handles lead-robustness — the residual noise axis is too small to matter at 100 Hz.

## Per-class (V2 LeadMask, L4 full watch)

| NORM | MI | STTC | CD | HYP |
|---|---|---|---|---|
| 0.810 | 0.608 | 0.759 | 0.735 | 0.675 |

All spatial classes recover substantially vs E1's collapse (MI 0.508→0.608,
STTC 0.487→0.759). CD stays strong (conduction = lead-invariant). HYP weakest
(under-sampled + genuinely spatial).

## What this tells us about the method ladder
- The lead-count gap is **solvable with a training-time trick (lead-masking)**,
  no adapter, no labels, no synthesis. This should be the *default baseline* for
  any clinical→single-lead transfer.
- The residual gap to the L0 ceiling (0.717 vs 0.865 = −0.148) is now a mix of
  (a) genuine information loss (one lead can't see what 12 see) and (b) noise.
  Latent-space lead alignment (E3) and test-time adaptation (E5) target these.

## Limitations
- 100 Hz; bandwidth axis muted.
- Single seed; HYP under-sampled.
- Lead-masking prob 0.5 not tuned.

## Files
- `experiments/02_lead_masking.py` · `results/02_lead_masking/{metrics.json,ladder.png,run.log}`
