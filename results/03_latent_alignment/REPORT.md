# Experiment 3 — Latent-Space Lead Alignment (SelfMIS Self-Cutting)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

Tests the synthesis report's top-ranked approach: align single-lead embeddings
to 12-lead embeddings directly in latent space (SelfMIS, arXiv:2509.19397),
rather than signal-level synthesis. Self-cutting: the same encoder processes both
the full 12-lead and the Lead-I (zero-padded) of the SAME record; InfoNCE pulls
them together. Then a linear probe is trained on the 12-lead embeddings and
applied to single-lead test inputs.

## Setup
- PTB-XL 100 Hz, 5 superclasses. Train n=1225, test n=497.
- Featurizer = ECGResNet1d backbone (64-d), FEAT_DIM=64.
- Phase A: 20-epoch contrastive pretrain (InfoNCE, τ=0.1, batch 128).
- Phase B: freeze encoder, 20-epoch linear probe on 12-lead embeddings.
- Two variants:
  - **A** clean self-cutting (single = clean Lead-I)
  - **B** watch-sim self-cutting (single = Lead-I through the F10 simulator)

## Results

| Variant | L1 clean Lead-I | L4 full watch |
|---|---|---|
| A latent (clean) | 0.686 | 0.655 |
| B latent (+watch-sim) | 0.714 | 0.700 |
| *ref: V2 LeadMask (E2)* | *0.750* | *0.717* |
| *ref: V1 Naive (E2)* | *0.470* | *0.521* |

## Findings

1. **Latent alignment works.** Both A (0.655) and B (0.700) on full watch beat
   naive transfer (0.521) by a wide margin — confirming the SelfMIS thesis that
   latent-space alignment helps single-lead transfer.

2. **Watch-sim variant (B) > clean variant (A)** (0.700 vs 0.655 on L4). Aligning
   *watch-like* single-lead embeddings to 12-lead embeddings is better than
   aligning clean single-lead — because the test distribution is watch-like. This
   is independent evidence the F10 simulator is useful as a *training* tool for
   alignment, not just an eval probe.

3. **But latent alignment underperforms end-to-end lead-masking** (B 0.700 vs
   V2 0.717 on L4). The likely reason: V2 trains the *full network end-to-end*
   for classification with lead-dropout, while E3 freezes the encoder after
   contrastive pretraining and trains only a *linear* probe on 12-lead embeddings.
   The probe is shallower and trained on z_full, not z_single. A fairer comparison
   would fine-tune the encoder post-contrastive (queued as future work).

## What this refines in the synthesis report
The report ranked latent-space alignment (B9) as the top solution based on
SelfMIS. SelfMIS showed latent alignment > **signal synthesis**, not > lead-masking.
E3's result refines the ranking for the lead-count axis specifically:
**end-to-end lead-masking (C9/E2) > latent alignment + linear probe (B9)** on PTB-XL.
The simple training-time augmentation wins. Latent alignment remains promising if
the encoder is fine-tuned (not frozen) — left to future work.

## Limitations
- Frozen encoder + linear probe is a weaker classifier than end-to-end — unfair
  to V2. Fine-tune variant queued.
- No signal-synthesis baseline (B1) yet to directly demonstrate the SelfMIS
  "synthesis < latent alignment" claim on this data — queued.
- 100 Hz; single seed.

## Files
- `experiments/03_latent_alignment.py` · `results/03_latent_alignment/{metrics.json,ladder.png,run.log}`
