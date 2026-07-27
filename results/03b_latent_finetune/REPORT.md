# Experiment 3b — Latent Alignment, Pretrain + End-to-End Fine-Tune (Fair Test)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

E3 compared latent alignment (frozen encoder + linear probe) against V2
lead-masking (end-to-end) — unfair. E3b fixes it: contrastive self-cutting
pretrain (variant B) THEN end-to-end fine-tune the whole network WITH
lead-masking. This is the fair test of whether contrastive pretraining adds
value over lead-masking alone.

## Result

| Model | L1 clean Lead-I | L4 full watch |
|---|---|---|
| V2 lead-masking (no pretrain) | 0.750 | 0.717 |
| **E3b pretrain + finetune (lead-masking)** | **0.721** | **0.690** |

## Finding
Contrastive pretraining does **not** add value on top of end-to-end
lead-masking — it's slightly *worse* (0.690 vs 0.717 on L4). The contrastive
objective may preserve lead-distinguishing structure that is unhelpful for
classification, or the 64-d bottleneck shifts the encoder into a representation
that fine-tunes less well than a from-scratch lead-masking model.

## Implication
For end-to-end training on PTB-XL superclasses, **lead-masking alone is the best
method; latent-space pretraining is not worth the extra complexity.** Latent
alignment remains the right tool when end-to-end training isn't possible (frozen
foundation model + adapter, the Solution-1 / LeadBridge path) — there its value
is in transferring a *frozen* FM, not in beating a from-scratch lead-masked model.

This refines the synthesis report's ranking: for the lead-count axis with
end-to-end training available, lead-masking (C9) > latent alignment (B9).

## Limitations
- Single seed; the 0.027 gap could partly be noise — but it's not in the
  direction latent alignment would need.
- 64-d feature dim, τ=0.1 not tuned.
- 100 Hz.

## Files
- `experiments/03b_latent_finetune.py` · `results/03b_latent_finetune/{metrics.json,run.log}`
