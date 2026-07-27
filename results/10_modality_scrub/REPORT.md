# E10 — INLP modality-direction scrubbing (Iterative Nullspace Projection)

**Date:** 2026-07-27 · **Status:** ⚠️❌ Neutral — modality is linearly separable & removable, but scrubbing it does NOT help pathology transfer

## Hypothesis
The recording-modality signature (clinical vs watch: filter response, electrode
physics, noise color) lives in a **low-rank linear subspace** of a trained
encoder's penultimate features. Iteratively projecting that subspace out (INLP,
Ravfogel et al. 2020 — the NLP fairness workhorse, *never before applied to ECG
modality*) makes a downstream pathology classifier invariant to the shift
**without retraining the encoder** — the cheapest possible invariance layer.

## Setup
- Backbone: 12-lead lead-masking ECGResNet1d (E2 V2, prob=0.5), frozen.
- Penultimate features: 32-dim, extracted for clinical (12-lead) and sim-watch
  (Lead-I zero-pad) inputs, train + test.
- Modality adversary: logistic regression separating clinical vs watch train
  features. INLP = iteratively (K rounds) fit adversary → project its weight
  direction out → repeat.
- Pathology probe (linear, sklearn): two regimes — **cross-domain** (train
  clinical features → test watch; the domain-generalization question) and
  **in-domain** (train watch → test watch; does scrubbing hurt = entanglement?).
- Ablate K = 0, 1, 2, 3, 5, 8. `experiments/10_modality_scrub.py`.

## Results

| K | modality-adversary acc | cross-domain AUROC | in-domain AUROC |
|---|----------------------:|-------------------:|----------------:|
| 0 | 0.999 | 0.708 | 0.724 |
| 1 | 0.906 | 0.707 | 0.724 |
| 2 | 0.889 | 0.708 | 0.725 |
| 3 | 0.825 | 0.708 | 0.724 |
| 5 | 0.738 | 0.712 | 0.724 |
| 8 | 0.732 | 0.703 | 0.723 |

## Verdict: ⚠️❌ Neutral — the linear modality direction exists but scrubbing it doesn't help
Three findings:

1. **The modality direction is real and linearly separable.** Baseline
   modality-adversary accuracy = **0.999** — clinical vs watch is almost
   perfectly predictable from the 32-dim penultimate features alone. So the
   encoder *does* encode modality, and it's linearly accessible. The INLP
   premise (a linear device direction exists) is confirmed.

2. **INLP removes it only partially.** Modality accuracy drops 0.999 → 0.906
   (K=1) → 0.825 (K=3) → 0.738 (K=5) but **plateaus at ~0.73 by K=8**. Even after
   removing 8 linear directions from 32-dim space, modality stays 73% predictable.
   The modality signature is **distributed / nonlinear**, not concentrated in a
   few linear directions — INLP can dent it but not remove it.

3. **Scrubbing the modality direction does NOT help pathology transfer.**
   Cross-domain AUROC: 0.708 → 0.712 (K=5, best) → 0.703 (K=8). The full sweep
   spans 0.703–0.712 — within seed noise (~±0.005). In-domain is flat at 0.724.
   Removing the device direction neither closes the cross-domain gap (domain
   generalization fails) nor hurts (pathology is not heavily entangled with the
   *linear* modality direction).

## Interpretation — what this tells the project
The residual cross-domain gap (0.708 cross vs 0.724 in-domain) is **NOT a
linearly-removable modality shortcut.** Combined with E6 (simulator over-degrades)
and E1 (lead-count is the dominant axis), the picture consolidates: **the gap is
genuine information loss from lead reduction + sim/real distribution mismatch —
not a removable nuisance direction.** A post-hoc linear projection cannot recover
information the single-lead recording never contained.

This is the principled rejection of the I8 frontier idea (cheap post-hoc
geometric invariance): the modality is encoded, and linearly, but the part that
matters for the gap is the *information deficit*, which no embedding-space
surgery restores. INLP would only help if the encoder had taken a modality
*shortcut* that crowded out pathology signal — here the encoder (lead-masking
trained) already uses content features, so there's no shortcut to remove.

## Honesty flags
- Single seed, single backbone (lead-masking E2 V2). 32-dim feature space — INLP
  rank-limited (removing 8 of 32 dims is substantial; non-linear modality
  structure evades INLP by construction).
- Linear probes only — a nonlinear modality adversary (MLP) would show higher
  residual separability and is the natural follow-up (but INLP is fundamentally
  linear, so this is a method-intrinsic limit, not just a probe choice).
- sim-watch, not real-watch (E6 caveat); the modality direction found is
  clinical-vs-sim, which may differ from clinical-vs-real.

## Lesson
INLP is the right *diagnostic* but the wrong *intervention* here. It diagnoses
that modality is linearly encoded (0.999) yet nonlinearly distributed (plateau
0.73), and that the linear part is not the bottleneck for pathology transfer.
For ECG modality invariance, post-hoc feature surgery is insufficient — the
lever is either (a) training-time invariance (MixStyle E15, IRM E9 — prevent the
encoder relying on style) or (b) recovering the missing lead information (signal
synthesis B1, latent alignment E3). The E10/E15 pair will contrast post-hoc
removal (E10, neutral) vs by-construction prevention (E15).

## Follow-ups
- **E15 (MixStyle)** — the by-construction counterpart; does preventing style
  reliance at train time succeed where post-hoc removal failed?
- **E10b — nonlinear modality scrubbing:** a kernel/MLP adversary + HSIC or
  iterative nonlinear projection, to test whether the residual 0.73 modality is
  the part that matters. (Likely still neutral given the gap is information loss.)

## Artifacts
- `results/10_modality_scrub/metrics.json`
- `results/10_modality_scrub/inlp_tradeoff.png`
