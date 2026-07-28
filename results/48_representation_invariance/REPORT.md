# E48 — Does learned representation-invariance beat implicit augmentation?

**Date:** 2026-07-27
**Script:** `experiments/48_representation_invariance.py`
**Seeds:** 20 · **λ (invariance weight):** 0.5 (a-priori, not tuned)

## Hypothesis
E43/E46 showed pure augmentation plateaus at ~+0.041 because the residual
clean→oracle gap lives *in-band* (QRS/mid-freq) and cannot be filled by injecting
out-of-band noise. A **learned modality-invariant representation** attacks the
problem from a different angle: instead of only showing the model shifted data,
add an explicit loss forcing the encoder to emit **identical features for a
clinical sample and its modality-shifted twin**. If a learned invariant feature
space can reach the target differently than augmentation, it should push past the
+0.041 augmentation ceiling.

## Setup
- Train: PTB-XL Lead-I, AFIB vs sinus. Test: **real** CinC 2017 AF vs Normal.
- Three arms (identical 1D-ResNet backbone, `n_leads=1`, per-record z-score):
  1. **clean** — train on clinical Lead-I only.
  2. **closed_aug** — E42 winner: train on closed-loop-calibrated views (implicit invariance).
  3. **invariance** — dual forward pass (clean view + calibrated view); loss =
     CE(clean) + CE(shifted) + λ·‖feat(clean) − feat(shifted)‖² (explicit
     feature-consistency invariance).
- Same calibrator (`ClosedLoopCalibrator`) defines the shifted view in both 2 & 3.

## Results (AUROC on real CinC, 20 seeds)

| Arm | AUROC | Δ vs clean | wins | p (paired t) |
|---|---|---|---|---|
| clean | 0.701 ± 0.048 | — | — | — |
| closed_aug | **0.742 ± 0.047** | **+0.041** | 15/20 | **0.0088** |
| invariance | 0.732 ± 0.043 | +0.031 | 14/20 | 0.083 (n.s.) |

**Head-to-head — does explicit beat implicit?**
`invariance − closed_aug: Δ = −0.010, wins 10/20, p = 0.47` → **NO.**

## Verdict ❌ (informative null)
Explicit feature-consistency invariance does **not** beat implicit augmentation —
it lands ~0.010 *below* it, well inside noise (p=0.47), and its own lift vs clean
(+0.031) fails to reach significance. Two arms that share the same modality-shift
generator converge to the same place regardless of whether invariance is enforced
implicitly (augmentation) or explicitly (feature loss).

## Interpretation — the ceiling is information-bound, not formulation-bound
This is the key takeaway. We now have **two independent method families**
(augmentation E42/E43, learned invariance E48) hitting the **same ~+0.041 wall**.
That strongly implies the wall is a property of the *information available in a
single clinical lead*, not an artifact of how we wrote the objective. You cannot
learn your way to invariance against a shift whose discriminative residual lives
in the same band as the signal you need — no loss function creates information
that the modality shift has already entangled with the label-relevant morphology.

**Consequence for the north star:** closing the remaining gap to oracle (~0.19 on
CinC) will NOT come from a cleverer single-lead invariance objective. It requires
either (a) real target labels (E46: ~50 labels → 0.855), or (b) *new information* —
multi-lead clinical structure distilled into the single-lead student, or auxiliary
channels (accelerometer/PPG) that watches actually carry. Method design should
pivot from "better invariance loss" to "inject more information."

## Honesty flags
- CinC dry-finger ≠ AW dry-wrist (proxy).
- AF-vs-Normal is the *easy* rhythm task; E47 showed morphology transfer is near-chance regardless.
- Single clinical train set (PTB-XL), single test set (CinC).
- λ=0.5 fixed a-priori, not swept — a tuned λ *might* recover a hair, but the
  head-to-head being −0.010 (not +0.00x) makes a large hidden win unlikely.
- 20 seeds, single architecture.
