# E51b — Mechanism control: E51's win IS real cross-modality invariance (CONFIRMED)

**Date:** 2026-07-27
**Script:** `experiments/51b_align_control.py`
**Seeds:** 20 · **λ:** 0.1 · **temp:** 0.1

## Question
E51's +0.078-vs-calibration win came with a variance collapse (0.048→0.023), the
footprint of a strong regularizer. Confound: is the gain from **same-patient
modality alignment** (true invariance) or from bolting **any** auxiliary InfoNCE loss
on extra real ECG (generic SSL regularization)? Matched controls, identical joint
loop, only the alignment positive-pair definition changes.

## Results (AUROC real CinC, 20 seeds)

| Arm | AUROC | Δ vs clean | meaning |
|---|---|---|---|
| clean | 0.701 ± 0.048 | — | floor |
| **joint** (correct pairs) | **0.807 ± 0.023** | +0.106 (20/20, p=5e-8) | E51 result reproduced |
| joint_shuffled (mismatched pairs) | 0.706 ± 0.041 | **+0.005 (p=0.76, n.s.)** | aux InfoNCE on real ECG, no correspondence |
| joint_selfclin (own calibrated view) | 0.742 ± 0.031 | +0.041 (p=0.005) | = calibration, no watch data |

**Decisive contrasts:**
- `joint − shuffled = +0.101` (20/20, **p=3e-10**)
- `joint − selfclin = +0.065` (19/20, p=4e-8)

## Verdict ✅✅ — CONFIRMED: the effect is genuine cross-modality invariance
Three hypotheses, cleanly separated:

1. **Generic aux-SSL regularization? NO.** Shuffling the pairs (clinical ↔ a
   *different* patient's watch) keeps the auxiliary InfoNCE loss and all the extra
   real ECG data, yet the gain **completely vanishes** (0.706 ≈ 0.701, p=0.76).
   If the benefit were just "adding a contrastive regularizer," shuffled would have
   kept it. It didn't. **Generic-regularization hypothesis falsified.**

2. **Just augmentation-consistency? NO.** Aligning clinical to its own calibrated
   view (no watch data at all) reproduces *exactly* the calibration effect (0.742 =
   E42's 0.742) and no more. So within-modality self-consistency = calibration; it
   does **not** explain the extra +0.065 that real pairs deliver.

3. **Same-patient cross-modality correspondence? YES.** Only the *correct* real
   clinical↔watch pairing produces the 0.807 — +0.101 over shuffled, unanimous,
   p=3e-10. The encoder is genuinely exploiting the fact that the same heart recorded
   two ways must map to the same features. **That is modality invariance, learned
   from real paired hardware, and it is the mechanism.**

## Why this is the project's headline result
This is the first method that (a) beats the calibration champion by a wide,
unanimous margin, (b) uses real paired hardware as the invariance signal, and (c)
survives a falsification control that killed the two boring explanations. It directly
delivers the north-star: **abundant clinical data + a modest unlabeled real-paired set
→ a modality-invariant encoder that transfers to real single-lead with zero target
disease labels**, reaching 0.807 (0.820 with calibration stacked, E51), closing ~52%
of the clean→oracle(0.93) gap.

The E48→E50 negatives were essential: they identified the exact failure mode
(unanchored invariance destroys pathology content) whose fix (the CE label-anchor in
the *same* step) is what makes E51 work. Alignment without the anchor destroys signal
(E50); the anchor without real correspondence gives only calibration (selfclin);
both together, on real pairs, is the win.

## Honesty flags (carried forward — still real)
- **Three devices:** the invariance is learned between SJLIFE's clinical+wrist and
  *tested* on CinC dry-finger. Real Apple Watch wrist is the true target and still has
  no public disease labels → the AW number remains a (now much stronger) prediction,
  not a measurement.
- **AF/rhythm task only.** E47 showed morphology doesn't transfer; expect this win to
  be rhythm-scoped too (untested for E51 — a clear next experiment).
- λ=0.1 and temp=0.1 fixed a-priori (not tuned — so likely not even optimal).
- SJLIFE n=243; single clinical train set; 20 seeds, single architecture.
- Effect shown on a proxy (CinC), not real labeled AW; the mechanism is real, the
  absolute AW AUROC is not yet measured.
