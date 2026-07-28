# E49 — Multi-lead → single-lead distillation: injecting clinical structure BACKFIRES

**Date:** 2026-07-27
**Script:** `experiments/49_distillation.py`
**Seeds:** 20 · **Teacher:** 12-lead ECGResNet1d, PTB-XL, train-acc 0.994 (frozen)
**KD:** Hinton soft-label, T=3, α=0.5 (a-priori, not tuned)

## Hypothesis
E48 concluded the +0.041 augmentation ceiling is *information-bound* — pivot to
**information injection**. We have abundant clinical data as full 12-lead; the
watch only sees Lead-I. So train a 12-lead teacher (uses all-lead structure) and
distil it into a single-lead student. Prediction: the student inherits multi-lead
structure it could never learn from Lead-I alone → new information the
augmentation ceiling can't reach, ideally stacking with calibration.

## Setup
- Train PTB-XL AFIB-vs-sinus; test **real** held-out CinC AF-vs-N; 20 seeds.
- Teacher: 12-lead, trained once on clinical, frozen, reused across seeds.
- Student (`n_leads=1`) arms:
  1. **clean** — CE on clean Lead-I (floor).
  2. **closed_aug** — E42 winner: closed-loop-calibrated Lead-I.
  3. **distill** — CE(Lead-I) + α·T²·KL(student ‖ 12-lead teacher).
  4. **distill_aug** — distill + calibrated student input (does it stack?).

## Results (AUROC real CinC, 20 seeds)

| Arm | AUROC | Δ vs clean | Δ vs aug |
|---|---|---|---|
| clean | 0.701 ± 0.048 | — | −0.041 |
| closed_aug | **0.742 ± 0.047** | +0.041 (p=0.009) | — |
| distill | 0.682 ± 0.036 | **−0.019** (p=0.041) | **−0.060** (p=0.0005) |
| distill_aug | 0.726 ± 0.037 | +0.025 (p=0.10) | −0.016 (p=0.21, n.s.) |

## Verdict ❌❌ (strong, informative negative)
Distillation from a 12-lead clinical teacher **hurts** real transfer: distill
lands *below the clean floor* (−0.019) and far below the augmentation winner
(−0.060, p=0.0005). Adding calibration on top (distill_aug) rescues most of the
damage but still does **not** recover to pure calibration (−0.016, n.s.). The two
interventions interact **destructively**, not synergistically.

## Interpretation — the *source* of information matters; clinical KD injects modality bias
The naive "inject more information" framing was incomplete. The teacher is trained
on 12-lead **clinical** data (train-acc 0.994 → confident, overfit) so its soft
labels encode **clinical-modality-specific** decision boundaries. Distilling that
"dark knowledge" pulls the student *toward* the clinical distribution — the exact
wrong direction for real-single-lead transfer. We are importing **modality bias**,
not modality-invariant structure. This is why:
- distill alone deepens the modality gap (worse than clean),
- calibration (which drags the student *toward* the watch distribution) fights the
  distillation pull → distill_aug nets below pure calibration.

**This sharpens E48 rather than contradicting it.** The residual gap is not simply
"missing information" you can pour in from anywhere — extra clinical information is
*modality-entangled*, and injecting it makes the model more clinical, not more
invariant. Useful auxiliary information must itself be **modality-invariant or
watch-anchored**, not clinical-distribution-bound.

## Consequence for the north star — pivot to watch-anchored information
Do not distil clinical teachers. The information that helps must be grounded across
*both* modalities. That points squarely at **real paired hardware (SJLIFE, E50)**:
learn alignment from clinical↔watch pairs of the *same patient*, so the shared
structure is modality-invariant by construction. That is the right form of
"information injection."

## Honesty flags
- CinC finger ≠ AW wrist; AF/N is the easy rhythm task.
- KD T=3/α=0.5 fixed a-priori — a tuned α *might* soften the harm, but distill_vs_aug
  = −0.060 (p=0.0005) is a large, robust effect; tuning is unlikely to flip it to a win.
- Teacher trained on the same PTB-XL cohort (no separate clinical holdout) — if
  anything this *favours* distillation (teacher well-matched to student's clinical
  data), and it still hurt.
- 20 seeds, single architecture, single clinical train set.
