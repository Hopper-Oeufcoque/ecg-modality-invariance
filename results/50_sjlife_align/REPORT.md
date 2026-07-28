# E50 — Real paired-hardware invariance pretraining (SJLIFE): alignment ≠ useful invariance

**Date:** 2026-07-27
**Script:** `experiments/50_sjlife_align.py`
**Seeds:** 20 · **Pretrain:** InfoNCE on 243 real SJLIFE patients (729 apple windows),
same-patient clinical↔watch = positive · **temp:** 0.1 · InfoNCE 4.01 → 0.63 (converged)

## Hypothesis
E49 taught that injected information must be **watch-anchored / modality-invariant**,
not clinical. Our only real paired asset is SJLIFE (243 patients recorded on BOTH
clinical 12-lead and Apple Watch). Both clinical-Lead-I and Apple are single-lead →
one shared encoder. Contrastively pretrain that encoder so the same patient's two
modalities map to the same point (label-free), learning a **real-hardware
modality-invariant** feature space, then fine-tune an AF/NORM head on PTB-XL and test
on real CinC. Prediction: real-paired invariance should transfer better than clean
and/or stack with calibration.

## Setup
Arms (test real CinC AF/N, 20 seeds; SJLIFE-pretrained encoder shared across seeds):
1. **clean** — from-scratch on clean Lead-I (floor).
2. **closed_aug** — E42 winner (calibrated Lead-I).
3. **sjlife_ft** — SJLIFE-pretrained init → fine-tune on clean Lead-I.
4. **sjlife_ft_aug** — SJLIFE-pretrained init → fine-tune on calibrated Lead-I.

## Results (AUROC real CinC, 20 seeds)

| Arm | AUROC | Δ vs clean | Δ vs aug |
|---|---|---|---|
| clean | 0.701 ± 0.048 | — | −0.041 |
| closed_aug | **0.742 ± 0.047** | +0.041 (p=0.009) | — |
| sjlife_ft | 0.669 ± 0.039 | **−0.032** (p=0.029) | **−0.073** (p=2e-5) |
| sjlife_ft_aug | 0.735 ± 0.040 | +0.034 (p=0.047) | −0.007 (p=0.51, n.s.) |

## Verdict ❌❌ (negative — with a sharp mechanistic lesson)
Real-paired contrastive pretraining **hurts**: sjlife_ft lands below the clean floor
(−0.032) and far below augmentation (−0.073, p=2e-5). Adding calibration on top
(sjlife_ft_aug) recovers to ≈ augmentation but does **not** beat it (−0.007, n.s.) —
the pretraining contributes nothing net once you augment.

## Interpretation — alignment was achieved, but it destroyed the wrong thing
InfoNCE **converged** (4.01 → 0.63): the encoder genuinely learned to map the same
patient's clinical and watch signals together. Yet transfer got worse. This is the
classic **invariance-by-information-destruction** trap:

- With only **243 patients** and a trivial "align the same patient's two views"
  objective, the cheapest way to satisfy the loss is to collapse onto low-level /
  patient-identity features that happen to match across modalities — **discarding the
  pathology-relevant morphology** the downstream AF task needs.
- You can *always* make features modality-invariant by making them uninformative.
  E50 did exactly that: it optimized invariance at the cost of discriminative content.
- `sjlife_ft_aug ≈ aug` confirms it: once calibration-augmented supervised training
  runs, it overwrites the pretrained init, so the pretraining adds zero.

**Crucially, this does NOT refute paired-hardware invariance in principle** — it shows
naïve label-free contrastive alignment on a tiny paired set is the wrong recipe. The
signal we want (cross-modality invariance) and the signal we destroyed (pathology
morphology) were traded off because nothing in the objective protected the latter.

## The bigger picture across E48–E50
Three different representation-level strategies — explicit invariance loss (E48),
clinical distillation (E49), real-paired contrastive pretraining (E50) — **all fail to
beat plain closed-loop calibration augmentation (+0.041).** The augmentation remains
the champion. The consistent lesson: on single-lead rhythm transfer, *representation
engineering that isn't anchored to the label trades away discriminative content*. The
robust wins come from (a) input-space calibration that leaves morphology untouched
(E42), and (b) real labels (E46). Representation methods need a **label-preserving
constraint** to help — which is the next design (E51: supervised-contrastive /
label-anchored alignment, or joint pretrain+classify so morphology is protected).

## Honesty flags
- SJLIFE has NO disease labels — label-free pretrain is the correct/only option, but
  it also means nothing protected pathology content during alignment.
- **Three devices**: CinC dry-finger ≠ Apple wrist ≠ SJLIFE wrist — the pretrained
  invariance is between SJLIFE's two devices, not CinC's.
- n=243 is small for contrastive learning; InfoNCE temp=0.1 fixed a-priori.
- AF/N easy task; single clinical train set; 20 seeds, single architecture.
