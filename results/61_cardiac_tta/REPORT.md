# E61 — Per-clip test-time adaptation via cardiac-cycle self-consistency (N3)

**Branch:** `explore/novel-methods` · **Idea:** N3 · **Date:** 2026-07-27 · **Seeds:** 10 · **Test:** real CinC 2017 AF vs NORM

## Question
Realistic deployment = a single unlabeled 30 s watch clip. The beats within one clip
are natural same-label views (one clip = one rhythm), which (a) give a real mini-batch
so single-clip BatchNorm adaptation is well-posed and (b) should predict consistently.
Adapt only BN affine params per clip (reset between clips), driven by prediction entropy
(TENT) + cross-view consistency (KL to batch-mean). Test whether **cardiac-cycle-aware
views** (crops phase-locked to detected R-peaks) beat a **random-circular-shift control**
and the no-adaptation floor.

## Result (AUROC on real CinC, 10 seeds)

| Arm | AUROC | vs clean |
|-----|-------|----------|
| clean (no TTA) | 0.701 ± 0.044 | — |
| closed_aug (calibration, train-time ref) | 0.737 ± 0.054 | +0.036 (9/10, p=0.068) |
| **shift_consist (random-shift TTA)** | **0.678 ± 0.035** | **−0.023 (0/10, p=0.006) HURTS** |
| **beat_consist (cardiac-locked TTA)** | **0.675 ± 0.035** | **−0.027 (0/10, p=0.002) HURTS** |
| beat vs shift | — | Δ−0.003 (p=0.0005, i.e. indistinguishable in mean, tight) |

## Verdict — ❌ NEGATIVE (per-clip TTA hurts; cardiac locking adds nothing)
- **Both TTA variants degrade transfer**, unanimously (0/10 seeds, p<0.01). Label-free
  per-clip adaptation does not help this task — it actively pulls AUROC below the
  no-adaptation floor.
- **Cardiac-cycle locking is irrelevant**: beat_consist ≈ shift_consist (Δ−0.003). The
  ECG-specific framing — the whole novelty of N3 — makes no measurable difference versus
  generic random-shift views. The control did its job and killed the hypothesis cleanly.
- Calibration (train-time) remains the thing that works (+0.036); TTA is not competitive.

## Why it fails (mechanism)
Two compounding problems, both intrinsic to per-clip TTA here:
1. **Tiny-batch BN corruption.** Recomputing BatchNorm statistics from just K=6 views of a
   *single* clip replaces the stable training-set statistics with a noisy per-clip estimate.
   On a 2-class morphological task that's a big variance injection for little bias
   correction — the classic TENT failure mode on small batches / near-single samples.
2. **Entropy minimization sharpens a possibly-wrong prediction.** Confidence/consistency
   losses push all K views toward whatever the clip already leans to; if the clean model
   is wrong on a shifted clip, TTA makes it *more* confidently wrong. No label signal exists
   to correct the direction.
And the reason the cardiac view doesn't rescue it: the beats within a CinC clip, once
R-peak-aligned, are near-duplicates → they don't inject the *diversity* that would make
consistency training informative; they just reinforce the same (possibly wrong) answer.
Random shifts are marginally different but equally uninformative.

## Relation to prior results
- Consistent with **E5** (test-time BN adaptation was marginal/slightly-hurt without shift,
  "finishing move only"). E61 extends that: even a cardiac-structure-aware TTA objective
  doesn't beat it — the problem is the per-clip adaptation regime itself, not the objective.
- Reinforces the project's throughline: gains come from **train-time** use of the modality
  structure (calibration's out-of-band injection, alignment's paired correspondence), not
  from **test-time** self-supervision on a single unlabeled clip. There is no free lunch at
  inference here.

## Consequence for the goal
Kills N3. Test-time adaptation is not a viable lever for this clinical→watch transfer at the
single-clip granularity that real deployment imposes. Effort stays on train-time methods.
Closes the novel-methods exploration batch (N1–N4) — see branch summary.

## Honesty flags
CinC dry-finger ≠ AW wrist; AF/NORM easy; BN-affine-only TTA (didn't try full-model or
prior-anchored TTA); K=6 / 5 steps / lr 5e-3 chosen a priori (a wider TTA-hyperparameter
search *might* find a non-hurting setting, but beat≈shift says cardiac locking wouldn't be
the winning ingredient regardless); circular-shift wrap artifact; 10 seeds.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/61_cardiac_tta && \
  python3 experiments/61_cardiac_tta.py > results/61_cardiac_tta/run.log 2>&1
```
Figure: `results/61_cardiac_tta/cardiac_tta.png`
