# E62 — Amplitude/baseline-invariant handcrafted features (N5)

**Branch:** `explore/novel-methods` · **Idea:** N5 · **Date:** 2026-07-27 · **Seeds:** 20 · **Test:** real CinC 2017 AF vs NORM

## Question
The modality gap is (a) baseline wander + (b) ~8× amplitude scaling. A feature that is
**amplitude- and baseline-invariant by construction** cannot be corrupted by either — so
it should transfer clinical→watch with zero modality leakage. Test a gradient-boosted
classifier on such features (RR-interval dynamics, normalized spectral shape, Hilbert
instantaneous-phase), alone and ensembled with the deep model, against the E51 reference bar.

**Invariance is exact, not approximate:** the 14-dim feature vector is bit-identical under
an 8.25× gain + baseline offset (the precise SJLIFE modality transform) — verified, max diff 0.0.

## Result (AUROC on real CinC, 20 seeds)

| Arm | AUROC | vs clean | vs calibration |
|-----|-------|----------|----------------|
| clean (deep Lead-I) | 0.701 ± 0.048 | — | — |
| closed_aug (calibration) | 0.742 ± 0.047 | +0.041 | — |
| **handcrafted (invariant features)** | **0.908 ± 0.006** | **+0.208, 20/20, p<1e-9** | **+0.166, 20/20** |
| ensemble (deep-clean + handcrafted) | 0.883 ± 0.024 | +0.182, 20/20 | — |
| ensemble_cal (deep-calib + handcrafted) | 0.890 ± 0.024 | — | +0.148, 20/20 |

Also beats the paired-alignment ceiling (E51, 0.807) — by a lot.

## Verdict — ✅ HUGE WIN, but ⚠️⚠️ TASK-SPECIFIC (AF is a timing task; do not over-generalize)
Two findings, and the caveat is as important as the result:

**1. Invariant-by-construction features dodge the modality gap entirely — on THIS task.**
handcrafted 0.908 vs deep-clean 0.701 (+0.208, unanimous, p<1e-9), with **variance collapsed
8× (0.006 vs 0.048)**. A model that literally cannot perceive amplitude or baseline is immune
to the two dominant modality-shift axes, so it transfers almost losslessly. That is a real,
mechanistically clean demonstration of the invariance principle.

**⚠️ BUT the margin is task-specific and near-tautological.** Feature importance is **78% CV
of RR-intervals** — and AF *is defined by* RR irregularity. RR-interval variability is a
near-direct readout of the label, and it happens to be pure timing (zero amplitude/baseline
dependence). So the 0.908 is "detect irregular rhythm with an irregularity feature" — it does
**not** demonstrate that invariant features are generally superior. On a **morphology** task
(where the discriminative signal lives in wave *shape/amplitude*, e.g. E53's N-vs-O, MI,
hypertrophy), amplitude-invariant timing features would carry little signal and this margin
should shrink or vanish. This is the E53/E47 lesson mirrored: methods that win on rhythm can
be null on morphology. **Claiming a general win here would be overreach.**

**2. The deep model is NOT complementary — it's a liability here.**
The "handcrafted complementary to learned" hypothesis (SignalMC-MED F6) **fails**:
ensemble 0.883 < handcrafted-alone 0.908 (−0.025, p=0.0005, deep model *drags it down*).
The modality-corrupted deep predictions add noise, not signal, to an already-near-ceiling
timing classifier. Ensembling only helps when both members are comparably good; here the
timing features so dominate that mixing in the weaker, modality-sensitive deep model hurts.
(ensemble_cal 0.890 > ensemble 0.883 — a better deep member hurts less, but still < handcrafted.)

## What this genuinely contributes (beyond the AF giveaway)
- **A design principle, validated:** for any AW target task whose signal is
  **timing/rhythm-based**, amplitude/baseline-invariant handcrafted features are the single
  strongest, cheapest, most modality-robust route — no watch data, no paired set, no deep
  model, 8× lower variance. This is a real deployment recommendation for the rhythm family.
- **A boundary, mapped:** the win is confined to the timing family. It sets up the obvious
  follow-up — run the identical handcrafted pipeline on a **morphology** task (E53's N-vs-O)
  to confirm the margin collapses there. That falsification test is what separates "invariance
  principle" from "AF giveaway," and it's the honest next step before this goes in any writeup.

## Consequence for the goal
Adds the strongest lever yet **for rhythm-family AW tasks specifically**, and cleanly
demonstrates the invariance-by-construction principle. Does NOT replace the deep/alignment
stack for morphology tasks (sex E60, MI, hypertrophy), where amplitude carries the signal and
these features go quiet. Two-track deployment picture: timing tasks → invariant handcrafted
features; morphology tasks → deep model + alignment/calibration.

## Honesty flags
CinC dry-finger ≠ AW wrist; **AF/N is a timing task and RR-irregularity is near-tautological
for AF — the 0.908 does NOT generalize to morphology tasks**; R-peak detection imperfect on
noisy real watch signal (untested here — CinC is cleaner); GB slightly optimistic on 14 features;
single clinical train set; 20 seeds. Morphology-task control (E62b, N-vs-O) REQUIRED before any
general claim.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/62_invariant_features && \
  python3 experiments/62_invariant_features.py > results/62_invariant_features/run.log 2>&1
```
Figure: `results/62_invariant_features/invariant_features.png`
