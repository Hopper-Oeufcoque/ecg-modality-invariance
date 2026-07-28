# E53 — Does label-anchored alignment break the RHYTHM-only boundary?

**Date:** 2026-07-27
**Script:** `experiments/53_align_morphology.py`
**Seeds:** 20 · **Task:** morphological Normal-vs-Other (E47's harder task)
**Train:** PTB-XL Lead-I NORM vs morphological-abnormal (MI/STTC/CD/HYP, AF excluded)
**Test:** real CinC 2017 N-vs-O · λ=0.1, temp=0.1

## Hypothesis
E47 established that closed-loop **calibration is rhythm-specific**: it lifts AF
+0.041 but is null (−0.002) on a morphological task, because it only perturbs the
out-of-band baseline-wander axis and leaves in-band P/QRS/T morphology untouched.
E51's label-anchored **alignment** is our confirmed headline win — but only tested
on AF. THE generality question: does alignment help morphology too (→ a far more
general mechanism than calibration), or does it share the same rhythm-only wall?

## Results (AUROC real CinC N-vs-O, 20 seeds)

| Arm | AUROC | Δ vs clean | note |
|---|---|---|---|
| clean | 0.561 ± 0.022 | — | near-chance (morphology is hard on single-lead) |
| closed_aug (calibration) | 0.558 ± 0.017 | **−0.002** (p=0.67) | **null — reproduces E47** |
| **joint (alignment)** | **0.594 ± 0.018** | **+0.034** (20/20, p=9e-9) | **significant lift where calibration fails** |
| oracle (train-on-real) | 0.750 ± 0.021 | +0.189 | ceiling; morphology intrinsically hard |

**AF (E51) vs morphology (E53), joint − clean:** +0.106 → +0.034.

## Verdict ✅ (alignment is MORE GENERAL than calibration)
Label-anchored alignment **partially breaks the rhythm boundary that stops
calibration cold.** On the exact task where calibration is null (−0.002), alignment
delivers a real, unanimous +0.034 (20/20 seeds, p=9e-9). It closes ~18% of the
clean→oracle gap on morphology — smaller than its +0.106 on AF, but categorically
different from calibration's zero.

## Interpretation — why alignment generalizes and calibration doesn't
Calibration operates in **input space** on a single nuisance axis (baseline-wander,
out-of-band), so it can only help decisions that live in the bands it touches →
rhythm/HRV. Alignment operates in **feature space**: the same-patient clinical↔watch
InfoNCE term forces the encoder to represent *the same cardiac content identically
across modalities*, whatever band that content occupies. That includes in-band
morphology — so alignment reaches decisions calibration structurally cannot.

The lift is smaller on morphology (+0.034 vs +0.106) for two compounding reasons,
both in the honesty flags: (1) single-lead morphological diagnosis is intrinsically
weak (oracle only 0.750 vs AF's 0.93), so there is less transferable signal to
recover; (2) the CinC "O" catch-all is a heterogeneous, noisily-labeled target that
does not cleanly match PTB-XL's abnormal taxonomy. Both shrink the measurable
ceiling; the fact that a significant lift survives them is the strong reading.

## Consequence for the north star
The confirmed method (E51/E51b) is **not limited to rhythm** the way calibration is.
It is a general modality-invariance mechanism that transfers clinical training to
real single-lead across both rhythm and (weaker) morphology, zero target disease
labels. This materially broadens the set of Apple-Watch downstream tasks it could
serve — the central goal of the project.

## Honesty flags
- CinC "O" is a heterogeneous catch-all (weak label); PTB-XL abnormal ≠ CinC "O"
  taxonomy — inflates task difficulty and caps the measurable lift.
- Morphology is intrinsically hard on single-lead (oracle only 0.750).
- Three devices: CinC dry-finger ≠ Apple wrist ≠ SJLIFE wrist.
- λ=0.1 / temp=0.1 a-priori (not tuned); single clinical train set; AF excluded from
  clinical training; 20 seeds, single architecture.
