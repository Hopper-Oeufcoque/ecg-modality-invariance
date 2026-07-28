# E51 — Label-anchored modality alignment: the first representation win (CONFIRMED by E51b)

**Date:** 2026-07-27
**Script:** `experiments/51_label_anchored_align.py`
**Seeds:** 20 · **λ (align weight):** 0.1 · **temp:** 0.1 · SJLIFE pairs: 243
**STATUS: CONFIRMED** — E51b mechanism control passed (joint − shuffled = +0.101,
p=3e-10): the gain is genuine same-patient cross-modality invariance, NOT generic
auxiliary-SSL regularization. See `results/51b_align_control/REPORT.md`.

## Hypothesis
E48/E49/E50 all lost to calibration because nothing protected the label signal
while aligning modalities (E50 literally destroyed pathology content to satisfy a
label-free alignment loss). Fix: align modalities and classify **jointly** in the
same optimization step, so the CE term forbids the encoder from collapsing onto
label-destroying invariance. Objective per step: CE(PTB-XL Lead-I) + λ·InfoNCE(real
SJLIFE clinical↔watch pairs), shared single-lead encoder.

## Results (AUROC real CinC, 20 seeds)

| Arm | AUROC | Δ vs clean | Δ vs aug |
|---|---|---|---|
| clean | 0.701 ± 0.048 | — | −0.041 |
| closed_aug (E42 champ) | 0.742 ± 0.047 | +0.041 | — |
| **joint** | **0.807 ± 0.023** | **+0.106** (20/20, p=5e-8) | **+0.065** (20/20, p=8e-6) |
| **joint_aug** | **0.820 ± 0.020** | **+0.119** (20/20, p=5e-9) | **+0.078** (20/20, p=9e-7) |

First representation-level method to **beat closed-loop calibration** — and by a
wide, unanimous margin (20/20 seeds on every comparison). joint_aug 0.820 is the
best clinical→real-single-lead transfer we've achieved without real target labels,
closing ~52% of the clean→oracle(0.93) gap vs calibration's ~18%.

## ⚠️ Why this is PROVISIONAL — the control that must pass first (E51b)
Two features demand skepticism before I call this a genuine invariance result:
1. **Variance collapse** (std 0.048 → 0.023) alongside a big mean gain is the classic
   footprint of a strong **regularizer**, not necessarily of learned invariance.
2. **Confound:** the gain could come from *same-patient modality alignment* (the
   invariance claim) OR merely from bolting **any** auxiliary InfoNCE loss on extra
   real ECG data (generic SSL regularization — a much weaker, less interesting claim).

**E51b (running)** decomposes this with matched controls (identical loop, only the
alignment positive-pair definition changes):
- `joint_shuffled` — pair clinical with a *different* patient's watch (destroys
  same-patient correspondence, keeps the aux real-ECG InfoNCE).
- `joint_selfclin` — align clinical to its *own* calibrated view (no SJLIFE watch at all).

Decision rule:
- If **joint ≈ joint_shuffled** → the effect is generic aux-ECG regularization; the
  "modality invariance" story is FALSE and must be retracted to "auxiliary SSL helps."
- If **joint > joint_shuffled** → same-patient modality correspondence genuinely
  matters → real invariance win.
- If **joint_selfclin** captures most of it → you don't even need SJLIFE; augmentation-
  consistency regularization suffices (cheapest deployable recipe).

**No headline claim about invariance until E51b returns.** This report will be updated
with the verdict.

## Honesty flags
- SJLIFE align term is label-free (no disease labels exist) but now gated by the CE anchor.
- Three devices: CinC dry-finger ≠ Apple wrist ≠ SJLIFE wrist — alignment learned is
  SJLIFE-internal, transfer tested on CinC.
- λ=0.1 fixed a-priori, not tuned; AF/N easy task; single clinical train set; 20 seeds.
- Provisional pending E51b mechanism control (see above).
