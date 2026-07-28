# E47 — Harder morphological task: calibration is RHYTHM-SPECIFIC (and clinical→real barely transfers)

**Date:** 2026-07-27
**Task:** Normal vs Other/abnormal (morphological, NOT rhythm-defined).
Train PTB-XL Lead-I NORM vs morphological-abnormal (MI/STTC/CD/HYP, AF excluded);
test real CinC 2017 N vs O. Calibrate to unlabeled CinC-ref bw. 20 seeds.

## Result

| arm | AUROC (20 seeds) |
|---|---|
| clean | 0.561 ± 0.022 |
| closed_loop | 0.558 ± 0.017 |
| oracle | 0.753 ± 0.015 |

- **closed − clean = −0.002, 8/20, p=0.67 → calibration does NOTHING here.**
- Compare AF/NORM (E42): +0.041, p=0.009.

## Two findings — both important, both honest

### 1. Calibration is RHYTHM-SPECIFIC
The wander-calibration lift that was real and significant on AF (+0.041)
**completely vanishes on the morphological task** (−0.002, n.s.), even though the
modality gap is the same (target bw 0.262, same CinC device). This confirms
E31's warning at the mechanism level: closing the baseline-wander gap helps a
model keep a **rhythm/HRV**-based decision robust across recording modality, but
it does nothing for **morphological** discrimination (P/QRS/T shape), because the
diagnostic signal there lives in the in-band QRS/mid frequencies that calibration
deliberately leaves untouched (to preserve morphology). **The method's benefit is
scoped to rhythm-type tasks.**

### 2. The bigger problem: clinical→real barely transfers AT ALL on this task
Even the clean clinical model is near-chance on real CinC N-vs-O (0.561), and the
**oracle itself is only 0.753** — i.e. this task is just hard on real single-lead
data, and clinical→real transfer is weak regardless of calibration. The
clinical-vs-real label taxonomies also don't align cleanly (PTB-XL morphological
superclasses ≠ CinC's heterogeneous "Other" bucket), so part of the 0.561 is
task-definition mismatch, not just modality.

## Implication for the north star (important, sobering)
"Leverage clinical data for downstream Apple Watch tasks" works **where the
downstream task is rhythm-detectable on a single lead** (AF, and likely other
arrhythmias). For **morphological** targets (ischemia, hypertrophy, structural
disease) two things break: (a) the modality-calibration trick gives no help, and
(b) clinical→single-lead transfer is intrinsically weak even with an oracle. This
matches the clinical literature: single-lead wearables are validated for rhythm
(AF), not for morphological diagnosis. **Do not claim the method generalizes to
arbitrary downstream tasks — it's a rhythm-transfer tool.**

## Honest flags
- CinC "O" is a heterogeneous catch-all (weak, noisy label); PTB-XL abnormal ≠
  CinC "O" taxonomy → approximate task mapping inflates the apparent difficulty.
- A cleaner morphological test (same-taxonomy train/test, e.g. a labeled
  single-lead MI set) would isolate modality-shift from task-mismatch — we don't
  have one. So E47 shows "no calibration benefit + weak transfer" but can't fully
  separate the two causes.
- CinC finger ≠ AW wrist; single clinical train set.

## Verdict ⚠️❌ (scopes the method — negative logged with full weight)
Closed-loop calibration is a **rhythm-task tool**, not a general modality-invariance
fix. On morphological tasks it neither helps nor (crucially) hurts, but baseline
clinical→real transfer is too weak to be useful there anyway. This bounds every
prior positive result (E41/E42/E46) to the rhythm regime.

## Artifacts
`metrics.json`, `harder_task.png`.
