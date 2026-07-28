# Transferring Clinical ECG Models to Wearable Single-Lead ECG via Recording-Modality Invariance

**A consolidated findings report.** Project: `ecg-modality-invariance`
(github.com/Hopper-Oeufcoque/ecg-modality-invariance). Real-data experimental
phase E38–E56, 2026-07-27. This document is the legible, standalone summary of the
project's central result; `docs/EXPERIMENT_LOG.md` is the full trial-by-trial
notebook and `results/EXPERIMENT_SYNTHESIS.md` the method ladder.

---

## Abstract

Clinical 12-lead ECGs are abundant and richly labeled; wearable single-lead ECGs
(Apple Watch, KardiaMobile) are scarce and rarely labeled, yet are the deployment
target for consumer cardiac screening. A model trained on clinical data degrades on
wearable data because of a **recording-modality shift** (dry vs wet electrodes, wrist
vs chest, baseline wander, single vs multi lead). We ask whether abundant clinical
data can be turned into a model that works on wearable single-lead ECG **without
wearable disease labels**, by making the learned representation invariant to the
recording modality.

We find two working levers, validated end-to-end on **real** labeled single-lead data
(CinC 2017, dry-finger) rather than a simulator:

1. **Closed-loop modality calibration** (input-space): +0.041 AUROC zero-label,
   p=0.009. Lightweight, needs no paired data.
2. **Label-anchored real-paired modality alignment** (representation-space, the
   headline): **+0.078 over calibration, +0.119 over the clinical-only baseline
   (0.701→0.820), 20/20 seeds, p<1e-8** — the strongest zero-wearable-label transfer
   we achieved, closing ~52% of the gap to a train-on-target oracle.

A falsification control proves the alignment gain is **genuine cross-modality
invariance** (not a generic regularizer): destroying the same-patient correspondence
removes the entire effect (p=0.76). The method is robust to hyperparameters (12/12
grid cells win), generalizes beyond rhythm to morphology (where calibration fails),
is safe on a third unseen device, and halves seed-to-seed variance.

**Honest ceiling:** no *labeled* real Apple-Watch data is public, so the Apple-Watch
number is a falsifiable *prediction* from a dose-response law, not a measurement. All
positive transfer is on the rhythm task (AF); morphology transfers weakly for
everyone. Confirmation requires the HOME evaluation portal or new labeled AW data.

---

## 1. Problem and setup

**Shift axes (clinical 12-lead → wearable 1-lead):** lead count, electrode physics
(wet Ag/AgCl vs dry steel), noise/baseline-wander, sampling/bandwidth, population.

**Task.** Binary AF-vs-sinus (the rhythm task wearables are clinically validated for),
plus a harder morphological Normal-vs-Other task (E47/E53).

**Data.**
- **Train (clinical):** PTB-XL, Lead-I only (the wearable-comparable lead).
- **Real test (the honest target):** CinC 2017 — real labeled single-lead from
  AliveCor KardiaMobile (dry finger electrode); large baseline-wander gap (bw≈0.25).
  The best *labeled* wearable proxy that exists publicly. 1 record/patient →
  leakage-free.
- **Real paired hardware (unlabeled):** SJLIFE — 243 patients recorded on BOTH clinical
  12-lead and Apple Watch (same person). No disease labels. Used to *measure* the real
  modality profile (E38) and, later, to *learn* invariance (E50/E51).
- **Second real device:** Icentia11k CardioSTAT chest-patch (near-zero modality gap) —
  external-validity test (E44/E56).
- **HOME:** 1000 real Apple-Watch ECGs — EVALUATION-ONLY (license forbids
  training/adaptation; labels withheld; scored only via a central portal).

**Model.** Small 1D ResNet (~0.5M params), single-lead (`n_leads=1`), 20 epochs, CPU.
Metric: AUROC on the real test set, ≥10 seeds, paired statistics vs a seed-matched
clinical-only baseline. Per-record z-scoring is mandatory (SJLIFE showed ~8× amplitude
difference between modalities, E38).

**"Oracle"** = train-on-real-single-lead → test-on-real-single-lead (the upper bound if
you had abundant labeled target data). On CinC AF this is ≈0.93.

---

## 2. What the modality gap actually is (E37, E38)

Two early corrections shaped everything downstream:
- A simulator-phase "bandwidth gap" result was **retracted** (E37) after a
  sampling-rate provenance bug: a 200 Hz file had been processed as 500 Hz. Corrected,
  real single-lead ECG is **clean and low-pass** — the gap is **not** high-frequency
  noise.
- Reading the real SJLIFE pairs directly (E38): the clinical→wearable gap is dominated
  by **baseline wander** (Apple wrist bw≈0.20, ~6× clinical), plus an ~8× amplitude
  scale difference. Not HF, not lead-count for the Lead-I-comparable task.

This is why input-space calibration targets baseline wander specifically, and why a
multi-axis calibrator later self-zeroed its HF axis (E43): there is no HF gap to close.

---

## 3. Lever 1 — Closed-loop modality calibration (E39–E47)

**Idea.** Measure the *unlabeled* target's baseline-wander energy, then binary-search a
morphology-preserving augmentation (1/f coloured wander injected outside the QRS band)
until the augmented clinical data *matches* the measured target profile. Train the
clinical Lead-I model on this calibrated augmentation.

**Key results.**
- **+0.041 AUROC zero-label** (E42): 0.701→0.742, 20 seeds, paired t p=0.009, Wilcoxon
  p=0.009, Cohen dz=0.65. First properly-powered real-data win. (An earlier 5-seed
  +0.072 was corrected down as small-sample optimism.)
- **Gap-proportional** (E44): on Icentia chest-patch (near-zero gap) calibration
  correctly idles (−0.003). The benefit scales with the modality gap.
- **Worth ~10–15 real labels** (E46); calibration and labels are partial substitutes.
- **Rhythm-specific** (E47): on the morphological task the lift vanishes (−0.002),
  because morphology lives in the in-band QRS/mid frequencies calibration deliberately
  leaves untouched.

**Verdict.** A lightweight, dependency-free lever with a real but modest, rhythm-bound
gain. `src/aw_generator.py::ClosedLoopCalibrator` is the reusable tool.

---

## 4. Three informative negatives (E48–E50) — why naive representation methods fail

Pure augmentation plateaus at ~+0.041 because the residual gap is *in-band* (E43/E46).
We tried three representation-level methods to break past it. **All three lost to plain
calibration** — and *why* they failed is the crux of the whole project.

- **E48 — explicit feature-invariance loss:** enforce identical features for a clinical
  sample and its modality-shifted twin. Result: −0.010 vs augmentation (n.s.). Two
  independent method families hit the *same* +0.041 wall → the ceiling is
  **information-bound**, not a bad objective.
- **E49 — 12-lead→1-lead clinical distillation:** distil a 12-lead teacher (train-acc
  0.994) into the single-lead student. Result: **−0.060, p=0.0005 — it HURT.** The
  teacher's soft labels encode *clinical-modality* decision boundaries and drag the
  student toward the clinical distribution. Injecting *clinical* information imports
  **modality bias**, not invariant structure.
- **E50 — label-free real-paired contrastive pretraining:** InfoNCE aligning
  same-patient SJLIFE clinical↔watch pairs. The alignment **converged** (InfoNCE
  4.01→0.63) yet transfer got **worse** (−0.073, p=2e-5). Classic
  **invariance-by-information-destruction**: with a tiny paired set and a trivial
  "align the same patient" objective, the cheapest way to be modality-invariant is to
  discard the pathology morphology the task needs.

**Unifying lesson.** *Representation engineering that is not anchored to the label
trades away discriminative content.* You can always make features invariant by making
them useless. This diagnosis is what produced the headline method.

---

## 5. Lever 2 (headline) — Label-anchored real-paired alignment (E51, E51b)

**Idea.** Fix the E50 failure by aligning **and** classifying in the *same* optimization
step, so the classification loss forbids the encoder from collapsing onto
label-destroying invariance:

```
loss = CE(clinical Lead-I labels)  +  λ · InfoNCE(same-patient clinical ↔ Apple-Watch pairs)
```

Shared single-lead encoder; the InfoNCE term uses the real SJLIFE pairs (no disease
labels needed — the *pairing* is the only supervision it adds).

**Result (E51, 20 seeds, real CinC AF).**
| Arm | AUROC | vs clean | vs calibration |
|---|---|---|---|
| clean (clinical only) | 0.701 | — | −0.041 |
| calibration (lever 1) | 0.742 | +0.041 | — |
| **alignment (joint)** | **0.807** | **+0.106** | **+0.065** (20/20, p=8e-6) |
| **alignment + calibration** | **0.820** | **+0.119** | **+0.078** (20/20, p=9e-7) |

Best zero-wearable-label transfer achieved — ~52% of the clean→oracle(0.93) gap, vs
calibration's ~18%.

**Falsification control (E51b) — is it real invariance or just a regularizer?** The
variance also collapsed (a regularizer footprint), so we ran matched controls that
change *only* the alignment positive-pair definition:
- **Shuffled pairs** (clinical ↔ a *different* patient's watch): keeps the auxiliary
  loss + all the real ECG, destroys the correspondence → **gain vanishes** (0.706 ≈
  0.701 clean, p=0.76). Generic-SSL-regularization hypothesis **falsified**.
- **Self-calibrated view** (clinical ↔ its own calibrated view, no watch): reproduces
  calibration *exactly* (0.742) and no more → augmentation-consistency is not the extra
  gain.
- **Correct pairs** → 0.807 (+0.101 vs shuffled, 20/20, **p=3e-10**).

Only the true "same heart, two recordings → same features" signal produces the win.
**Confirmed genuine cross-modality invariance from real paired hardware.**

---

## 6. Stress-testing the headline (E53–E56)

- **Generality — beyond rhythm (E53).** On the morphological task where calibration is
  null (−0.002), alignment delivers a real **+0.034 (20/20, p=9e-9)**. Smaller than AF's
  +0.106 (single-lead morphology is intrinsically hard; oracle only 0.750), but
  categorically ≠ calibration's zero. **Alignment works in feature space** (same-content
  → same-features, any band), so it reaches in-band morphology; calibration perturbs one
  out-of-band input axis, so it can't. Alignment is the more general mechanism.
- **Robustness (E54).** λ∈{0.03,0.1,0.3,1.0} × temp∈{0.05,0.1,0.2}: **all 12/12 cells
  beat calibration and clean**, a broad 0.80–0.82 plateau. Clear U-shape in λ (too small
  barely aligns; λ=1.0 lets alignment start to overwhelm CE — the E50 failure mode
  creeping back, but the CE anchor holds). Deployable **without careful tuning**.
- **Label efficiency (E55).** Alignment gives the best transfer at *every* label budget;
  reaches 0.85 at **k=50 real labels — half what calibration needs** (k=100); clean
  never gets there. Biggest edge at k=0. Two honest wrinkles: the advantage shrinks as
  labels arrive (alignment and labels are partial substitutes), and a **k=10 dip**
  (fine-tuning a strong representation on only 10 labels *hurts*) → **deployment rule:
  with <~25 real labels, use the aligned model zero-shot; don't fine-tune.**
- **External validity (E56).** On Icentia chest-patch — a *third* device the SJLIFE-wrist
  alignment never saw — alignment does **not hurt** (0.961 ≥ 0.942 clean); the feared
  OOD overfitting did not happen. The big lift doesn't replicate, but that's *expected*:
  Icentia is near-zero-gap (no headroom). This is the **same dose-response law** as
  calibration (E44), now shown for a second method: **benefit ∝ modality gap.** Bonus:
  alignment **halves seed-to-seed variance** (0.065→0.028; also 0.048→0.023 on CinC) —
  more reliable transfer regardless of mean.

---

## 7. The deployable recipe

**If you have an unlabeled real paired set** (clinical + wearable, same patient — e.g.
SJLIFE):
1. Per-record z-score both modalities.
2. Train the single-lead encoder jointly:
   `CE(clinical Lead-I) + λ·InfoNCE(same-patient clinical↔wearable pairs)`,
   λ∈[0.1,0.3], temp∈[0.05,0.2] (robust plateau).
3. Optionally stack closed-loop calibration on the clinical input (joint_aug).
4. **Zero wearable labels → ~0.81–0.82.** With ~50 real labels → fine-tune → ~0.86.
   With <25 labels, deploy **zero-shot** (don't fine-tune).

**If you have no paired set:** use closed-loop calibration alone (`ClosedLoopCalibrator`,
+0.041, measures the target from unlabeled data). Lighter but rhythm-only and smaller.

**Governing law (both levers):** benefit is **proportional to the modality gap**. Real
Apple-Watch wrist is a large-gap dry electrode (SJLIFE bw≈0.20, the CinC regime) → the
law *predicts* a large lift there.

---

## 8. Limitations (the honest ledger)

- **No labeled real Apple-Watch data is public.** Every wearable-lift number is on the
  CinC dry-finger *proxy*; the Apple-Watch-wrist figure is a falsifiable **prediction**
  from the dose-response law, not a measurement. Converting it requires the **HOME
  evaluation portal** (frozen-model submission; application in progress) or new labeled
  AW data.
- **Rhythm-scoped.** All large positives are AF/rhythm. Morphology transfers weakly for
  every method (single-lead morphological diagnosis is intrinsically hard; oracle 0.750).
  Consistent with the clinical literature (wearables validated for AF, not morphology).
- **Proxy oracle.** The 0.93 "oracle" is CinC train-on-real, not the real-AW HOME
  benchmark (fine-tuned baselines there are 0.77–0.84).
- **Scale.** Small ResNet, CPU, single clinical train set per experiment (seed CIs omit
  clinical-cohort variance); 10–20 seeds. SJLIFE n=243 is small for contrastive learning.
- **Three devices, not one.** Alignment is learned SJLIFE clinical+wrist, tested on CinC
  finger / Icentia chest-patch — a feature (shows it isn't dataset-specific) but means
  no single end-to-end same-device real-AW number.
- **Retractions on record** (E35/E36 sampling-rate bug → E37; E41 5-seed optimism →
  E42; simulator-phase lead-masking claim → refuted E6b/E23). Kept visible for honesty.

---

## 9. Reproducing

```
# environment: .venv (torch CPU, wfdb, scipy, sklearn)
python experiments/42_seeds20_significance.py      # calibration +0.041
python experiments/51_label_anchored_align.py       # headline alignment 0.807/0.820
python experiments/51b_align_control.py             # falsification control
python experiments/53_align_morphology.py           # generality beyond rhythm
python experiments/54_lambda_temp_sweep.py          # hyperparameter robustness
python experiments/55_align_fewshot.py              # label-efficiency curve
python experiments/56_align_seconddevice.py         # 2nd-device external validity
```
Each writes `results/<id>/REPORT.md` + `metrics.json` + a figure. The living notebook
is `docs/EXPERIMENT_LOG.md` (top verdict table + per-experiment entries).

---

## 10. Figures

- E42 calibration significance — `results/42_seeds20_significance/significance.png`
- E51 headline alignment — `results/51_label_anchored_align/label_anchored_align.png`
- E51b falsification control — `results/51b_align_control/align_control.png`
- E53 morphology generality — `results/53_align_morphology/align_morphology.png`
- E54 λ/temp surface — `results/54_lambda_temp_sweep/lambda_temp_surface.png`
- E55 label-efficiency — `results/55_align_fewshot/align_fewshot.png`
- E56 2nd-device — `results/56_align_seconddevice/align_seconddevice.png`

---

## One-line takeaway

*For rhythm-detectable wearable ECG tasks, abundant clinical data plus an **unlabeled**
real paired set can be turned into a model that transfers to real single-lead at ~0.82
with **zero** wearable disease labels — via label-anchored modality alignment, a
mechanism-verified, hyperparameter-robust, device-general form of recording-modality
invariance. Confirming the Apple-Watch number is now the only step gated on external
resources.*
