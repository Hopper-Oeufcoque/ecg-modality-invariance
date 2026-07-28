# ECG Modality Invariance

**Goal:** Compile proven and novel methods for using **clinical ECG signals** to build AI models that transfer to **Apple Watch single-lead ECGs** for disease detection/prediction — with a focus on making models **invariant to the recording modality**.

## The Core Problem

Clinical ECGs (12-lead, ~500–1000 Hz, gel electrodes, supine patient) and Apple Watch ECGs (single Lead-I-equivalent, ~512 Hz, dry electrodes, wrist+finger, ~30 s, higher noise, ambulatory) differ in:
- **Lead set / spatial coverage** (12 leads → 1 lead)
- **Electrode physics** (wet Ag/AgCl vs dry stainless steel)
- **Noise profile** (baseline wander, EMG/motion, contact impedance)
- **Sampling rate, bandwidth, filtering, quantization**
- **Population / context** (clinic vs free-living)

A model trained on clinical data typically **degrades sharply** on watch data due to this *domain shift*. We want methods that close that gap.

## Scope of Methods Explored

- Signal processing / preprocessing transforms that normalize across modalities
- Feature engineering robust to lead/modality
- Model architectures (domain-invariant, lead-agnostic, self-supervised)
- Statistical / domain-adaptation methods (DANN, CORAL, etc.)
- Reconstruction / lead-synthesis approaches (12-lead ↔ single-lead)
- Novel / unexplored ideas from adjacent fields (speech, sensor fusion, wearables)

## Structure

- `docs/method_taxonomy.md` — 9 categories (A–I), ~50 methods tagged proven/adjacent/novel
- `docs/synthesis_report.md` — 3 ranked solutions + validation plan
- `docs/EXPERIMENT_LOG.md` — **living lab notebook**: what's been tried, what worked, what didn't (negative results included)
- `docs/FUTURE_APPROACHES.md` — actionable backlog of approaches not yet tried (growing)
- `notes/idea_log.md` — 13 frontier ideas with adjacent-field grounding
- `results/` — per-experiment REPORT.md + metrics.json + figures; **`results/EXPERIMENT_SYNTHESIS.md` is the current headline document** (real-data findings E38–E47)
- `src/` — `aw_generator.py` (**ClosedLoopCalibrator — the reusable tool**), `watch_simulator.py`, `dataset.py`, `model.py` (1D ResNet)
- `experiments/` — numbered, reproducible experiment scripts
- `references/` — 155-paper searchable corpus + priority abstracts

## Status / headline result (2026-07-27, real-data phase)

**Two working levers for zero-label clinical→real-single-lead transfer (AF/rhythm):**

**1. Label-anchored real-paired modality alignment (headline, E51/E51b).** Train the
encoder to classify clinical Lead-I AND align same-patient clinical↔Apple-Watch pairs
(real SJLIFE hardware) in the same step — the classification loss anchors invariance so
it can't destroy pathology. **0.807 alone / 0.820 with calibration (+0.078 vs
calibration, +0.119 vs clean, 20/20 seeds, p<1e-8)** — best zero-real-label transfer to
date, ~52% of the clean→oracle(0.93) gap. A falsification control (E51b) confirms the
gain is **genuine cross-modality invariance**: shuffling the pairs (different-patient
watch) kills it entirely (p=0.76), only the true same-patient correspondence works
(joint−shuffled +0.101, p=3e-10).

**2. Closed-loop modality calibration (input-space, E42).** Measure the unlabeled
target's baseline-wander, binary-search a morphology-preserving augmentation to match
it. **+0.041 AUROC, p=0.009**; worth ~10–15 real labels (E46); gap-proportional (E44).
Lighter-weight; stacks under lever 1.

**Boundaries (honest):** both are **rhythm-specific** — no benefit on morphological
tasks (E47); neither is yet validated on real *labeled* Apple-Watch wrist data (none is
public — the AW number is a strong prediction, not a measurement); shown on the CinC
proxy. The three representation negatives that led here (E48 learned-invariance null, E49
clinical-distillation backfire, E50 unanchored-contrastive collapse) identified the
failure mode — unanchored invariance destroys signal — whose fix is lever 1.

**Tools:** `src/aw_generator.py::ClosedLoopCalibrator` +
`experiments/51_label_anchored_align.py` (joint align+classify). See
`results/EXPERIMENT_SYNTHESIS.md` for the full synthesis and honest limitations,
and `docs/EXPERIMENT_LOG.md` for the complete trial-and-error record (negative
results included).

> **Superseded:** the earlier simulator-phase claim ("lead-masking is the
> decisive winner") was validated only on *simulated* watch data and **failed on
> real single-lead** (E6b, E23). The real-data arc (E37+) is authoritative.

