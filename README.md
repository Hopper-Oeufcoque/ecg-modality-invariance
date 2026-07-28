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

**Closed-loop modality calibration** works for **rhythm-detectable** wearable
tasks, with clear boundaries:

- **Zero-label AF transfer:** +0.041 AUROC (0.701→0.742), p=0.009, 20 seeds, on
  **real** labeled single-lead (CinC 2017). Calibration uses only *unlabeled*
  target data. (E41/E42)
- **Worth ~10–15 real labels**; best minimal-tuning recipe = calibrate + ~50
  labels → 0.855. (E46)
- **Gap-proportional** (helps dry-electrode, idles on clean chest-patch — E44),
  **rhythm-specific** (no benefit on morphological tasks — E47), augmentation
  ceiling ~+0.041 (E43).

**Tool:** `src/aw_generator.py::ClosedLoopCalibrator`. See
`results/EXPERIMENT_SYNTHESIS.md` for the full synthesis and honest limitations,
and `docs/EXPERIMENT_LOG.md` for the complete trial-and-error record (negative
results included).

> **Superseded:** the earlier simulator-phase claim ("lead-masking is the
> decisive winner") was validated only on *simulated* watch data and **failed on
> real single-lead** (E6b, E23). The real-data arc (E37+) is authoritative.

