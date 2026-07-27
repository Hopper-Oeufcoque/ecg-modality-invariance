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
- `results/` — per-experiment REPORT.md + metrics.json + figures; `EXPERIMENT_SYNTHESIS.md` ties them into a method ladder
- `src/` — `watch_simulator.py` (forward-physics F10), `dataset.py`, `model.py` (1D ResNet)
- `experiments/` — numbered, reproducible experiment scripts
- `references/` — 155-paper searchable corpus + priority abstracts

## Status

Bootstrapped 2026-07-26 (literature survey). Experimental phase 2026-07-27:
forward-physics watch simulator validated; **lead-masking (K-MERL) is the
decisive winner** (closes the lead-count gap from 0.52→0.72 on simulated watch
with zero target-domain labels). See `docs/EXPERIMENT_LOG.md` for the full
trial-and-error record and `results/method_ladder.png` for the ranking.
