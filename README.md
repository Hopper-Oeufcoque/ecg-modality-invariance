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

- `docs/` — synthesis reports, findings, method taxonomy
- `literature/` — annotated papers, per-method notes
- `notes/` — working notes, idea log
- `references/` — extracted data, citation lists, datasets
- `src/` — any code (prototypes, feature extractors)

## Status

Bootstrapped 2026-07-26. Literature survey in progress.
