# Synthesis Report: Clinical ECG → Apple Watch Modality Invariance

**Date:** 2026-07-26 · **Author:** Hopper · **Repo:** github.com/Hopper-Oeufcoque/ecg-modality-invariance

## 1. Problem, precisely

A model trained on clinical 12-lead ECG degrades on Apple Watch single-lead ECG because of **domain shift across five axes**:

1. **Lead count / spatial coverage** — 12 leads → 1 Lead-I-equivalent.
2. **Electrode physics** — wet Ag/AgCl (low impedance, stable) → dry stainless-steel wrist+finger (high, variable impedance).
3. **Noise profile** — clinic (quiet, supine) → ambulatory (motion, EMG, contact artifacts, baseline wander).
4. **Sampling / bandwidth / quantization** — ~500–1000 Hz, wideband → 512 Hz, Apple's fixed ~0.3–40 Hz bandpass + ADC.
5. **Population / context** — selected clinic patients → free-living general users.

The goal is methods that make a model **invariant to the recording modality** so a clinical-trained detector/predictor works on the watch.

## 2. What the literature has proven

**(a) Lead synthesis / reconstruction is mature but has a trap.** GAN (ECGNet `2310.03753`), VAE (WearECG `2510.11442`), MAE (MCMA `2407.11481`, *npj Cardiovasc Health*), U-Net (`2502.00559`), diffusion (`2301.08227`), and RF-with-history (`1811.08035`) all reconstruct 12-lead from reduced leads at decent fidelity. **But SelfMIS (`2509.19397`) shows signal-level reconstruction leaves a *latent-space gap* that degrades downstream detection.** → *Prefer latent-space alignment over signal synthesis for the detection task.*

**(b) Self-supervised foundation models generalize across datasets and leads.** CPC (`2103.12676`), SwAV/BYOL (`2304.06427`), JEPA (`2410.08559`), ECG-FM (`2408.05178`), CLEF (`2512.02180`), CoRe-ECG (`2604.11359`), xECG (`2509.10151`). Scaling study (`2605.12241`) — CPC best, state-space > transformer. **CLEF is the direct evidence: pretrained on 12-lead, tested on Lead-I → +2.6% AUROC.** K-MERL (`2502.17900`) does arbitrary-lead masking → +16% AUC on partial-lead zero-shot.

**(c) Direct wearable disease detection is real.** AnyECG-Echo (`2606.09332`) — contrastive single-lead↔echo reports, detects 13 structural-heart subtypes, AUROC 0.87–0.93, *external* cohort n=16,621. RawECGNet (`2401.05411`) — raw single-lead AFib, generalizes across geography/ethnicity/lead-position, F1 0.91–0.94. CogAdapt (`2605.22774`) — adapts clinical FM to wearable via LeadBridge adapter + progressive fine-tune, +11–16 pp.

**(d) Domain adaptation/generalization tooling exists for biosignals.** DG benchmark for ECG/EEG (`2303.11338`); biosignal DG (`2011.06207`); zero-shot sparse DA (`2207.07089`); personalized aug+DA (`2502.04973`).

**(e) Physiologic / hand-crafted features are modality-invariant by construction and complementary to FMs.** SignalMC-MED (`2603.09940`) — hand-crafted ECG features give a strong baseline *and* complementary value when concatenated with FM embeddings. RR-interval AFib (`2302.07648`); motif/DTW signatures (`2606.00107`).

## 3. The five axes — which methods address which

| Shift axis | Strongest proven methods | Strongest frontier |
|---|---|---|
| Lead count | C9 K-MERL lead-masking; E1 modally-reduced; E5 LeadBridge; B9 latent alignment | I1 learnable VCG; E3 set-invariant nets |
| Electrode physics | A3/A4 normalization; A5 noise aug | I2 AdaIN; I6 disentanglement |
| Noise | A1/A5/A6/A7; F7 raw robust | I10 physics sim; I4 scattering |
| Sampling/bandwidth | A2 matched filter; A1 resample | I10 physics sim |
| Population/context | C1–C7 SSL/FM; D6 DG; F7 raw | I5 IRM; I7 OT; I8 TTA |

## 4. Recommended solutions (ranked)

### Solution 1 — "Proven-stack pipeline" (lowest risk, mostly published)
> Clinical FM (CPC-pretrained or CLEF) → **LeadBridge learnable adapter** (watch→FM input space) → **progressive fine-tune** (limit drift) → classifier head, with **FM-embedding ⊕ hand-crafted physiologic features** (RR, morphology, HRV) → **test-time BN adaptation** per clip.

- Why: every component is individually validated; CogAdapt (`2605.22774`) + SignalMC-MED (`2603.09940`) + TTA (`2303.15361`) combine cleanly. Modality gap closed at adapter + feature + test-time layers.
- Risk: still needs *some* labeled or aligned watch data for the adapter.

### Solution 2 — "Invariance-by-construction" (novel, high upside)
> **Forward-physics watch simulator** (clinical 12-lead → Lead-I + Apple transfer function + noise, domain-randomized) generates unlimited labeled pseudo-watch data → train/finetune with **IRM/REx across environments** {clinical sites, sim-watch variants} → **modality-direction scrubbing** of FM embeddings → **per-clip TTA**.

- Why: closes bandwidth/noise/electrode axes *by construction* (simulator); IRM removes residual device shortcuts principled-ly; scrubbing + TTA are cheap and label-free.
- Novelty: combines F10+F5+F11+F8 — not found as a unified pipeline in the literature.
- Risk: simulator realism; IRM fragility (mitigate with REx/DRO).

### Solution 3 — "Label-free cross-modal" (no ECG labels needed)
> **Echo-report / clinical-text contrastive grounding** (AnyECG-Echo / K-MERL style) on clinical data → **latent-space lead alignment** (self-cutting, SelfMIS) → feed **scattering + speech-channel-robust features** as modality-tolerant inputs → zero/few-shot watch inference.

- Why: bypasses ECG-label scarcity entirely; grounding in physiology/semantics is naturally modality-agnostic.
- Risk: needs paired ECG-echo/text data; morphology detail loss (mitigate with feature ensemble).

### Cross-cutting "always-do" baselines
- Match Apple's bandpass/resampling on clinical data (A2).
- Per-record + per-channel normalization (A3/A4).
- Beat-aligned segmentation + signal-quality gating (A1/A7).
- Augmentation: lead-dropout, time-warp, colored/motion noise, mixup (A5/A6).

## 5. Concrete next steps to validate
1. Pick a clinical 12-lead dataset with labels (PTB-XL / CODE-15 / MIMIC-IV-ECG) and a single-lead target (PhysioNet single-lead / any watch set).
2. Build the forward-physics watch simulator (F10) and validate that a clinical model's accuracy drop on *real* watch data is reproduced on *simulated* watch data (sanity check the simulator).
3. Implement Solution 1 as the baseline; measure AUROC drop clinical→watch and the recovery from each added layer (adapter / FT / features / TTA).
4. Layer Solution 2 components on top; ablate IRM/scrubbing/TTA.
5. Report with the user's standard rigor: external cohorts, placebo/bootstrap where applicable, flag data-quality issues.

## 6. Caveats & data-quality flags
- Most "watch ECG" public datasets are small and AFib-focused; broad disease detection on real Apple Watch data is under-published (AnyECG-Echo is the standout). Expect limited held-out watch validation → lean on simulator + external clinical cohorts.
- Lead-synthesis quality is usually reported as signal fidelity (MSE/correlation), which **does not** imply downstream detection gains (SelfMIS warning). Always evaluate at the *diagnostic* level (ECGGenEval in `2407.11481` does this correctly).
- Foundation models trained predominantly on 12-lead may encode a "lead-count prior"; lead-masking (K-MERL) and channel-agnostic encoders (E1) are the proven mitigations.

## 7. Key references (verified arXiv IDs)
ECG core: 2405.19359, 2606.09332, 2509.19397, 2502.17900, 2605.22774, 2512.02180, 2408.05178, 2103.12676, 2304.06427, 2410.08559, 2407.11481, 2310.03753, 2510.11442, 2502.00559, 2410.13528, 1811.08035, 2301.08227, 2409.17833, 2604.11359, 2603.17248, 2602.04279, 2303.11338, 2011.06207, 2207.07089, 2502.04973, 2401.05411, 2606.00107, 2012.00110, 2605.12241, 2503.00711, 2509.10151, 2603.09940, 2310.09203, 2101.02362, 2603.15539, 2509.12991, 2301.12178, 1510.02541, 2307.02806, 2302.07648, 2211.02678, 2008.07263, 2410.18094.
Adjacent: 1907.02893 (IRM), 1602.07576 (G-CNN), 1703.06868 (AdaIN), 2101.00588 (Style-Norm DG), 2112.10474 (Reciprocal Norm), 1304.6763 / 1807.08869 / 1512.02125 (scattering), 1705.08848 / 1803.10081 (OT-DA), 2303.15361 / 2402.09604 / 2506.23424 / 2406.04295 (TTA), 2007.04964 / 2108.04441 / 2302.09795 (disentanglement), 2002.03562 / 2210.00721 / 2411.03085 (speech channel robustness), 2304.01973 (ERM++).
