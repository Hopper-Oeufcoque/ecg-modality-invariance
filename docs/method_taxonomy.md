# Method Taxonomy: Clinical ECG → Apple Watch Single-Lead Modality Invariance

**Scope:** Methods to make AI models trained on clinical ECG (12-lead, gel, ~500 Hz, clinic) transfer to Apple Watch single-lead ECG (Lead-I, dry electrodes, 512 Hz, ~30 s, ambulatory/noisy). Focus = **recording-modality invariance**.

**Legend:** novelty flag = `[P]` proven/published in ECG context · `[A]` proven in adjacent field (adaptable) · `[N]` novel/unexplored for this problem.

Categories: **A. Preprocessing/Normalization** · **B. Lead Synthesis/Reconstruction** · **C. Representation Learning / SSL / Foundation Models** · **D. Domain Adaptation & Generalization** · **E. Architectural (lead-agnostic)** · **F. Feature Engineering (physiologic/morphology)** · **G. Cross-modal / Multimodal** · **H. Test-time / Post-training** · **I. Novel / Frontier (unexplored)**.

---

## A. Preprocessing & Signal Normalization

The first line of defense: transform both domains so their *distributions* converge before/during learning.

### A1. Beat-aligned segmentation + resampling `[P]`
Segment into PQRST cycles (R-peak anchored), resample to common rate. Removes sampling-rate and length mismatch. Universal in ECG pipelines; basis of almost every wearable pipeline (e.g., MIT-BIH beat classification). **Watch-relevance: high** — watch clips are ~30 s; beat-averaging yields a robust template that is rate-invariant.
- Evidence: beat-to-beat representation + factor-analysis denoise for wearables — Chan, Miller, Fox 2020 (`arXiv:2012.00110`, Apple Heart & Movement Study lineage).

### A2. Bandpass/baseline-wander standardization matched to watch filter `[P]`
Apple Watch applies a fixed bandpass (~0.3–40 Hz notch @ 60). Apply the *same* filter to clinical leads so both share bandwidth. **Watch-relevance: critical** — bandwidth mismatch is a top-3 source of shift. [P] standard practice; underused as an explicit alignment trick.

### A3. Z-score / per-record normalization (amplitude invariance) `[P]`
Per-record mean/variance normalization kills electrode-impedance & gain differences. Cheap, effective. Watch-relevance: high.

### A4. Per-lead (channel-wise) normalization → channel-agnostic embeddings `[P]`
Normalize each channel independently then learn a shared encoder. Foundation of "modally reduced" representation (B/E below). — Ibtehaz & Mortazavi 2024 (`arXiv:2405.19359`).

### A5. Noise simulation / domain-randomized augmentation `[P]`
Synthesize watch-like noise (motion, baseline wander, contact impedance, quantization) and add to clinical training data so the model sees the target distribution at train time. — structured noise model + simulator (Chan et al. 2020, `arXiv:2012.00110`); DE-PADA augments T-wave by heart-rate (Abu Saleh et al. 2025, `arXiv:2502.04973`).
- **Augmentations to borrow:** time-warp, lead dropout, Gaussian + colored noise, baseline spline drift, amplitude scaling, respiration modulation, muscle-artifact (EMG) injection, random resample, mixup of beats.

### A6. Frequency-domain augmentation (FDA) `[P]`
Perturb signals by *frequency-domain importance* rather than naive transforms — physiologically faithful. Part of CoRe-ECG (`arXiv:2604.11359`).

### A7. Signal-quality gating `[P]`
Reject/weight low-SNR watch segments before inference (SQI). Prevents garbage-in. — wearable textile ECG quality study (`arXiv:2508.21554`); multimodal quality (`arXiv:2105.10046`).

### A8. Scattering transforms / time-frequency representations `[N/P]`
Wavelet scattering yields translation-invariant, stable features that are *robust to deformations* — naturally modality-tolerant. Used for AFib (`arXiv:1510.02541` synchrosqueezing) but underused as an explicit modality bridge. **Frontier candidate.**

---

## B. Lead Synthesis / Reconstruction (single → 12-lead)

Reconstruct a (pseudo) 12-lead from the single lead, then feed a 12-lead-trained model. *Caveat from literature: signal-level reconstruction can leave a latent-space gap that hurts downstream detection* (SelfMIS, `arXiv:2509.19397`). Use with care; prefer latent-space variants (C/D).

### B1. LSTM / Seq2seq lead reconstruction `[P]`
Temporal + spatio-temporal LSTM-UNet to reconstruct 12-lead from reduced leads. — Mallick et al. 2024 (`arXiv:2410.13528`); Gradowski & Buchner 2025 U-Net (`arXiv:2502.00559`).

### B2. GAN-based synthesis (1D Pix2Pix, BiLSTM-gen + CNN-disc) `[P]`
— ECGNet (`arXiv:2310.03753`); Pix2Pix GAN (`arXiv:2410.13528`). State-of-the-art on signal fidelity; preserves P-Q segment & R-peaks.

### B3. Multi-channel masked autoencoder (MCMA) `[P]`
Reconstruct 12-lead from *arbitrary* single lead via masking; benchmarked at signal/feature/diagnostic level (ECGGenEval). — Chen et al. 2024 (`arXiv:2407.11481`, npj Cardiovascular Health).

### B4. Variational autoencoder (VAE) `[P]`
WearECG VAE reconstructs 12-lead from {II, V1, V5}; validated via cardiologist Turing test + ECGFounder fine-tune. — Guan et al. 2025 (`arXiv:2510.11442`).

### B5. Random-forest morphology estimation (classic, with historical priors) `[P]`
First successful synchronous 12-lead synthesis from single-lead; uses subject's *historical* 12-lead to time missing leads; R²>0.90; validated on AliveCor/Kardia. — Afrin et al. 2018 (`arXiv:1811.08035`). **Watch-relevance:** requires prior 12-lead of same patient (not always available) but powerful for enrolled patients.

### B6. Diffusion-based lead/ECG generation `[P]`
Conditional diffusion with structured state-space models; also P2Es (PPG→12-lead, demographic-aware). — (`arXiv:2301.08227`, `arXiv:2509.25480`).

### B7. ODE-constrained generative cardiac dynamics `[P]`
Physics-grounded 12-lead synthesis. — (`arXiv:2409.17833`). Frontier for *physiologically faithful* synthesis.

### B8. Pathology-aware multi-view contrastive reconstruction `[P]`
Regularize latent with pathology manifold to filter anatomical "nuisance" variables; 76% RMSE reduction in patient-independent setting. — Youssef & Singla 2026 (`arXiv:2603.17248`).

### B9. Joint-embedding / latent-space alignment (NOT signal reconstruction) `[P]`
Align single-lead embedding to 12-lead embedding directly in latent space — avoids the signal-level gap. **Key insight from SelfMIS:** transformation-invariance ≠ single-lead enrichment; "self-cutting" pairs multi-lead with its single-lead segment. — Jin et al. 2025 (`arXiv:2509.19397`).

### B10. Vectorcardiography (VCG) / Frank-lead inverse transform `[A/P]`
Recover the 3D cardiac dipole (VCG) then project to any lead set — physics-based modality bridge. Historical basis (Dower matrix). Underused with modern DL. **Frontier:** learn the lead-field transform.

---

## C. Representation Learning / SSL / Foundation Models

Learn modality-robust representations from massive *unlabeled* ECG (clinical), then transfer.

### C1. Contrastive predictive coding (CPC) — top performer `[P]`
First comprehensive ECG SSL study; CPC ~0.5% below supervised, +noise robustness. — Mehari & Strodthoff 2021 (`arXiv:2103.12676`). Confirmed best pretraining objective in scaling study (`arXiv:2605.12241`).

### C2. SimCLR / BYOL / SwAV for ECG `[P]`
SwAV best of three; SSL generalizes ID≈OOD across datasets. — Soltanieh et al. 2023 (`arXiv:2304.06427`). OpenECG: BYOL+MAE > SimCLR (`arXiv:2503.00711`).

### C3. Joint-Embedding Predictive Architecture (JEPA) `[P]`
Predict in latent space (not raw) — avoids reconstructing noise; CroPA masked attention for 12-lead. — ECG-JEPA (`arXiv:2410.08559`).

### C4. Hybrid contrastive + generative (MAE) foundation model `[P]`
ECG-FM: 1.5M ECGs, hybrid SSL, cross-dataset generalizable (AFib AUROC 0.996). — McKeen et al. 2024 (`arXiv:2408.05178`).

### C5. xLSTM / structured state-space (Mamba) backbones `[P]`
State-space models beat transformers for ECG representation (strong inductive bias). — scaling study (`arXiv:2605.12241`); xECG/xLSTM + SimDINOv2 (`arXiv:2509.10151`).

### C6. Contrastive + reconstructive synergy `[P]`
CoRe-ECG: global semantics + local waveform recovery; FDA augmentation + spatio-temporal dual masking to break cross-lead shortcuts. — (`arXiv:2604.11359`). Directly addresses modality shortcuts.

### C7. Clinically-guided contrastive (metadata-as-label) `[P]`
CLEF: use clinical risk scores to weight negative pairs; pretrained on 12-lead, **tested on lead-I** → +2.6% AUROC. — Shu et al. 2025 (`arXiv:2512.02180`). Direct clinical→watch transfer evidence.

### C8. Inter-intra period-aware SSL `[P]`
Physiology-informed pretraining (RR irregularity, P-wave absence) for AFib. — (`arXiv:2410.18094`).

### C9. Knowledge-enhanced multimodal (text reports) with arbitrary-lead masking `[P]`
K-MERL: LLM-extracted structured knowledge + lead-aware encoder + dynamic lead masking → **+16% AUC on partial-lead zero-shot**. — Liu et al. 2025 (`arXiv:2502.17900`). One of the most directly relevant.

### C10. Echo-report supervised contrastive (cross-modal grounding) `[P]`
AnyECG-Echo: contrastive pre-training between single-lead ECG and echo reports; detects 13 structural-heart subtypes from wearables; AUROC 0.87–0.93; *external* cohort validation. — He et al. 2026 (`arXiv:2606.09332`). **Strongest direct "wearable single-lead disease detection" result.**

---

## D. Domain Adaptation & Generalization

### D1. Domain-Adversarial Neural Networks (DANN) `[P/A]`
Adversarial feature alignment so a domain classifier can't tell clinical vs watch. CV-classic; adapted to ECG. — DG benchmark (`arXiv:2303.11338`) adapts CV DG algorithms to 1D biosignals.

### D2. CORAL / moment-matching `[A]`
Align second-order statistics (mean/cov) of source/target feature distributions. Cheap, no labels needed at target.

### D3. Maximum Mean Discrepancy (MMD) alignment `[A]`
Non-parametric distribution alignment in feature space. Common DA baseline.

### D4. Sparse-representation domain adaptation (zero-shot, per-user) `[P]`
Project other users' signals onto new user's signal space via sparse dictionary learning → zero-shot wearable arrhythmia detection (98.2% acc). — Yamaç et al. 2022 (`arXiv:2207.07089`).

### D5. Personalized augmentation + DA across physiological states `[P]`
DE-PADA: heartbeat segmentation (PQRS vs ST, HR-sensitive), subject-specific T-wave augmentation, dual-expert DA. — (`arXiv:2502.04973`).

### D6. Domain generalization benchmarks (leave-one-dataset-out) `[P]`
BenchECG/OpenECG/DomainGen benchmark: prove DG problem exists; multi-layer-representation arch improves OOD. — (`arXiv:2303.11338`, `arXiv:2509.10151`, `arXiv:2503.00711`).

### D7. Multi-view knowledge transfer (MVKT) `[P]`
Efficient single-lead classification via multi-view knowledge transfer from 12-lead teacher. — (`arXiv:2301.12178`).

### D8. Cross-dataset OOD SSL (ID≈OOD) `[P]`
Evidence SSL representations generalize across datasets with little loss. — (`arXiv:2304.06427`).

---

## E. Architectural (lead-agnostic)

### E1. Modally-reduced / channel-agnostic representation `[P]`
Joint optimization of reconstruction + alignment so different channels → unified embedding; moderate 12-lead approximation from single channel. — Ibtehaz & Mortazavi 2024 (`arXiv:2405.19359`). **Directly on-topic.**

### E2. Lead-aware encoder with dynamic lead masking `[P]`
Encoder accepts arbitrary lead subsets; masks encode which leads present. — K-MERL (`arXiv:2502.17900`).

### E3. Permutation/set-invariant networks (DeepSet/Transformer over leads) `[A/N]`
Treat leads as an unordered set → naturally lead-count-agnostic. Borrowed from point-cloud/set literature. **Frontier for ECG.**

### E4. Multi-lead-branch fusion (MLBF-Net) `[P]`
Per-lead branches fused; can train with missing leads. — (`arXiv:2008.07263`).

### E5. LeadBridge learnable adapter `[P]`
Map 3-lead wearable → 12-lead-compatible representation via adapter (no full synthesis). — CogAdapt (`arXiv:2605.22774`). Cleanly separates "modality adapter" from "task head".

### E6. Modality-decoupled architecture + interleaved modality dropout `[P]`
ECG-R1: robust when either signal or image missing; modality-agnostic MLLM. — (`arXiv:2602.04279`).

### E7. Hypercomplex / parameter-efficient nets `[P]`
Quaternion/hypercomplex nets capture inter-lead correlations with fewer params; AFib. — (`arXiv:2211.02678`).

---

## F. Feature Engineering (physiologic / morphology — modality-robust by construction)

Hand-crafted / physiologic features are *invariant to recording modality* because they describe the heart, not the sensor.

### F1. RR-interval / rhythm features `[P]`
Rate & rhythm irregularity — fully modality-invariant. AFib from RR alone (`arXiv:2302.07648`). Cheap, robust, watch-friendly. **But loses morphology (f-waves).**

### F2. Morphology features (P/QRS/T amplitude, duration, intervals, axis) `[P]`
Clinical gold-standard descriptors; extracted after delineation. Modality-invariant if normalized. Motif-based signatures (`arXiv:2606.00107`); singular-value AFib marker (`arXiv:2307.02806`).

### F3. Beat-template / motif signatures (DTW-aligned) `[P]`
DTW-minimized representative motifs; drift metrics vs NSR/personalized baseline. — Bijlani & Villarroel 2026 (`arXiv:2606.00107`). Strongly modality-tolerant.

### F4. Heart-rate variability (HRV) + nonlinear dynamics `[P]`
Time/freq/nonlinear HRV (entropy, DFA, Poincaré). Fully modality-invariant; used with short ECG + long-term HRV fusion (`arXiv:2403.15408`).

### F5. Kalman / spectro-temporal features `[P]`
Kalman spectro-temporal analysis; HKF denoising with online evolution priors. — (`arXiv:1812.05555`, `arXiv:2210.12807`).

### F6. Hand-crafted features as complement to foundation models `[P]`
SignalMC-MED: hand-crafted ECG features give strong baseline + *complementary value* when combined with FM embeddings. — (`arXiv:2603.09940`). **Practical recipe: FM embeddings ⊕ physiologic features.**

### F7. Raw-waveform end-to-end (rhythm + morphology) `[P]`
RawECGNet: raw single-lead, generalizes across geography/ethnicity/lead-position (F1 0.91–0.94). — Ben-Moshe et al. 2023 (`arXiv:2401.05411`). **Direct evidence raw single-lead can be domain-robust.**

---

## G. Cross-modal / Multimodal

### G1. ECG↔PPG Siamese shared-information learning `[P]`
SiamAF: joint ECG+PPG → predict AF from either; fewer labels; robust to low-quality. — (`arXiv:2310.09203`).

### G2. PPG→12-lead diffusion (P2Es) `[P]`
— (`arXiv:2509.25480`).

### G3. Cross-domain joint dictionary learning PPG→ECG `[P]`
K-SVD joint dictionaries; label-consistent variant. — Tian et al. 2021 (`arXiv:2101.02362`).

### G4. ECG↔echo contrastive (cross-modal grounding) `[P]`
AnyECG-Echo (C10) — strongest wearable-disease result.

### G5. SCG (vibrational) → ECG reconstruction `[P]`
Vib2ECG: IMU chest vibrational → 12-lead; flags generative "hallucination" risk. — (`arXiv:2603.15539`).

### G6. ECG signal + image (modality-decoupled MLLM) `[P]`
ECG-R1 (E6).

### G7. ECG + sampled long-term HRV fusion `[P]`
— (`arXiv:2403.15408`).

---

## H. Test-time & Post-training Adaptation

### H1. Post-training strategy (stochastic depth + linear-probe preview) `[P]`
Bridges foundation-model vs task-specific gap; +0.7–8.9% AUROC; 30% data beats full-data baseline. — Zhou et al. 2025 (`arXiv:2509.12991`).

### H2. Progressive fine-tuning (limit representational drift) `[P]`
ProFine: unfreeze encoder layers in stages → adapt clinical FM to wearable with minimal drift. — CogAdapt (`arXiv:2605.22774`).

### H3. Test-time adaptation (TTA) `[A/N]`
Adapt on the fly to each watch recording (entropy-min / batch-norm adaptation). Big in CV; **largely unexplored for ECG modality shift.** Frontier.

### H4. Test-time training / self-supervised TTA on the watch clip `[A/N]`
Run a quick SSL objective on the patient's own watch data at inference. Frontier.

### H5. LeadBridge adapter at test time `[P]`
Learnable adapter maps watch→clinical-FM input space. — CogAdapt (E5).

---

## I. Novel / Frontier (unexplored or underexplored for this problem)

### I1. Physics-informed lead-field transform (learnable VCG) `[N]`
Learn the biophysical mapping from any electrode pair to a canonical 3D cardiac dipole (VCG), then project to clinical leads. *Grounds* the modality bridge in electrophysiology rather than pure data. (B10 extended.)

### I3. Speech-processing channel-invariance techniques `[N]`
Microphone-channel robustness (feature-domain channel normalization, speaker/condition embedding subtraction, x-vector conditioning). ECG ≈ periodic quasi-stationary signal → speech methods (MFCC-like cepstral features, RASTA, fMLLR) map naturally. **Underexplored.** Adjacent: speaker-channel NPLDA (`arXiv:2002.03562`), Guided-GAN for mismatched acoustic env (`arXiv:2210.00721`), speech-separation pretrained frontend to minimize domain mismatch (`arXiv:2411.03085`).

### I4. Domain-invariant scattering / wavelet features + classifier `[N]`
Scattering coefficients are provably stable to deformations & group actions → modality-tolerant by construction (A8 deepened). Deep Scattering Spectrum (`arXiv:1304.6763`); Joint Time-Frequency Scattering (`arXiv:1807.08869`, `arXiv:1512.02125`).

### I5. Invariant Risk Minimization (IRM) / environment-based OOD `[N]`
Exploit multiple "environments" (clinical sites + simulated watch) to learn predictors that use only invariant features. **Unexplored for ECG modality.** Canonical: Arjovsky et al. (`arXiv:1907.02893`); ERM++ strong DG baseline (`arXiv:2304.01973`).

### I6. Causal/disentangled representations (modality vs content) `[N]`
Disentangle "modality" (device/noise) from "pathology" factors; intervene/ablate modality at inference. Adjacent: style-content disentanglement (`arXiv:2007.04964`, `arXiv:2108.04441`, `arXiv:2302.09795`).

### I2. Instance/subject normalization + AdaIN-style modality transfer `[N]`
Borrow adaptive instance normalization (style transfer) to swap "clinical style" ↔ "watch style" in feature space at no label cost. AdaIN (`arXiv:1703.06868`); Style Normalization & Restitution for DG (`arXiv:2101.00588`); Reciprocal Normalization for DA (`arXiv:2112.10474`).

### I7. Simulated watch-from-clinical physics pipeline `[N]`
Forward model: 12-lead → project to Lead-I → apply watch transfer function (electrode impedance, filter, quantization, motion) → train on augmented. Closes domain gap *by construction*.

### I8. Foundation-model probing of "device signature" + scrubbing `[N]`
Probe FM for a linear "device" direction; subtract/project it out (analogous to debiasing).

### I9. Optimal-transport (OT) feature alignment `[N]`
OT/Wasserstein matching of clinical↔watch feature distributions; unbalanced-OT for noisy watch. Stronger than MMD/CORAL for heavy-tailed biosignals.

### I10. Self-supervised TTA per patient clip `[N]` (H4 deepened)

### I11. Federated/personalized on-watch adaptation `[N]`
On-device continual adaptation with privacy; patient-specific adapter.

### I12. Synthetic watch data via generative world model `[N]`
Generate diverse synthetic watch ECG from clinical labels (diffusion + physics) to enrich target-domain data.

### I13. Multi-task auxiliary "modality prediction" as regularizer `[N]`
Auxiliary head predicts recording modality; gradient-reversal → forces main features to be modality-blind (DANN but explicit).

### I14. Frequency-warped / phase-invariant features (group-equivariant nets) `[N]`
Group-equivariant CNNs over heart-phase; built-in invariance to rate & electrode timing.

### I15. Foundation-model embeddings ⊕ hand-crafted physiologic features (ensemble) `[P]`
SignalMC-MED shows complementarity (F6). Practical, low-risk recipe.

---

## Cross-cutting recommendation matrix

| Goal | Best-bet proven methods | Best-bet frontier |
|---|---|---|
| Close bandwidth/sampling shift | A2 matched filter, A5/A6 noise aug | I7 physics sim |
| Close lead-count shift | B9 latent alignment, E1/E5 channel-agnostic+adapter, C9 lead masking | I1 learnable VCG, E3 set-invariant |
| Close electrode/noise shift | A1/A3/A4 norm, A5 noise aug, A7 SQI | I2 AdaIN, I6 disentanglement |
| Cross-dataset generalization | C1–C7 SSL/FM, D6 DG benchmark, F7 raw | I5 IRM, I9 OT |
| Few/no watch labels | C7/C10 metadata/echo grounding, D4 zero-shot | I3 speech methods, I10 SSL-TTA |
| Watch disease detection | C10 AnyECG-Echo, F7 RawECGNet, F6 ensemble | I12 synthetic watch |

---

## Key references (arXiv IDs)
2405.19359 · 2606.09332 · 2509.19397 · 2509.25480 · 2502.00559 · 2410.13528 · 2407.11481 · 1811.08035 · 2310.03753 · 2410.08559 · 2408.05178 · 2103.12676 · 2303.11338 · 2207.07089 · 2304.06427 · 2509.12991 · 2512.02180 · 2401.05411 · 2606.00107 · 2605.12241 · 2503.00711 · 2509.10151 · 2012.00110 · 2410.18094 · 2502.17900 · 2605.22774 · 2310.09203 · 2101.02362 · 2510.11442 · 2502.04973 · 2604.11359 · 2603.17248 · 2602.04279 · 2301.12178 · 1510.02541 · 2307.02806 · 2302.07648 · 2211.02678 · 2008.07263 · 1804.06812 · 1907.01513
