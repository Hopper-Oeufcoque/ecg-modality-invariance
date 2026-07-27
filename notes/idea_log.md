# Idea Log: Novel / Frontier Methods (Phase 4)

Running log of plausible-but-unexplored methods for clinical→Apple-Watch ECG modality invariance, including ideas borrowed from adjacent fields. Each entry: **idea · mechanism · why it should work · adjacent grounding · risk/gap.**

---

## F1. Physics-informed learnable lead-field transform (learnable VCG)
- **Mechanism:** Learn a network that maps any electrode-pair signal to a canonical 3D cardiac vector (VCG / heart vector), plus a projector from VCG→any lead. Train on 12-lead data (leads are linear projections of the same dipole). At inference, Apple Watch Lead-I → VCG → clinical-lead features.
- **Why:** Grounded in electrophysiology (Einthoven triangle, Dower matrix). The 3D dipole is *modality-invariant*; only the projection coefficients differ. Decouples "what the heart does" from "where you measure."
- **Grounding:** classic VCG/Frank leads; Afrin 2018 uses VCG axis for validation (`arXiv:1811.08035`); ODE cardiac dynamics (`arXiv:2409.17833`).
- **Risk:** single Lead-I is information-poor → VCG underdetermined; pathological dipoles may not be reconstructible. Mitigate with a *distribution* over VCGs (probabilistic) rather than point estimate.

## F2. Adaptive Instance Normalization (AdaIN) modality-style swap
- **Mechanism:** Treat "clinical" vs "watch" as image *styles* in feature space. Use AdaIN to re-normalize watch features to clinical feature statistics (mean/var per channel) at train or test time — no labels needed.
- **Why:** Style is exactly the recording-modality signature (filter, noise color, electrode response); content is the pathology. Style normalization for DG is proven in CV.
- **Grounding:** AdaIN (`arXiv:1703.06868`); Style Normalization & Restitution (`arXiv:2101.00588`); Reciprocal Norm (`arXiv:2112.10474`); biosignal DG (`arXiv:2011.06207`).
- **Risk:** if pathology correlates with "style" (it shouldn't across modalities), over-normalization can remove signal. Validate with ablation.

## F3. Speech channel-invariance stack applied to ECG
- **Mechanism:** Port the microphone-channel-robustness toolkit: cepstral mean/variance normalization (→ per-patient baseline subtraction), RASTA filtering (band-pass on cepstral trajectory → removes slow electrode drift), fMLLR/i-vector speaker adaptation (→ patient/device adaptation), x-vector conditioning (→ condition the classifier on a device embedding).
- **Why:** ECG and speech are both quasi-periodic, quasi-stationary over short windows; "channel" in speech ≈ "electrode/recording chain" in ECG. Decades of channel-robustness transfer directly.
- **Grounding:** NPLDA speaker verification (`arXiv:2002.03562`); Guided-GAN mismatched env (`arXiv:2210.00721`); pretrained frontend for domain mismatch (`arXiv:2411.03085`).
- **Risk:** ECG morphology carries diagnostic info that cepstral features may blur; use as *complementary* features (per F6 ensemble finding `arXiv:2603.09940`), not replacement.

## F4. Scattering-transform features (deformation-stable by construction)
- **Mechanism:** Compute wavelet scattering coefficients (1st+2nd order) → translation-invariant, Lipschitz-stable to time-warp & amplitude deformations. Feed to linear/shallow classifier or concat to FM embedding.
- **Why:** Modality shift is *largely a deformation + additive-noise perturbation* — exactly what scattering is provably stable to. No training needed → no overfit to clinical distribution.
- **Grounding:** Deep Scattering Spectrum (`arXiv:1304.6763`); Joint Time-Freq Scattering (`arXiv:1807.08869`, `arXiv:1512.02125`); synchrosqueezing for single-lead (`arXiv:1510.02541`).
- **Risk:** scattering loses phase/localization needed for some pathologies (ST elevation). Pair with morphology features.

## F5. Invariant Risk Minimization across environments
- **Mechanism:** Define environments = {clinical sites A,B,C, simulated-watch}. Train predictor s.t. it uses only features whose risk is *invariant* across environments. The model is forced to drop device-correlated shortcuts.
- **Why:** Clinical FMs likely exploit device-specific shortcuts (CoRe-ECG explicitly breaks cross-lead linear shortcuts `arXiv:2604.11359`). IRM removes them principled-ly.
- **Grounding:** IRM (`arXiv:1907.02893`); ERM++ (`arXiv:2304.01973`); biosignal DG benchmark (`arXiv:2303.11338`).
- **Risk:** IRM is finicky/fragile in practice; REx / DRO may be more stable. Needs ≥2 distinct environments.

## F6. Modality/content disentanglement with adversarial scrubbing
- **Mechanism:** Two encoders: z_modality (predicts device/lead/noise-bin) and z_content (predicts pathology). Adversarial gradient-reversal so z_content can't encode modality. Use only z_content downstream.
- **Why:** Explicitly factors out the nuisance. At inference, watch z_modality is discarded → content embedding lives in clinical-aligned space.
- **Grounding:** DANN lineage; style-content disentanglement (`arXiv:2007.04964`, `arXiv:2108.04441`, `arXiv:2302.09795`); K-MERL lead masking (`arXiv:2502.17900`).
- **Risk:** disentanglement is hard to enforce; may need VAE/β-VAE or mutual-info penalty.

## F7. Optimal-transport feature alignment (unbalanced OT)
- **Mechanism:** Align clinical↔watch feature distributions via Wasserstein/OT (DeepJDOT at train, or Sinkhorn at test). Unbalanced-OT tolerates the noisy/discarded watch samples.
- **Why:** OT handles heavy-tailed, multi-modal biosignal feature distributions better than moment-matching (CORAL/MMD).
- **Grounding:** JDOT (`arXiv:1705.08848`); DeepJDOT (`arXiv:1803.10081`).
- **Risk:** compute cost; sample-size mismatch (few watch labels). Use entropic/Sinkhorn regularization.

## F8. Test-time adaptation per patient clip (TTA/TTT)
- **Mechanism:** At inference, run a quick SSL objective (e.g., masked-lead prediction) on the patient's own ~30 s watch clip; update BatchNorm/adapter params; then predict.
- **Why:** Each watch recording is its own mini-domain. TTA is the cheapest path to per-recording invariance, needs no labels, no retraining.
- **Grounding:** TTA survey (`arXiv:2303.15361`); single-image TTA InTEnt (`arXiv:2402.09604`); TTA for time-series (`arXiv:2506.23424`); diffusion synthetic-domain TTA (`arXiv:2406.04295`).
- **Risk:** 30 s may be too short to adapt reliably; entropy-min can collapse. Constrain to BN stats / LoRA adapters only.

## F9. Group-equivariant / phase-canonical networks
- **Mechanism:** Re-parameterize ECG by cardiac phase (R-peak anchored) and use group-equivariant convs over the phase group (translation + reflection). Built-in invariance to rate & lead timing.
- **Why:** Removes a major modality axis (heart-rate/timing) by construction rather than data.
- **Grounding:** G-CNNs (`arXiv:1602.07576`); motif/DTW beat alignment (`arXiv:2606.00107`).
- **Risk:** pathology sometimes *is* a timing change (AV block) — don't make fully invariant to timing.

## F10. Simulated-watch-from-clinical physics pipeline (forward-model augmentation)
- **Mechanism:** Build a differentiable forward model: 12-lead → pick Lead-I → apply watch transfer function (Apple's bandpass, electrode-impedance filter, ADC quantization, motion/EMG noise, baseline wander) → realistic watch clips. Train model on clinical+simulated-watch so target domain is in-distribution at train time.
- **Why:** Generates unlimited labeled "watch" data with known ground-truth pathology from clinical labels. Closes gap *by construction*.
- **Grounding:** structured noise simulator (`arXiv:2012.00110`); DE-PADA HR augmentation (`arXiv:2502.04973`); FDA aug (`arXiv:2604.11359`).
- **Risk:** simulator fidelity limits realism; domain gap between sim-watch and real-watch remains. Use domain-randomization over a *range* of plausible transfer functions.

## F11. Foundation-model "device direction" probing + scrubbing
- **Mechanism:** Probe a frozen ECG FM for a linear direction that predicts recording modality (clinical/watch). Project embeddings orthogonally away from that direction ("scrubbing") before the classifier — analogous to bias-direction removal in NLP fairness.
- **Why:** Cheapest possible post-hoc invariance on top of existing FMs; no retraining.
- **Grounding:** concept-probing/debiasing lineage; CogAdapt adapter philosophy (`arXiv:2605.22774`).
- **Risk:** device direction may overlap with pathology direction; orthogonalize carefully + validate.

## F12. Latent-space (not signal) lead alignment — confirmed-best synthesis variant
- **Mechanism:** Don't reconstruct 12-lead *signals* (leaves latent gap → hurts detection, SelfMIS `arXiv:2509.19397`). Instead align single-lead embedding ↔ 12-lead embedding directly (contrastive/self-cutting). Optionally use LeadBridge adapter (`arXiv:2605.22774`).
- **Why:** Avoids generative error compounding; aligns where the classifier actually reads.
- **Status:** partially proven — but combo with FM + scrubbing + TTA is novel.

## F13. Cross-modal grounding with echo/clinical reports (label-free disease signal)
- **Mechanism:** Contrastively align watch ECG ↔ echocardiography reports ↔ clinical text (no ECG labels needed). AnyECG-Echo already shows this works for wearables (`arXiv:2606.09332`); K-MERL grounds with text (`arXiv:2502.17900`); CLEF with metadata (`arXiv:2512.02180`).
- **Why:** Unlocks the clinical-label firehose without manual ECG annotation; naturally modality-agnostic (grounded in physiology/semantics).
- **Frontier:** combine with F2/F6 for a full label-free, modality-invariant watch pipeline.

---

## Synthesizing combos (most promising "real solutions")
1. **Proven-strong baseline:** Clinical FM (CPC/CLEF/ECG-FM) → LeadBridge adapter (E5) + progressive fine-tune (H2) → FM-embedding ⊕ hand-crafted physiologic features (F6) → test-time BN adaptation (H3). Low-risk, mostly published components.
2. **Novel high-upside:** Forward-physics watch simulation (F10) + IRM/REx across {clinical, sim-watch} (F5) + modality-direction scrubbing (F11) + per-clip TTA (F8). Targets invariance by construction + principled removal.
3. **Label-free path:** Echo/text cross-modal grounding (F13) + latent alignment (F12) + scattering & speech-channel features (F3/F4) as modality-robust inputs.
