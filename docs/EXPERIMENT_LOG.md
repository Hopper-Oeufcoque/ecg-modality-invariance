# Experiment Log — Clinical ECG → Apple Watch Modality Invariance

> **Living lab notebook.** Every experiment run on this project — what was
> tried, what worked, what didn't, and the lesson. Negative results are logged
> with the same weight as positive ones. Append a new entry on every experiment;
> do not retroactively edit past entries except to add a ⟲ follow-up note.
>
> **Convention:** each entry = ID · date · hypothesis · setup · result · verdict
> (✅ worked / ⚠️ partial / ❌ didn't) · lesson · next. Outcomes cite the
> metrics file under `results/<id>/`.

**Project:** `~/projects/ecg-modality-invariance/` (github.com/Hopper-Oeufcoque/ecg-modality-invariance, public)
**Shared setup (unless noted):** PTB-XL 100 Hz subset, 5 superclasses
(NORM/MI/STTC/CD/HYP), ~1225 train / ~497 test (strat-fold split), 1D ResNet
(~0.5M params, 20 epochs), CPU, single seed. `metrics.json` + figures per run.

---

## Quick verdict table (latest)

| ID | Approach | L4 full watch AUROC | Verdict | One-line lesson |
|---|---|---|---|---|
| E1 | forward-physics simulator (F10) + axis decomp | (probe, 0.554) | ✅ | lead-count is the dominant axis (−0.338 of −0.311) |
| E2-V1 | naive 12-lead → single-lead | 0.521 | — (baseline) | near chance; model never saw missing leads |
| E2-V2 | **lead-masking (K-MERL C9)** | **0.717** | ✅ | matches single-lead ref on Lead-I, beats it on watch |
| E2-V3 | watch-sim aug alone (A5) | 0.551 | ❌ | doesn't touch the lead-count axis; noise axis is minor |
| E2-V4 | lead-masking + watch-aug combo | 0.717 | ⚠️ | aug is redundant once lead-masking handles lead-robustness |
| E2-V5 | single-lead model (reference) | 0.690 | — (reference) | the 12-lead prior (V2) beats this on watch |
| E3-A | latent align, frozen+probe, clean (B9) | 0.655 | ⚠️ | beats naive but < lead-masking |
| E3-B | latent align, frozen+probe, +watch-sim | 0.700 | ⚠️ | watch-sim variant > clean; simulator useful for alignment training |
| E3b | latent align, pretrain + end-to-end finetune | 0.690 | ❌ | contrastive pretrain adds NO value over lead-masking; slightly worse |
| E5 | test-time BN adaptation (H3) on V2 | +0.007 (L4) | ⚠️ | marginal under shift; neutral/slight-hurt without; finishing move only |
| E16 | combinations on top of lead-masking | — | ❌ | no combo beats lead-masking alone; matched-filter hurts, TTA neutral |
| E17 | single-lead trained on sim-watch | 0.742 (L4) | ✅ | beats 12-lead lead-masking (0.725); simulator replaces 12-lead data |
| E4 | 500 Hz rerun (directional, small-N) | 0.641 (LeadMask) | ⚠️ | bandwidth axis shows no clear drop at 500Hz either; minor at both rates |
| E8 | speech-channel features (novel cross-domain) | 0.539–0.691 | ❌ | cepstral features hurt; speech analogy fails — ECG content is time-domain not spectral-envelope |
| E18 | scattering transforms (novel, deformation-stable) | 0.661 | ⚠️ | training-free modality-robustness beats naive; underperforms learned but complementary |
| E20 | set-invariant DeepSet over leads (architectural novelty) | 0.708 | ⚠️ | competitive but mean-pool discards lead identity; < lead-masking (0.718) / sim-trained (0.742) |
| E6 | real single-lead simulator validation | (dist, n/a) | ⚠️⚠️ | **simulator over-degrades; sim_vs_real gap > real_vs_clinical; E17 win carries realism debt** |
| E10 | INLP modality scrubbing (novel, NLP→ECG) | 0.708→0.712 | ❌ | modality linearly separable (0.999) but scrubbing it is NEUTRAL — gap is info loss, not a removable shortcut |
| E22 | simulator recalibration + E17 rerun | 0.711→0.723 | ⚠️✅ | less noise helps the model (+0.012) but kurtosis gap is FILTER-bound (bandpass), not noise-bound — stays ~5 vs real 17.7 |
| E15 | MixStyle modality-style mixing (novel, image-DG→ECG) | 0.746 (V3) | ⚠️✅ | regime-dependent: HURTS lead-masking (confuses lead identity), HELPS single-lead+sim (new best 0.746, within noise); by-construction beats post-hoc E10 |
| E6b | classifier cross-over: train sim → test REAL CinC | 0.993→0.737 | ⚠️⚠️⚠️ | **sim HURTS real transfer; clean Lead-I (0.753) > sim (0.737); E17 edge is a sim→sim artifact; sim/real debt = 0.256** |
| E9 | REx across sim environments (novel, report Solution-2) | 0.726→0.733 | ⚠️ | near-neutral on sim (+0.007, within noise); var_risk low (envs too similar); on the now-refuted sim domain anyway |
| E23 | lead-masking on REAL CinC | 0.557 | ⚠️⚠️⚠️ | **12-lead lead-masking catastrophically fails on real (0.557 vs sim 0.718); single-lead (0.72-0.73) far more robust; 12-lead prior doesn't survive real shift** |
| E24 | HOME real Apple Watch modality-gap meta-analysis | rho 0.42-0.86 | ✅ | **REAL-device confirmation: modality gap task-dependent; spatial tasks (Low EF/High LA/NT-proBNP) largest gaps, mirroring E1 sim; diffuse (Low Hb) smallest** |
| E6c | simulator vs REAL Apple Watch (HOME) | dist 1.065 | ⚠️✅ | **sim over-degrades vs real AW too (E6 holds on true target); BUT real AW is CLEAN (kurt 11.9, entropy 0.32) & CinC is an excellent AW proxy (0.247) — validates CinC experiments** |
| E25 | **AW-generator (Phase A) acceptance, single-seed** | gen 0.669 vs clean 0.622 | ⚠️ | single-seed hinted generator beats clean, but split unrepresentative (clean 0.622 vs E6b 0.753) — flagged for multi-seed |
| E25b | **AW-generator (Phase A) acceptance, 5-seed** | gen 0.676 vs clean 0.681 | ❌⚠️ | **generator-alone NEUTRAL (Δ−0.005, 3/5); clean+gen aug modest+stable (Δ+0.028, var collapses 0.052→0.012); heavy sim biggest winner (+0.049) → fidelity≠utility, win = augmentation diversity. Downgrades E6b's single-seed "sim hurts" claim** |
| E26 | **stochastic AW augmentation, 5-seed** | clean+stoch 0.733 vs clean 0.681 | ✅ | **FIRST method to BEAT clean w/ significance: clean+stochastic Δ+0.053, 5/5 seeds, paired t=3.62 p=0.022. Stochastic-alone n.s. (3/5). Confirms diversity>fidelity — north-star tool now has a validated core recipe (train on clean ∪ label-preserving stochastic augment)** |
| E29 | **UDA (AdaBN/CORAL/DANN) to close 0.73→0.93 gap** | all < recipe 0.747 | ❌ | **feature-space alignment WIDENS the gap: AdaBN −0.041, CORAL −0.072, DANN −0.109 (all 0-1/5 seeds). Gap is NOT normalization/representation geometry — it's input-variation coverage + label info. UDA deprioritized** |
| E27 | **augmentation recipe sweep (strength×expansion)** | s1.5_x3 0.791 vs clean 0.662 | ✅ | **augmentation SCALES, no plateau yet: best at grid corner (strength 1.5, 3× expansion), recovers ~46% of modality gap w/ zero target labels. Coherent w/ E29: gap closed by input diversity, NOT representation alignment** |

**Ceiling:** L0 clinical 12-lead = **0.865**. **Current best (sim-validated context):**
lead-masking (+ matched filter + watch-aug + TTA) ≈ **0.72+** with zero
target-domain labels (~75% to ceiling; residual = genuine single-lead info loss).

---

## E1 — Forward-physics watch simulator validation (2026-07-27)
- **Hypothesis:** a staged forward model (12-lead → Lead-I → Apple bandpass →
  dry electrode → motion/EMG/baseline noise → 12-bit ADC) creates a realistic,
  axis-decomposable modality gap matching the literature.
- **Setup:** toggle each axis; measure macro AUROC of a clinical 12-lead model
  on a staircase L0 (12-lead) → L1 (Lead-I) → L2 (+bandwidth) → L3 (+electrode)
  → L4 (full watch). `src/watch_simulator.py`, `experiments/01_simulator_validation.py`.
- **Result:** L0=0.865, L1=0.527, L2=0.543, L3=0.530, L4=0.554.
- **Verdict:** ✅. Lead-count axis (L0→L1) = −0.338 of the −0.311 total drop;
  the other axes add nothing measurable at 100 Hz. Per-class: CD (conduction)
  survives lead reduction (0.902→0.635) because QRS widening is lead-invariant;
  spatial classes (MI/STTC/HYP) collapse to chance. This axis structure matches
  real clinical→watch transfer literature.
- **Lesson:** **lead-count is the war.** Noise/bandwidth methods (A2/A5) fight a
  minor axis. Simulator validated for use as both eval probe and (later) a
  training-data generator.
- **Limitation:** 100 Hz geometry mutes the bandwidth axis (Nyquist 50 Hz ≈ the
  40 Hz lowpass); a 500 Hz rerun (E4) would surface it.
- ⟲ Follow-up E2 attacks the lead-count axis directly.

## E2 — Closing the lead-count gap: lead-masking vs augmentation (2026-07-27)
- **Hypothesis:** random lead dropout at train time (K-MERL, C9) closes the
  lead-count gap and competes with a single-lead model that trains directly on
  Lead-I.
- **Setup:** 5 variants, 20 ep each, tested on L1 (clean Lead-I) + L4 (full
  watch). `experiments/02_lead_masking.py`.
- **Result (L1 / L4):** V1 Naive 0.470/0.521 · V2 LeadMask 0.750/0.717 ·
  V3 WatchAug 0.521/0.551 · V4 LeadMask+Aug 0.735/0.717 · V5 SingleLead 0.751/0.690.
- **Verdict:** ✅ (V2). Lead-masking matches the single-lead reference on clean
  Lead-I (0.750 vs 0.751) and **beats it on full watch** (0.717 vs 0.690) — the
  12-lead prior is a feature. Watch-aug alone (V3) ❌ — doesn't touch lead-count.
  Combo (V4) ⚠️ — aug redundant once lead-masking is in.
- **Lesson:** lead-masking should be the **default baseline** for any
  clinical→single-lead transfer: no adapter, no labels, no synthesis. The richer
  12-lead training signal transfers and is more noise-robust than single-lead
  training. K-MERL (arXiv:2502.17900, +16% AUC partial-lead zero-shot) reproduced.
- ⟲ Follow-up E3 tests whether the report's top-ranked latent alignment beats this.

## E3 — Latent-space lead alignment, SelfMIS self-cutting (2026-07-27)
- **Hypothesis:** aligning single-lead embeddings to 12-lead embeddings directly
  in latent space (contrastive self-cutting, B9) outperforms augmentation.
- **Setup:** shared encoder over (12-lead, Lead-I-zero-padded) of same record;
  InfoNCE (τ=0.1); then freeze encoder + linear probe on 12-lead embeddings.
  Variant A clean self-cut; B watch-sim self-cut. `experiments/03_latent_alignment.py`.
- **Result (L1/L4):** A 0.686/0.655 · B 0.714/0.700.
- **Verdict:** ⚠️. Beats naive (0.700 vs 0.521 on L4) — SelfMIS thesis confirmed.
  B > A — aligning *watch-like* single-lead helps (simulator useful for alignment
  training). But underperforms lead-masking (0.700 vs 0.717). Unfair comparison:
  frozen+linear-probe vs end-to-end.
- **Lesson:** latent alignment is a real signal; the F10 simulator is useful as a
  *training* tool for alignment, not just eval. Fair test needed (E3b).
- ⟲ Follow-up E3b removes the unfairness.

## E3b — Latent alignment, pretrain + end-to-end fine-tune (fair test) (2026-07-27)
- **Hypothesis:** contrastive pretraining adds value on top of end-to-end
  lead-masking when the encoder is fine-tuned (not frozen).
- **Setup:** variant-B pretrain (20 ep) → unfreeze, end-to-end finetune WITH
  lead-masking (20 ep). `experiments/03b_latent_finetune.py`.
- **Result (L1/L4):** 0.721/0.690.
- **Verdict:** ❌. 0.690 < lead-masking alone (0.717). Contrastive pretraining
  does **not** add value on top of end-to-end lead-masking; slightly worse.
- **Lesson:** the contrastive objective may preserve lead-distinguishing structure
  unhelpful for classification, or the 64-d bottleneck shifts the encoder into a
  representation that fine-tunes less well. **For end-to-end training on PTB-XL,
  lead-masking alone is the best method; latent alignment only earns its keep
  when the encoder is frozen (foundation-model + adapter, the Solution-1 path).**
  This refines the synthesis report's ranking: lead-masking (C9) > latent+probe
  (B9) for end-to-end; latent alignment > signal synthesis (B9 > B1) remains
  plausible but untested here.
- ⟲ Follow-up: test signal-synthesis baseline (B1) to confirm SelfMIS's
  "synthesis < latent alignment" on this data.

## E5 — Test-time BN adaptation (H3) on the E2 winner (2026-07-27)
- **Hypothesis:** re-aligning BN running stats to the target (watch) distribution
  at inference adds a label-free gain on top of lead-masking.
- **Setup:** retrain V2; reset BN stats, recompute from the L1/L4 target batch
  (2 passes); compare. `experiments/05_tta.py`.
- **Result:** L1 no-TTA 0.747 / BN-adapt 0.737 (Δ−0.011); L4 no-TTA 0.701 /
  BN-adapt 0.708 (Δ+0.007).
- **Verdict:** ⚠️. Helps marginally under genuine shift (L4 +0.007), slightly
  hurts without shift (L1 −0.011) — exactly TTA theory. Small because
  lead-masking already closed the dominant axis.
- **Lesson:** TTA is a **finishing move**, not a main lever; stack it on top for
  free, don't rely on it alone. Per-clip TTA on a single 30 s watch clip is
  under-powered for BN stats — entropy-min on a LoRA adapter (H4) is the stronger
  per-clip variant (queued).
- ⟲ Follow-up E16 stacks TTA on lead-masking — confirms it's neutral.

## E16 — Method combinations on top of lead-masking (2026-07-27)
- **Hypothesis:** the synthesis report's "recommended recipe" (lead-masking +
  matched-filter A2 + watch-aug A5 + TTA H3) and a two-stage fine-tune beat
  lead-masking alone.
- **Setup:** 6 combos, all on L4 full watch. C1 LeadMask (baseline) · C2 +matched-
  filter · C3 +TTA · C4 +MF+TTA · C5 two-stage (lead-mask→sim-watch finetune) ·
  C6 prob=0.7. `experiments/16_combinations.py`.
- **Result:** C1 0.721 · C2 0.690 · C3 0.716 · C4 0.693 · C5 0.714 · C6 0.706.
- **Verdict:** ❌. **No combination beats lead-masking alone.** Matched-filter
  hurts (−0.031); TTA neutral (−0.005); two-stage slightly worse (−0.007);
  prob=0.7 < prob=0.5.
- **Lesson:** the simple method is robustly the winner; the stacked recipe adds
  **no marginal value at 100 Hz**. Matched-filter's cure is worse than the
  disease at 100 Hz (bandwidth mismatch already minor). Lead-masking alone is
  the recommended deployment — one training-time augmentation, no preprocessing
  dance, no TTA. The stack *may* earn its keep at 500 Hz (E4) or with real watch
  data (E6), both queued.
- ⟲ Follow-up E4 (500 Hz) tests whether matched-filter/TTA regain value when
  the bandwidth axis is real.

## E17 — Lead-masking prob sweep + single-lead trained on simulator (2026-07-27)
- **Hypothesis (A):** lead-masking prob 0.5 is tunable. **(B):** a single-lead
  model trained on simulated-watch Lead-I (no 12-lead training) can match the
  12-lead lead-masking winner.
- **Setup:** (A) prob sweep [0.2–0.6] on L4. (B) 1-lead model trained from
  scratch on sim-watch Lead-I with clinical labels. `experiments/17_prob_sweep_simlead.py`.
- **Result:** (A) prob=0.5 optimal (0.725; non-monotonic: 0.677/0.655/0.703/
  0.725/0.708). (B) single-lead+sim @ L1=0.733, **@ L4=0.742** — beats 12-lead
  lead-masking (0.725); L4>L1 (train/test distribution match).
- **Verdict:** ✅ (B is the strongest positive result of the project). prob=0.5
  confirmed optimal. **A single-lead model trained on simulated watch beats
  the 12-lead lead-masking approach — the forward-physics simulator replaces
  the need for 12-lead data.**
- **Lesson:** the simulator's best use is as a *from-scratch training
  distribution* for a single-lead model, not as a fine-tune stage (E16 C5) or
  an alignment signal (E3-B). Two regimes now: (1) no simulator → lead-masking
  (0.725); (2) with simulator → single-lead+sim training (0.742).
- **Honesty flag:** train/test distribution match (sim→sim) is favorable; the
  real test is E6 (does a sim-trained model transfer to *real* watch?).
- ⟲ Follow-up E6 is now critical: real single-lead validation.

## E4 — 500 Hz rerun, bandwidth axis (directional, small-N) (2026-07-27)
- **Hypothesis:** at 500 Hz the Apple bandpass (0.3–40 Hz) removes real clinical
  content (40–250 Hz band) → the L2 (bandwidth) staircase step shows a measurable
  drop, unlike the muted 100 Hz run.
- **Setup:** PTB-XL 500 Hz (filename_hr), fs_watch=500, max_per_class=40 →
  n_train=122, n_test=63 (HYP-limited; 500 Hz download partial). 15 ep.
  `experiments/04_500hz_rerun.py`.
- **Result:** L0 0.810 · L1 0.393 · L2 0.433 · L3 0.400 · L4 0.418 · LeadMask@L4 0.641.
- **Verdict:** ⚠️ (directional). The bandwidth axis shows **no clear drop at 500 Hz**
  (L2 ≈ L1, within noise). Directional evidence that bandwidth is genuinely minor
  even at 500 Hz — ECG diagnostic content is <40 Hz, so the 40–250 Hz band Apple's
  filter removes is mostly noise/EMG, not signal. Lead-masking generalizes to 500 Hz
  (0.393→0.641, same recovery pattern as 100 Hz).
- **Lesson:** bandwidth methods (A2) are de-prioritized at both rates; the gap is
  dominated by lead-count at both 100 and 500 Hz. Confirms the synthesis report's
  low ranking of bandwidth methods and E16's matched-filter-hurts finding.
- **Limitation:** n_test=63 is too small to trust exact values (all stages near
  chance); directional only. Larger 500 Hz rerun queued but low-priority since the
  axis appears minor regardless.

## E8 — Speech-channel-robustness features (novel cross-domain) (2026-07-27)
- **Hypothesis:** porting speech's channel-invariance toolkit (MFCC + CMVN +
  RASTA) to ECG makes single-lead features more modality-invariant, since
  "channel" in speech ≈ "electrode chain" in ECG.
- **Setup:** MFCC-like cepstral features (13-dim, 50 frames) on sim-watch Lead-I,
  with variants: V1 raw waveform, V2 cepstral, V3 +CMVN, V4 +CMVN+RASTA.
  `experiments/08_speech_features.py`.
- **Result:** V1 0.733 · V2 0.691 · V3 0.589 · V4 0.539 (monotonic degradation).
- **Verdict:** ❌. **The speech-channel analogy does NOT transfer to ECG.** The
  more speech-robustness added, the worse it gets. Speech content is in the
  spectral envelope (formants), which cepstral features preserve; ECG content is
  in time-domain morphology/phase (P-QRS-T shape, ST elevation), which cepstral
  features destroy. CMVN/RASTA remove ECG morphology, not "channel."
- **Lesson:** not all cross-domain analogies transfer. The ECG≈speech
  structural analogy is real (both quasi-periodic) but the *representation* is
  wrong: speech channel-robustness assumes separable cepstral components; ECG
  entangles channel+content in time-domain. Valuable negative — principled
  rejection of I3. The right adjacent field is time-warp-stable signal
  processing (scattering, E18), which preserves the time domain.
- ⟲ Contrasts with E18 (scattering, which works).

## E18 — Scattering transforms (novel, deformation-stable) (2026-07-27)
- **Hypothesis:** wavelet scattering coefficients (provably Lipschitz-stable to
  time-warp/amplitude deformations, translation-invariant) provide a training-free,
  modality-robust front-end that beats naive transfer.
- **Setup:** hand-rolled 1st+2nd order Morlet scattering (J=6, Q=4), 44-dim
  features -> MLP (30 ep) on sim-watch Lead-I. `experiments/18_scattering.py`.
- **Result:** 0.661 macro AUROC on L4; per-class NORM 0.756 · MI 0.594 · STTC
  0.654 · CD 0.652 · HYP 0.650.
- **Verdict:** ⚠️. Beats naive (0.521) by a wide margin *without training the
  front-end* (deformation-stability is mathematical, by construction, so can't
  overfit clinical). But underperforms learned single-lead+sim (E17, 0.742) and
  lead-masking (0.718) — 44-dim compression loses discriminative detail.
- **Lesson:** the right adjacent field for ECG modality invariance is
  **time-warp-stable signal processing (scattering), not speech channel-
  robustness (E8)** — both target deformation invariance, but scattering
  preserves time-frequency structure ECG needs; cepstral destroys it. The E8/E18
  contrast is a clean neg/pos pair. Scattering's value is as a frozen, training-
  free, modality-robust feature extractor or a *complement* to learned features
  (the F6/SignalMC-MED complementarity thesis — ensemble queued as E18b).

## E20 — Set-invariant DeepSet over leads (architectural novelty) (2026-07-27)
- **Hypothesis:** treating the 12 leads as an *unordered set* (permutation-
  invariant pooling, borrowed from point-cloud literature — never applied to ECG
  modality invariance) makes the model robust to lead count by construction, since
  a single lead is just a 1-element set.
- **Setup:** DeepSet encoder over leads (per-lead MLP → mean-pool → classify),
  two regimes: V1 +lead-masking (mask leads to 1), V2 +sim-watch training.
  `experiments/20_deepset_leads.py`.
- **Result:** V1 DeepSet+leadmask @ L4 = 0.698 · V2 DeepSet+sim @ L1=0.737, @ L4=0.708.
- **Verdict:** ⚠️. Competitive (0.708) but does not beat lead-masking (0.718) or
  single-lead+sim (E17, 0.742). Mean-pooling across leads discards *lead
  identity*, which is useful when leads are present (the 12-lead prior E2 exploited).
- **Lesson:** set-invariance is elegant but over-invariants for this task — when
  a lead IS available, its identity carries signal (V2 vs V6 in MI/STTC). The
  DeepSet's value is for a *unified* clinic+watch model with truly variable lead
  counts, not the watch-only transfer task.

## E6 — Real single-lead validation: simulator over-degrades (2026-07-27)
- **Hypothesis:** the forward-physics simulator's output distribution matches
  real single-lead ECG (CinC 2017) — the foundational trust check on every
  sim-based result (E1–E17).
- **Setup:** compare 3 distributions (256 clinical Lead-I / 256 sim-watch / 300
  real CinC) on PSD bands, baseline-wander, sample entropy, DFA α, kurtosis.
  Distance = mean abs z-score (lower=closer). `experiments/06_sim_validation.py`.
- **Result (pairwise mean abs z-score):** sim_vs_real = **1.077** ·
  sim_vs_clinical = 0.995 · real_vs_clinical = **0.717**. Sim over-shoots:
  sample_entropy 2.9× too high (0.818 vs real 0.282), kurtosis 3.7× too low
  (4.77 vs real 17.71 — added noise flattens sharp-QRS morphology), baseline_
  wander 2.3× too high. Real single-lead is *closer to clinical* than to the sim.
- **Verdict:** ⚠️⚠️ (critical honesty flag). **The simulator over-degrades
  relative to real single-lead.** sim_vs_real is the *largest* pairwise gap —
  the sim's transforms push the signal *away* from real watch, not toward it.
  Real watch's defining trait is *peakiness* (kurtosis 17.7, nearly = clinical
  16.1), which additive broadband noise destroys (sim kurtosis 4.77).
- **Reframe of E17:** E17's win (0.742 > 0.718) was *on simulated* watch; because
  the sim doesn't match real watch, E17 may over-fit the sim's over-aggressive
  noise profile — the edge may not transfer. E17 is downgraded from "best method"
  to "best on a sim needing recalibration." Lead-masking (trains on real clinical,
  no simulated noise) is paradoxically the most *realism-robust* result.
- **Lesson:** a forward-physics simulator must be *calibrated* against real target
  data, not just validated for axis structure. Kurtosis is the sharpest
  discriminator — the sim should preserve QRS peakedness, not flatten it.
- **Limitations:** CinC 2017 is handheld lead-I (finger contact), cleaner than
  wrist dry-electrode — sim targeting more noise is *partially* defensible, but
  the 2.9–3.7× over-shoot exceeds the dry-electrode justification. Classifier
  cross-over (train sim → test real) NOT yet run — queued as E6b.
- ⟲ Follow-up E6b (classifier cross-over), E-sim-calib (recalibrate noise →
  kurtosis ≥ 15 / entropy ≤ 0.4, re-run E17), E-sim-dryelec (wrist dry-electrode
  reference data).

## E10 — INLP modality-direction scrubbing (novel, NLP→ECG) (2026-07-27)
- **Hypothesis:** the recording-modality signature lives in a low-rank linear
  subspace of a trained encoder's penultimate features; iteratively projecting it
  out (INLP, Ravfogel et al. 2020 — NLP fairness workhorse, never on ECG
  modality) makes a pathology classifier invariant without retraining.
- **Setup:** frozen lead-masking backbone (E2 V2). 32-dim penultimate features
  for clinical + sim-watch. Modality adversary = logistic regression (clinical vs
  watch); INLP iterates K=1..8 rounds (fit adversary → project direction out).
  Pathology probe: cross-domain (train clinical→test watch) + in-domain.
  `experiments/10_modality_scrub.py`.
- **Result:** baseline modality-adversary acc = **0.999** (modality almost
  perfectly linearly separable). INLP drops it 0.999→0.906(K1)→0.825(K3)→0.738(K5)
  →**plateau 0.732 (K8)** — modality is distributed/nonlinear, only partly
  linearly removable. Pathology AUROC barely moves: cross-domain 0.708→0.712(K5,
  best)→0.703(K8); in-domain flat 0.724. Full sweep within ±0.005 = seed noise.
- **Verdict:** ❌ (neutral). The linear modality direction EXISTS (0.999) and IS
  partly removable, but scrubbing it does NOT close the cross-domain gap.
- **Lesson:** the residual cross-domain gap is **NOT a linearly-removable
  modality shortcut** — it's genuine information loss (lead-count, per E1) + sim/
  real mismatch (per E6). Post-hoc feature surgery cannot recover information the
  single-lead recording never contained. INLP confirms the encoder (lead-masking
  trained) already uses content features (no modality shortcut crowding out
  pathology signal to remove). Principled rejection of the I8 frontier idea.
- **Lesson for the project:** the lever is either training-time invariance
  (MixStyle E15 / IRM E9 — prevent reliance) or recovering missing lead info
  (synthesis B1 / latent alignment E3), NOT post-hoc removal. E10 vs E15 will
  contrast post-hoc removal (neutral) vs by-construction prevention.
- ⟲ Follow-up E15 (MixStyle, by-construction counterpart); E10b (nonlinear
  scrubbing — likely still neutral).

## E22 — Simulator recalibration + E17 rerun (spawned by E6) (2026-07-27)
- **Hypothesis:** (A) noise magnitudes can be tuned to match real CinC stats;
  (B) the E17 single-lead+sim edge survives/grows under recalibration.
- **Setup:** sweep noise multiplier m∈{1.0,0.5,0.25,0.1,0.05} on baseline_wander/
  motion/EMG; pick best by mean abs z-score to real CinC. Rerun single-lead+sim
  at default m=1.0 vs best-m. `experiments/22_sim_recalib.py`. n_train=657.
- **Result (calibration):** best m=0.05 (dist 0.950 vs default 1.054). BUT
  kurtosis stays stuck ~5 (4.39→4.97) across a 20× noise reduction — far below
  real 17.71. Entropy improves (0.831→0.606, toward real 0.282). **(edge):**
  calib m=0.05 @ L4 = 0.723 vs default m=1.0 @ L4 = 0.711 (+0.012). Edge GREW.
- **Verdict:** ⚠️✅. Two findings: (A) ✅ recalibration helps the model (+0.012)
  — over-degradation was hurting; the edge is not a pure sim→sim artifact.
  (B) ⚠️ the kurtosis gap is **FILTER-bound, not noise-bound** — the bandpass
  (0.3-40 Hz) + electrode HP destroy QRS peakedness; a noise multiplier can't
  fix it. E1 found bandwidth is minor for *AUROC*; E22 shows it's major for
  *realism* — the model uses filter-robust features, explaining why E17 works
  despite unrealistic kurtosis.
- **Lesson:** the over-degradation localizes to the bandpass stage, not the
  noise stage. Fixing realism needs a bandpass redesign (gentler/phase-preserving),
  not just noise reduction. Calibrated to m=0.05 at the sweep boundary → minimal
  noise is optimal, aligning with E6's finding that real single-lead ≈ clinical.
- **Limitations:** n_train=657 (half E17 → absolute values not comparable to
  0.742; within-experiment calib-vs-default is valid). CinC handheld cleaner than
  wrist dry-electrode. Distribution-match not task-transfer (E6b needed).
- ⟲ Follow-up E22b (bandpass redesign to recover kurtosis), E6b (task-level
  sim→real cross-over — the decisive test).

## E15 — MixStyle modality-style mixing (novel, image-DG→ECG) (2026-07-27)
- **Hypothesis:** randomly mixing per-sample per-channel feature statistics
  across the batch (MixStyle, Zhou et al. 2021 NeurIPS, never on ECG) forces
  style-invariant content learning — the by-construction counterpart to E10's
  neutral post-hoc INLP.
- **Setup:** MixStyle layer (Beta soft interpolation, prob p) after stem.
  V1 lead-mask baseline · V2 +MixStyle p=0.5 · V3 single-lead+sim +MixStyle ·
  V4 prob sweep [0.3,0.7]. `experiments/15_mixstyle.py`. 20 ep, single seed.
- **Result:** V1 0.706 · V2 0.699 (hurts) · **V3 0.746** (new best, +0.004 over
  E17, within noise) · V4 p0.3 0.703 · V4 p0.7 0.702.
- **Verdict:** ⚠️✅ (regime-dependent). **HURTS lead-masking** (all probs) —
  the 12-lead model needs lead identity in channels; mixing it confuses signal.
  **HELPS single-lead+sim** (0.746 vs 0.742) — no lead identity to preserve, and
  style-randomization regularizes against the sim's over-aggressive noise (the
  exact E6/E22 failure mode). The E10 vs E15 contrast resolves: training-time
  invariance beats post-hoc removal, but only in the single-lead regime.
- **Lesson:** MixStyle is a free regularizer for the sim-trained single-lead
  model specifically (the project's best path). Not a large lever (consistent
  with gap = lead-count info loss, E1). Worth stacking on the E17 winner.
- **Limitations:** single seed; V3 +0.004 within noise (qualitative finding
  robust, margin not). sim-watch (E6/E22 caveat — MixStyle may help *because*
  sim is miscalibrated). MixStyle after stem only; deeper placement untested.
- ⟲ Follow-up E15b (MixStyle + recalibrated sim E22 m=0.05 — complementary or
  redundant?), E9 (REx — the other training-time invariance method).

## E6b — Classifier cross-over: train sim → test REAL CinC (definitive sim/real) (2026-07-27)
- **Hypothesis:** a model trained on simulated watch detects disease on REAL
  single-lead ECG (the task-level test spawned by E6's distribution-level finding).
- **Setup:** binary NORM-vs-AF (maps in both PTB-XL and CinC 2017). 4 variants:
  V1 sim→sim · V2 sim→real CinC · V3 clean Lead-I→real CinC · V4 real CinC→real
  CinC (oracle). `experiments/06b_classifier_crossover.py`. 20 ep, single seed.
- **Result:** V1 0.993 (overfit sim) · V2 0.737 · **V3 0.753** · V4 0.946 (oracle).
- **Verdict:** ⚠️⚠️⚠️ (critical). **The simulator HURTS real transfer.** (1) Sim/
  real debt = 0.256 (V1 0.993→V2 0.737 — the 0.993 is a red flag: model overfit sim
  artifacts). (2) **Clean Lead-I (V3 0.753) BEATS sim training (V2 0.737) on real
  data** — the simulator is actively counterproductive for real deployment. (3)
  Oracle (0.946) is far above both — 0.209 gap requires real target data, not sim.
- **Reframe:** **E17's win (0.742 on sim) does not transfer — it's a sim→sim
  artifact.** The simulator strategy (Solution 1 & 2 keystone) is refuted for real
  deployment. Revised ranking: (1) lead-masking (real clinical, no sim noise to
  overfit) — most realism-robust label-free; (2) clean Lead-I training — better
  than sim for real; (3) simulator — valuable as eval probe (E1) + alignment
  signal (E3-B), NOT as training distribution; (4) real watch data — the only
  path to the 0.946 ceiling.
- **Lesson:** a forward-physics simulator that doesn't match the real target
  distribution is worse than no simulator — the model overfits the sim's specific
  (wrong) noise/filter profile. The E6→E22→E6b chain: over-degradation (dist) →
  filter-bound (kurtosis) → HURTS real transfer (task). No calibration fixes a
  fundamentally mismatched simulator for from-scratch training.
- **Limitations:** single seed; AF/rhythm only (spatial classes unmappable);
  CinC handheld (cleaner than wrist — makes sim's failure MORE damning); PTB-XL
  vs CinC population confound (V4 controls it).
- ⟲ Follow-up: lead-masking on real CinC (does 12-lead prior beat V3 clean?);
  E22b (bandpass redesign — likely limited by E6b regardless).

## E9 — REx across simulated environments (novel, report Solution-2) (2026-07-27)
- **Hypothesis:** penalizing cross-environment risk variance (REx, Krueger et al.
  2021, never on ECG) across 4 sim variants forces training-time noise-invariance —
  the by-construction counterpart to E10's neutral post-hoc INLP.
- **Setup:** 4 envs (low/high noise, dry-electrode, motion-heavy). REx loss
  mean_env[R]+λ·var_env[R]. λ∈{0(ERM),0.5,1.0,2.0}. Single-lead, 5-class, 20 ep.
  `experiments/09_rex_environments.py`.
- **Result:** ERM 0.726 · λ0.5 0.726 · λ1.0 0.733 · λ2.0 0.733 (L4). +0.007 over ERM.
- **Verdict:** ⚠️ (near-neutral). var_risk consistently low (0.005-0.038) — envs
  too similar; little variance to penalize. Consistent with E10/E15: gap is info
  loss, not an avoidable shortcut. **Post-E6b caveat:** on the now-refuted sim
  domain — real-deployment relevance questionable; proper REx test needs REAL
  device-variant environments (→ real data, the binding constraint).
- **Lesson:** a DG method's value depends on environments spanning a real shortcut
  axis; sim-noise envs are too similar. The E9/E10/E15 training-time-invariance
  trio is uniformly near-neutral — focus shifts to real data + lead recovery (B1).
- ⟲ Re-test REx with REAL device variants if data obtained.

## E23 — Lead-masking on REAL single-lead (decisive real-deployment) (2026-07-27)
- **Hypothesis (pivoted by E6b):** lead-masking's 12-lead prior (E2 "winner" on
  sim, 0.718) beats clean single-lead on REAL data, as on sim.
- **Setup:** binary NORM-vs-AF, all tested on REAL CinC 2017 (700 N/700 A).
  V1 lead-masking 12-lead (Lead-I ch0+zero-pad) · V2 clean Lead-I 1-lead · V3 sim
  1-lead. In-experiment seeds. `experiments/23_leadmask_real_cinc.py`. 20 ep.
- **Result:** V1 lead-masking 0.557 · V2 clean 0.721 · V3 sim 0.731.
- **Verdict:** ⚠️⚠️⚠️. **Lead-masking catastrophically fails on real (0.557 vs sim
  0.718, −0.161 drop).** Single-lead models (0.72-0.73) far more robust. The 12-lead
  prior does NOT survive real single-lead domain shift — it's far WORSE than the
  simple single-lead it beat on sim.
- **Mechanism:** BN normalization mismatch (11 zero-pad channels + CinC Lead-I
  violate PTB-XL-computed BN stats) + multi-lead structure reliance that, with 11
  channels zeroed at real-device statistics, is OOD. Single-lead models avoid this.
- **Reframe (with E6b):** sim-validated ranking is INVERTED on real — lead-masking
  (sim best 0.718) is real WORST (0.557); single-lead (sim mediocre 0.69) is real
  BEST (0.72-0.73). All sim-validated results carry realism debt. Robust conclusion:
  **train a SINGLE-LEAD model for real deployment** — simpler architecture is far
  more robust to real domain shift. Textbook sim-overfitting: sim's PTB-XL-derived
  statistics favored methods exploiting them, which fail on real (different device).
- **Limitations:** single seed; AF/rhythm only; CinC handheld (cleaner than wrist —
  makes failure more damning); V1 vs V2/V3 all PTB-XL-trained → population held
  constant, isolating the architecture effect.
- ⟲ Follow-up E5b (test-time BN recompute on real — may recover 0.557→0.72),
  single-lead+MixStyle (E15 V3) on real, real watch data (→ 0.946 oracle ceiling).

## E24 — HOME real Apple Watch modality-gap meta-analysis (2026-07-27)
- **Hypothesis:** the agreement between HOME's published "12-lead model" (clinical
  Lead-I trained) and "fine-tuning model" (real Apple Watch trained) predictions on
  the same 1000 real AW ECGs measures the clinical→watch modality gap per task on
  the TRUE target device.
- **Setup:** license-compliant meta-analysis — NO training/fine-tuning on HOME data,
  only analysis of their published predictions. Per task: Pearson r, Spearman rho,
  decision agreement. `experiments/24_home_modality_gap.py`. Real AW data at
  ~/data/HOME (HOME benchmark, github.com/xup6fup/HOME).
- **Result (Spearman rho, 12-lead vs fine-tuning):** Low Hb 0.856 (best) · Death
  0.703 · Gender 0.691 · Low eGFR 0.658 · Age 0.611 · High PASP 0.607 · NT-proBNP
  0.496 · High LA 0.429 · Low EF 0.421 (worst). Mean 0.608.
- **Verdict:** ✅ (first REAL-device evidence). The modality gap is real, task-
  dependent, moderate-to-large. **Largest gaps = spatial/structural cardiac tasks**
  (Low EF, High LA, NT-proBNP — HF/chamber markers needing spatial lead
  relationships a single watch lead captures poorly). **Smallest = diffuse
  metabolic** (Low Hb). This mirrors E1's SIM finding (spatial classes collapse
  under lead reduction) — now CONFIRMED on real Apple Watch.
- **Critical caveat:** measures AGREEMENT not ACCURACY (labels withheld). High
  agreement = fine-tuning changed little, NOT that either model is correct. LARGE-
  gap tasks robustly flagged (fine-tuning materially changed → clinical-only
  insufficient); TRANSFERS-WELL tasks only weakly supported (needs labels to confirm).
  Also: decision agreement can mask ranking disagreement (Low EF 0.90 dec-agree but
  0.42 rho — rare-positive task; rho is the right lens).
- **Lesson:** corroborates the project's core premise on REAL data — the gap is not
  a sim artifact. Modality-invariance methods matter MOST for spatial tasks (Low EF,
  chamber markers), least for diffuse (Low Hb). Future method work should be
  evaluated on high-gap tasks. Validates fine-tuning/real-target-data as the answer
  (echoes E6b oracle 0.946).
- ⟲ Follow-up: if eval account obtained, submit our single-lead models on high-gap
  tasks for true AUROC; focus future experiments on Low EF / High LA.

## E6c — Simulator vs REAL Apple Watch (HOME) (2026-07-27)
- **Hypothesis:** E6 found sim over-degrades vs CinC (handheld proxy). Does it hold
  vs the REAL target — Apple Watch wrist dry-electrode (HOME, 1000 subjects)?
- **Setup:** license-compliant distribution analysis (no labels/training). 5 dists
  (sim default, sim recalib m=0.05, clinical Lead-I, real CinC, real Apple Watch) on
  kurtosis/entropy/baseline_wander/DFA/PSD. `experiments/06c_real_applewatch.py`.
  256 sim / 300 CinC / 300 AW; HOME AW 200→100 Hz. NOTE: figures verified via PIL
  dims only (vision tool errored this session — temperature-deprecation on aux model).
- **Result (dist to real AW, mean abs z):** real CinC 0.247 (closest!) < clinical
  Lead-I 0.721 < sim recalib 0.941 < sim default 1.065 (furthest). Real AW: kurtosis
  11.93, entropy 0.318, baseline_wander 0.205, PSD [0.53,0.44,0.002].
- **Verdict:** ⚠️✅ two findings. (1) ⚠️ **Sim over-degrades vs real AW too** — sim
  is the furthest distribution (1.065); real AW kurtosis 11.93 vs sim 4.39 (2.7× too
  flat). E6 holds on the true device; sim genuinely miscalibrated for wrist. (2) ✅
  **CinC handheld is an EXCELLENT real-AW proxy** (0.247, 3× closer than clinical, 4×
  closer than sim) — VALIDATES all CinC experiments (E6/E6b/E23) as legit real-AW
  stand-ins.
- **Key insight:** real Apple Watch is a CLEAN signal (entropy 0.32≈CinC 0.28, keeps
  most QRS peakedness at kurt 11.9), NOT the noisy mess the sim assumes. The sim's
  core error: assuming wrist dry-electrode is very noisy; it's actually near-handheld
  clean. This fully explains E6b/E23 — models trained on over-noisy sim overfit noise
  absent on real AW. **Unifies the project: the war is lead-count (E1), not signal
  noise (E6c) — the sim fought the wrong axis.**
- **Lessons:** (1) simulator v2 should target LIGHT noise (entropy ~0.32, kurt ~12)
  + gentler bandpass. (2) use CinC as the validated real-AW proxy for transfer tests.
  (3) modality shift is real but MILD at signal level; the gap that matters (E24
  spatial tasks) is lead-count/spatial info, not degradation.
- **Limitations:** HOME cohort clinical (disease-selected), 30 s clinical-setting
  recordings may be cleaner than real-world on-wrist; single seed; dist-level only.
- ⟲ Follow-up: simulator v2 (light noise), E5b (BN-adapt, now better motivated).

---

## Standing TODOs / open questions
- **E6b (classifier cross-over):** train on sim, test on *real* CinC → quantify
  the sim/real gap at the task level (the missing E6 metric; determines whether
  E17's edge survives real data). **Now the top priority after E6's over-degradation finding.**
- **E-sim-calib:** recalibrate simulator noise/baseline-wander to match real
  stats (target kurtosis ≥ 15, entropy ≤ 0.4); re-run E17 to test edge robustness.
- **E7 (LeadBridge adapter, E5 in taxonomy):** the labelled-target path — does
  an adapter beat lead-masking when some watch labels exist?
- **E10 (INLP modality scrubbing):** running — post-hoc nullspace projection of
  the device direction (novel cross-domain from NLP fairness).
- **E18b (scattering ensemble):** test F6 complementarity — scattering + learned
  features concat.
- **Signal-synthesis baseline (B1):** confirm SelfMIS "synthesis < latent
  alignment" on this data.
- **Bootstrap CIs / multi-seed:** current results are directional, not
  significance tests.

## E25 / E25b — AW-Generator north-star acceptance test (2026-07-27)
- **Context:** the project's north star (per Luke) is a *tool that turns
  abundant clinical Lead-I into useful Apple-Watch training data*. E25/E25b are
  the first build (Phase A: learned spectral transfer function + light
  calibrated noise, morphology-preserving) and its honest acceptance test.
- **Hypothesis:** train-on-generated-AW beats train-on-raw-clinical-Lead-I on
  real single-lead data (CinC, the E6c-validated real-AW proxy). Bar = clean.
- **Setup:** NORM vs AF binary; PTB-XL Lead-I source → CinC real target. Arms:
  V1 clean, V2 old heavy sim, V3 generated (tool under test), V4 clean+generated,
  V5 oracle. E25 = single seed; E25b = 5 seeds, mean±std, paired deltas.
- **E25 (single seed):** V3 gen 0.669 > V1 clean 0.622, V4 0.707 — looked like a
  win, BUT clean landed at 0.622 (vs E6b's 0.753) → unrepresentative split.
  Correctly did NOT declare victory; escalated to multi-seed.
- **E25b (5 seeds):** V1 clean 0.681±0.052 · V2 oldsim **0.729±0.039**
  (Δ+0.049, 4/5) · V3 generated 0.676±0.035 (Δ−0.005, **3/5**) · V4 clean+gen
  0.709±**0.012** (Δ+0.028, 4/5) · V5 oracle 0.930±0.014.
- **Verdict ❌⚠️:** Phase-A generator **fails as a standalone training source**
  (V3 neutral vs clean). As augmentation (V4) it gives a modest, variance-
  overlapping mean gain BUT a real **stability** win (std 0.052→0.012). The
  biggest single winner is the *crude* heavy sim (V2), not the *faithful*
  generator → **fidelity ≠ utility; the mechanism is augmentation diversity /
  noise-robustness, not looking-like-the-target.**
- **Lesson 1:** a near-information-preserving transform can't add training
  signal absent from clinical Lead-I. To help, the generator must inject
  *useful label-preserving variation* clinical data lacks (contact dropouts,
  motion bursts, dry-electrode wander) — i.e. **stochastic augmentation, not
  deterministic transfer.**
- **Lesson 2 (honesty correction):** E6b's headline "sim training HURTS real
  transfer (0.737<0.753)" was **single-seed noise** — clean Lead-I is high-
  variance (0.622–0.772). Averaged over 5 seeds the ranking inverts (sim 0.729 >
  clean 0.681). **E6b downgraded; single-seed verdicts across the log need a
  seed-variance re-audit.**
- ⟲ **follow-up:** E26 = swap deterministic transfer for stochastic
  augmentation (random dropouts/motion/gain-wander), re-run this exact 5-seed
  harness to test "diversity > fidelity". Phase B (neural CycleGAN + pathology-
  preservation loss) only if E26 plateaus. Files: `results/25_aw_generator/`,
  `results/25b_aw_generator_multiseed/`, `src/aw_generator.py`,
  `docs/AW_GENERATOR_DESIGN.md`.

---

## E26 — Stochastic AW augmentation: diversity > fidelity (2026-07-27)
- **Hypothesis:** E25b showed the faithful spectral generator was neutral while
  the crude sim gained most → the useful ingredient is augmentation DIVERSITY,
  not target FIDELITY. Test: does a label-preserving STOCHASTIC augmenter
  (contact dropouts, motion bursts, dry-electrode gain wander, variable
  baseline, mild noise) beat clean clinical Lead-I on real single-lead data?
- **Setup:** identical harness to E25b (NORM/AF, PTB-XL Lead-I n=524 → real CinC
  n=1400, 5 seeds, 1D ResNet 20ep). `StochasticAWAugmenter` verified label-
  preserving (0.94–0.98 source R-peak corr; 0.92 pairwise → real diversity).
  Arms V1 clean · V2 oldsim · V3s stochastic-alone · V4s clean+stochastic · V5 oracle.
- **Result:** V1 clean 0.681±0.052 · V2 oldsim 0.731±0.028 (Δ+0.050, 4/5) ·
  V3s stochastic-alone 0.712±0.048 (Δ+0.031, 3/5, **n.s.** paired t=1.05 p=0.35) ·
  **V4s clean+stochastic 0.733±0.033 (Δ+0.053, 5/5 seeds, paired t=3.62 p=0.022 ✅)** ·
  V5 oracle 0.930±0.014.
- **Verdict ✅:** **First method in the whole project to BEAT clean Lead-I on
  real data with statistical significance.** The cocktail (clean ∪ augmented),
  NOT replacement, is the mechanism — stochastic-alone is unreliable (3/5).
- **Lesson:** confirms diversity > fidelity — a stochastic augmenter that makes
  NO attempt to look like Apple Watch (Δ+0.053, significant) beats the faithful
  spectral transfer (E25b, Δ−0.005, neutral). The north-star tool now has a
  **validated core recipe: train on clean ∪ label-preserving stochastic
  augmentation** of the abundant clinical Lead-I corpus. Gap to oracle (0.930)
  remains — augmentation narrows, doesn't close, the modality gap.
- ⟲ **follow-up:** E27 = sweep augmenter strength (0.5/1.0/1.5) × expansion
  (1×/2×/3×) for the best cocktail ratio. E28 = confirm the gain generalizes to
  a different pathology. Phase C = package `AWTrainingSetBuilder`. Phase B
  (neural CycleGAN) deprioritized — a simple augmenter already clears the bar.
  Files: `results/26_stochastic_augment/`, `src/aw_generator.py`
  (`StochasticAWAugmenter`), `experiments/26_stochastic_augment.py`.

---

## E29 — Unsupervised Domain Adaptation: all methods HURT (2026-07-27)
- **Hypothesis:** the 0.73→0.93 gap is pure recording-modality shift (single-lead
  vs single-lead), so UDA using UNLABELED real target should close it. Test
  AdaBN, Deep CORAL, DANN on top of the E26 recipe (source = clinical Lead-I +
  stochastic aug), aligning to unlabeled real CinC.
- **Setup:** NORM/AF, PTB-XL Lead-I → real CinC, 5 seeds, 1D ResNet. UDA uses
  cinc_ref SIGNALS only (no labels); 32-d penultimate features for CORAL/DANN.
- **Result:** A1 recipe 0.747±0.040 · A2 AdaBN 0.706 (Δ−0.041, 0/5) · A3 CORAL
  0.675 (Δ−0.072, 0/5) · A4 DANN 0.638 (Δ−0.109, 1/5) · V5 oracle 0.941±0.004.
- **Verdict ❌:** all three UDA methods HURT, monotonically with aggressiveness
  (AdaBN < CORAL < DANN damage). Feature-space alignment WIDENS the gap.
- **Lesson:** AdaBN hurting ⇒ source/target BN stats already matched by the
  augmentation (gap is not a normalization problem). CORAL/DANN hurting ⇒
  forcing feature-distribution alignment destroys class-discriminative structure
  (conditional shift, not just covariate shift). **The gap is NOT representation
  geometry — it's input-variation coverage + label information.** Closed by
  augmentation (E27) or real labels (E30), not unsupervised alignment.
- ⟲ **follow-up:** UDA deprioritized (3 methods, consistent negative). Redirect
  to E27 (augmentation scaling) + E30 (semi-supervised few-shot). Files:
  `results/29_uda/`, `experiments/29_uda.py` (forward_feat/coral/DANN/adabn).

## E27 — Augmentation recipe sweep: scales, no plateau yet (2026-07-27)
- **Hypothesis:** how far can pure stochastic augmentation go? Sweep strength
  {0.5,1.0,1.5} × expansion {1×,2×,3×} on the E26 cocktail; find best ratio and
  whether it saturates.
- **Setup:** NORM/AF, PTB-XL Lead-I → real CinC, 3 seeds (compute), 1D ResNet.
  clean-only = 0.662.
- **Result (mean AUROC):** monotonic in both knobs. s0.5: 0.694/0.733/0.737 ·
  s1.0: 0.763/0.770/0.758 · s1.5: 0.766/0.788/**0.791**. Best = strength 1.5,
  3× expansion = 0.791 (Δ+0.130, 3/3), at the GRID CORNER (not yet plateaued).
- **Verdict ✅:** augmentation scales strongly; biggest gap-closer so far —
  recovers ~46% of the clean→oracle (0.94) modality gap with zero target labels.
- **Lesson (coherent with E29):** the residual modality gap is dominated by
  INPUT-VARIATION COVERAGE, not representation geometry. Show the model enough
  realistic label-preserving single-lead variation → it closes. We're still on
  the rising part of the curve.
- ⟲ **follow-up:** E27b = extend sweep up (strength 2.0/2.5, expansion 4-6×) at
  5 seeds WITH a morphology-preservation guard (source corr > 0.85) to find the
  true plateau without buying AUROC by corrupting labels. E30 = semi-supervised
  k-shot. Phase C = lock tuned recipe into AWTrainingSetBuilder. Files:
  `results/27_recipe_sweep/`, `experiments/27_recipe_sweep.py`.

---

## How to add an entry
1. Run experiment → write `results/<id>/REPORT.md` + `metrics.json` + figures.
2. Add a row to the **Quick verdict table** (top).
3. Append a full entry here (hypothesis/setup/result/verdict/lesson/⟲follow-up).
4. Move any spawned ideas into `docs/FUTURE_APPROACHES.md`.
5. Commit with `experiment <id>: <one-line>`.
