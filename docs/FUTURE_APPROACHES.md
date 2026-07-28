# Future Approaches — Experimental Backlog (Growing)

> **Living catalog of approaches NOT yet tried** on this project, beyond the
> completed experiments in `docs/EXPERIMENT_LOG.md`. Each entry: what it is,
> why it might help *here* (grounded in our findings so far), how to implement
> concretely, expected difficulty/value, and status. Append new ideas as they
> surface; mark ✅/❌ and link the experiment ID once run.
>
> Companion to `notes/idea_log.md` (13 frontier ideas F1–F13) and
> `docs/method_taxonomy.md` (full method catalog). This doc focuses on the
> *actionable backlog* — what to run next and why.

## Priority key
🔴 high — directly addresses a found gap/limitation · 🟠 medium — untested proven method · 🔵 frontier/novel

---

## ═══ REAL-DATA ERA BACKLOG (E38+, current) ═══

> Everything below the next divider is **simulator-era** (E1–E23) and largely
> superseded — the simulator over-degrades (E6) and sim-training hurts real
> transfer (E6b). Kept for provenance. The live backlog is here.
>
> **Where we are (post-E48):** closed-loop calibration gives +0.041 zero-label on
> real CinC AF (E42), gap-proportional (E44), rhythm-specific (E47). The +0.041
> augmentation ceiling is **information-bound** — a learned invariance loss hits
> the same wall (E48). So the frontier is **information injection**, not better
> invariance objectives.

### 🔴 E49 — 12-lead→1-lead distillation (RUNNING)
- **Status:** 🔄 running. Train a 12-lead teacher on clinical PTB-XL (train-acc
  0.994), distil (Hinton KD) into a single-lead student that only sees Lead-I.
  Arms: clean / closed_aug / distill / distill+aug on real CinC, 20 seeds. Tests
  whether multi-lead clinical structure is information the augmentation ceiling
  can't reach, and whether it **stacks** with calibration.

### 🔴 ~~E50 — SJLIFE paired-hardware feature alignment~~ (RUN ❌❌)
- **Status:** ✅ ran 2026-07-27. Label-free InfoNCE on 243 real clinical↔watch
  pairs CONVERGED (4.01→0.63) but transfer got WORSE (sjlife_ft 0.669, −0.073 vs
  aug p=2e-5). Invariance-by-information-destruction: tiny paired set + trivial
  same-patient objective → encoder discards pathology morphology. Confirms
  E48/E49/E50 pattern: unanchored representation engineering loses to calibration.

### 🔴 ~~E51 — LABEL-ANCHORED alignment~~ (RUN ✅✅ CONFIRMED — HEADLINE WIN)
- **Status:** ✅ ran + controlled 2026-07-27. Joint CE(clinical Lead-I) +
  λ·InfoNCE(SJLIFE real pairs) → **joint 0.807 / joint_aug 0.820** (+0.078 vs
  calibration, 20/20, p=9e-7). E51b control CONFIRMED it is genuine same-patient
  cross-modality invariance (shuffled pairs → null p=0.76; joint−shuffled +0.101
  p=3e-10), NOT generic SSL regularization. First representation method to beat
  calibration. Now the project headline (see README + EXPERIMENT_SYNTHESIS).

### 🔴 ~~E53 — Rhythm-breadth / morphology test of the E51 WIN~~ (RUN ✅)
- **Status:** ✅ ran 2026-07-27. On E47's morphological task (N-vs-O) where
  calibration is NULL (−0.002), alignment gives **+0.034 (20/20, p=9e-9)** — real,
  unanimous, categorically ≠ calibration's zero. Smaller than AF's +0.106 (morphology
  intrinsically hard: oracle only 0.750; CinC-O weak label). **Alignment is more
  general than calibration** — feature-space, not one input axis. See E53 log entry.

### 🟠 E54 — λ / temperature sensitivity of E51
- **Why here:** λ=0.1, temp=0.1 were a-priori, untuned — the 0.820 may not be optimal.
  A small sweep (λ∈{0.03,0.1,0.3,1.0}, temp∈{0.05,0.1,0.2}) maps robustness and
  whether the win is fragile or broad. **Difficulty:** low. **Value:** medium — shows
  the effect isn't a lucky hyperparameter, and may push past 0.820.

### 🟠 E55 — Few-shot stacking on the E51 champion (best deployable recipe)
- **Why here:** E46 showed calibration + ~50 real labels → 0.855. Does label-anchored
  alignment + k real labels reach oracle (0.93) faster? Maps the best achievable recipe
  under a realistic label budget, now from a much higher zero-label base (0.820).
- **How:** E46 few-shot curve (k=0/25/50/100) with joint_aug as the base.
- **Difficulty:** low (reuse E46). **Value:** high — the deployment recipe.

### 🟠 E52 — Beyond-AF rhythm breadth (flutter / PVC / PAC)
- **Why here:** E47 showed the lift is rhythm-specific but only tested AF vs a
  morphological task. Does the rhythm-transfer claim generalize to *other*
  arrhythmias? Icentia has beat-level flutter/PVC labels (mine more patients,
  patient-disjoint to avoid E44's leakage).
- **How:** mine a patient-disjoint flutter/PVC cohort; repeat the E42 calibration
  test. **Difficulty:** medium (mining + leakage control). **Value:** medium-high
  — bounds the generality of the one positive result we have.

---

## ═══ SIMULATOR-ERA BACKLOG (E1–E23, largely superseded) ═══

## 🔴 ~~E4 — 500 Hz rerun to surface the bandwidth axis~~ (RUN — directional, small-N)
- **Category:** infrastructure / honest rerun
- **Status:** ✅ ran 2026-07-27 (directional, n=122/63 due to partial 500 Hz
  download + HYP rarity). Finding: **bandwidth axis shows no clear drop at 500 Hz
  either** (L2≈L1) — ECG diagnostic content is <40 Hz, so the filtered 40–250 Hz
  band is mostly noise. Bandwidth methods (A2) de-prioritized at both rates.
  Lead-masking generalizes to 500 Hz (0.393→0.641). See E4 entry in
  `docs/EXPERIMENT_LOG.md`. Larger 500 Hz rerun is low-priority (axis is minor).

## 🔴 ~~E6 — Real single-lead validation against actual watch shift~~ (RUN — ⚠️⚠️ critical)
- **Category:** external validation (biggest honesty flag)
- **Status:** ✅ ran 2026-07-27. **The simulator OVER-DEGRADES relative to real
  single-lead.** sim_vs_real distance (1.077) > real_vs_clinical (0.717) — the
  sim pushes the signal AWAY from real watch, not toward it. Sim's added noise:
  kurtosis 3.7× too low (4.77 vs real 17.71 — flattens QRS peakedness),
  sample_entropy 2.9× too high, baseline_wander 2.3× too high. Reframes E17: the
  0.742 win was on sim, may not transfer; lead-masking is the most realism-robust
  result. See E6 entry in `docs/EXPERIMENT_LOG.md`. Follow-ups spawned: E6b
  (classifier cross-over), E-sim-calib (→ E22).

## 🟠 E7 — LeadBridge learnable adapter (taxonomy E5, CogAdapt arXiv:2605.22774)
- **Category:** architectural adapter (labelled-target path)
- **Why here:** lead-masking (E2) is the winner with *zero* target labels.
  LeadBridge asks: with *some* watch labels, does a learnable adapter
  (watch→FM input space) + progressive fine-tune beat lead-masking? Tests
  whether the adapter/finetune layers of Solution 1 add value over the
  label-free baseline.
- **How:** freeze ECGResNet1d backbone (or a pretrained FM), train a small
  adapter MLP on (simulated-watch → backbone), progressive-unfreeze. Compare to
  E2-V2 at matched label budgets.
- **Difficulty:** medium. **Value:** medium-high — defines when labels are
  worth collecting.

## 🟠 E8 — Scattering + speech-channel-robust features (I4, I3)
- **Category:** modality-robust feature engineering
- **Why here:** these are *invariant by construction* (scattering is provably
  Lipschitz-stable to deformations; speech channel-norm tackles the
  electrode/recording-chain nuisance directly). Per SignalMC-MED (F6,
  arXiv:2603.09940) hand-crafted features are *complementary* to FM embeddings,
  not redundant. Test as an ensemble add-on to the E2 winner, not a replacement.
- **How:** compute wavelet scattering coefficients (Kymatio) + cepstral
  features on Lead-I; concat to ECGResNet1d penultimate features; linear probe.
- **Difficulty:** medium. **Value:** medium — likely a small ensemble gain,
  confirms/rejects the F6 complementarity claim on our data.

## 🟠 B1 — Signal-synthesis baseline (lead reconstruction, then classify)
- **Category:** lead synthesis (the SelfMIS warning)
- **Why here:** E3 confirmed latent alignment (B9) beats naive but not
  lead-masking. The untested half of SelfMIS's claim is "signal synthesis (B1)
  < latent alignment (B9) for detection." Run a GAN/VAE/U-Net 12-lead-from-1
  reconstructor, feed reconstructed 12-lead to the clinical classifier, measure.
  Expected: lower than latent alignment — confirming synthesis leaves a latent gap.
- **How:** small LSTM/UNet recon (arXiv:2410.13528, 2502.00559) on PTB-XL;
  evaluate at *diagnostic* level (not just MSE, per ECGGenEval arXiv:2407.11481).
- **Difficulty:** medium-high. **Value:** medium — closes the SelfMIS evidence loop.

## 🔵 E9 — IRM/REx across environments (I5, arXiv:1907.02893)
- **Category:** domain generalization (novel for ECG modality)
- **Why here:** once the simulator generates multiple watch *variants*
  (different noise levels, electrodes), treat them as environments and train
  with Invariant Risk Minimization so the predictor uses only features invariant
  across environments — principled removal of device shortcuts. The synthesis
  report's Solution-2 keystone.
- **How:** define environments = {clinical sites, sim-watch variants}; IRM loss
  (IRMv1 or REx, which is more stable). Ablate vs ERM.
- **Difficulty:** high (IRM is finicky). **Value:** high if it works — the
  principled shortcut-removal the report bet on. Mitigate fragility with REx/DRO.

## 🔵 ~~E10 — Modality-direction scrubbing of FM embeddings (I8)~~ (RUNNING)
- **Category:** post-hoc invariance (no retraining) — NOVEL cross-domain (NLP INLP)
- **Status:** 🔄 running 2026-07-27. Iterative Nullspace Projection (Ravfogel et
  al. 2020, NLP fairness) applied to ECG modality — fit a linear modality
  adversary on penultimate features, project its direction(s) out, re-probe for
  pathology. Baseline modality-adversary acc = 0.999 (modality nearly perfectly
  linearly separable in 32-dim — strong signal to scrub). Ablates K=1..8 INLP
  rounds; cross-domain (train clinical→test watch) and in-domain probes.

## 🔵 E11 — Forward-physics watch data as augmentation for a *foundation model*
- **Category:** data generation (F10 + C)
- **Why here:** E1 validated the simulator; E2 showed sim-aug alone is weak for
  *lead-count* but E3-B showed sim is useful for *alignment*. The untested combo:
  use the simulator to generate unlimited labelled pseudo-watch data and
  *pretrain* a foundation model (CPC/SimCLR) on clinical+sim-watch jointly, then
  linear-probe. Tests whether sim-aug helps when combined with SSL pretraining
  (where it might, vs end-to-end supervised where it didn't).
- **How:** CPC/ByOL pretrain on PTB-XL 12-lead + sim-watch; linear probe on
  single-lead test. Compare to from-scratch E2.
- **Difficulty:** high (SSL pretraining is heavy on CPU). **Value:** high —
  directly tests the report's Solution-1/2 FM path.

## 🔵 E12 — Per-clip entropy-min TTA on a LoRA adapter (H4, I10)
- **Category:** test-time training (stronger than BN-adapt)
- **Why here:** E5's BN-adapt was marginal because BN stats need a batch and
  lead-masking already closed the main axis. Per-clip TTA on a single 30 s watch
  clip is the realistic deployment scenario; entropy-min on a small LoRA adapter
  (not full BN) is the stronger, label-free per-clip method.
- **How:** attach LoRA to the E2 winner; at test, run masked-lead SSL objective
  on the patient's clip, update LoRA only, predict.
- **Difficulty:** medium. **Value:** medium — the realistic per-clip invariance layer.

## 🔵 E13 — Optimal-transport feature alignment (I9, arXiv:1705.08848)
- **Category:** domain adaptation
- **Why here:** OT/Wasserstein matches heavy-tailed multi-modal biosignal feature
  distributions better than moment-matching (CORAL/MMD). Align clinical↔sim-watch
  embeddings via Sinkhorn at train; unbalanced-OT tolerates noisy watch.
- **How:** DeepJDOT loss on the E2 backbone; entropic regularization.
- **Difficulty:** medium-high. **Value:** medium — alternative to latent alignment;
  may close residual noise-axis shift latent alignment didn't.

## 🔵 E14 — Group-equivariant / phase-canonical networks (I14, arXiv:1602.07576)
- **Category:** architectural invariance-by-construction
- **Why here:** re-parameterize ECG by cardiac phase (R-peak anchored) and use
  group-equivariant convs over the phase group → built-in invariance to rate &
  electrode timing, removing a modality axis *by construction* rather than data.
- **How:** G-CNN over (translation+reflection); phase-canonical pooling.
- **Difficulty:** high. **Value:** medium — elegant but risk: some pathologies
  *are* timing changes (AV block) — don't over-invariant.

## 🔵 ~~E15 — AdaIN modality-style swap / MixStyle~~ (SCRIPTED — variant: MixStyle)
- **Category:** feature-space style mixing — NOVEL (image DG → ECG)
- **Status:** 📝 scripted 2026-07-27 (queued after E10/E22). Pivoted from plain
  AdaIN to **MixStyle** (Zhou et al. 2021, NeurIPS) — the principled version:
  randomly mix per-sample per-channel feature statistics across the batch during
  training to simulate novel modality styles, forcing style-invariant content.
  The by-construction sibling of E10's post-hoc INLP. Variants: V1 lead-masking
  baseline, V2 +MixStyle, V3 single-lead+sim +MixStyle, V4 prob sweep.
- **Difficulty:** low. **Value:** medium-high — cheap, stacks on any backbone.

## 🔴 E22 — Simulator recalibration + E17 rerun (spawned by E6) (RUNNING)
- **Category:** simulator calibration (direct response to E6 over-degradation)
- **Status:** 🔄 running 2026-07-27. Sweep a global noise multiplier
  m∈{1.0,0.5,0.25,0.1,0.05} on baseline_wander/motion/EMG sigma to find the
  config matching real CinC stats (target kurtosis ≥15, entropy ≤0.4). Then
  re-run the E17 single-lead+sim winner at best-m vs default-m=1.0. Tests
  whether the E17 edge survives recalibration (grows → over-degration was
  hurting; vanishes → edge was a sim→sim artifact).
- **Honesty:** CinC is handheld (cleaner than wrist dry-electrode); matching it
  may under-model true Apple Watch noise. Distribution-match, not task-transfer.

## 🔴 E6b — Classifier cross-over: train sim, test real CinC (spawned by E6)
- **Category:** task-level sim/real validation (the missing E6 metric)
- **Why here:** E6 was distribution-level only. E6b is the task-level test: train
  a classifier on sim-watch, evaluate on REAL CinC → the AUROC drop vs
  sim→sim quantifies the sim/real gap at the thing that matters (detection).
  Determines whether E17's edge survives contact with real data.
- **How:** build a binary NORM-vs-AF label (AFib maps cleanly in both PTB-XL
  scp_codes and CinC REFERENCE), train on PTB-XL sim-watch, test on real CinC.
- **Difficulty:** medium (label mapping). **Value:** critical — the definitive
  sim-real transfer test.

---

## Ideas surfaced during experiments (not yet full entries — promote when ready)
- The contrastive pretraining *hurt* (E3b) — investigate WHY: is it the 64-d
  bottleneck, the τ, or the objective preserving unhelpful lead-structure? An
  ablation (feature dim sweep, different positive-pair construction) would
  clarify whether latent alignment is genuinely unhelpful or just mis-tuned here.
  *(Still open — promote to E18 if pursued.)*
- ~~Lead-masking prob (0.5) was not tuned — a sweep may push V2 higher.~~
  → **E17 ran the sweep [0.2–0.6].** Results in `results/17_prob_sweep_simlead/`.
- ~~The single-lead model (V5) underperforms on watch (0.690 vs V2 0.717) partly
  because it trained on *clean* Lead-I. Train V5 on *sim-watch* Lead-I → does
  it close the gap to V2?~~ → **E17 ran this too.** Tests whether the simulator
  alone (no 12-lead training) suffices.
- **E16 finding to propagate:** no combination (matched-filter A2, TTA H3,
  two-stage fine-tune) beats lead-masking alone at 100 Hz; matched-filter
  actively hurts. The stacked recipe only *might* regain value at 500 Hz (E4) —
  re-test combinations there once E4 data is in. *(→ add as E4b combo rerun.)*
