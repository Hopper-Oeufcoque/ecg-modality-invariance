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

## 🔴 E4 — 500 Hz rerun to surface the bandwidth axis
- **Category:** infrastructure / honest rerun
- **Why here:** E1 found the bandwidth axis is muted at 100 Hz (Nyquist 50 Hz ≈
  the 40 Hz Apple lowpass). The synthesis report ranks bandwidth methods (A2)
  highly, but we couldn't exercise them. PTB-XL has a 500 Hz subset
  (`filename_hr`) — download it, rerun E1's staircase. Expect L2 (bandwidth) to
  show a real drop, making matched-filter (A2) and watch-aug (A5) more impactful.
- **How:** swap `fs_col="filename_hr"`, set `FS=500`, `SIGLEN=5000`, simulator
  `fs_watch=512` (real resample). Reuse `experiments/01_simulator_validation.py`.
- **Difficulty:** low (mechanical). **Value:** high — fixes the biggest
  methodology limitation of E1–E3.

## 🔴 E6 — Real single-lead validation against actual watch shift
- **Category:** external validation (biggest honesty flag)
- **Why here:** all results are on *simulated* watch. The simulator is
  validated by axis structure matching the literature, but real-watch validation
  is the real test. PhysioNet has single-lead sets (e.g., Icentia11/ICBEB,
  MIT-BIH single-lead, AF single-lead). Measure: does a clinical model's drop
  on *real* single-lead match its drop on *simulated* single-lead? If yes →
  simulator is trustworthy as a stand-in; if no → calibrate simulator params.
- **How:** download a PhysioNet single-lead set, map labels to PTB-XL
  superclasses where possible (rhythm classes like AFib map cleanly; spatial
  less so), evaluate the E2 winner on real single-lead.
- **Difficulty:** medium (label alignment). **Value:** critical — converts
  "simulated" results into trustworthy claims.

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

## 🔵 E10 — Modality-direction scrubbing of FM embeddings (I8)
- **Category:** post-hoc invariance (no retraining)
- **Why here:** cheapest possible invariance on top of a foundation model —
  probe for a linear "device" direction, project it out before the classifier
  (analogous to NLP debiasing). Tests whether a simple geometric intervention
  closes residual shift.
- **How:** take a frozen FM (or our ECGResNet1d), collect embeddings from
  clinical + sim-watch, train a linear probe to predict modality, orthogonalize
  embeddings away from that direction, re-evaluate.
- **Difficulty:** low. **Value:** medium — cheap post-hoc layer; if it works,
  stacks on any model.

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

## 🔵 E15 — AdaIN modality-style swap (I2, arXiv:1703.06868)
- **Category:** feature-space style transfer
- **Why here:** treat clinical vs watch as image "styles"; AdaIN re-normalizes
  watch features to clinical feature statistics (mean/var per channel) at train
  or test — no labels. Style ≡ recording-modality signature (filter, noise color,
  electrode response); content ≡ pathology.
- **How:** adaptive instance norm on the E2 backbone's mid features; train with
  style-aug, optionally apply at test.
- **Difficulty:** medium. **Value:** medium — cheap style normalization; validate
  pathology isn't correlated with "style" via ablation.

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
