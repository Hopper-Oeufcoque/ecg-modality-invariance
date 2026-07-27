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

**Ceiling:** L0 clinical 12-lead = **0.865**. **Current best stack:**
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

---

## Standing TODOs / open questions
- **E4 (500 Hz rerun):** surface the bandwidth axis properly — currently muted.
- **E6 (real single-lead data):** validate the simulator against *real* watch
  shift (PhysioNet single-lead sets); biggest honesty flag remaining.
- **E7 (LeadBridge adapter, E5 in taxonomy):** the labelled-target path — does
  an adapter beat lead-masking when some watch labels exist?
- **E8 (scattering / speech-channel features, I4/I3):** modality-robust inputs
  as a complement, not replacement (per F6 ensemble finding).
- **Signal-synthesis baseline (B1):** confirm SelfMIS "synthesis < latent
  alignment" on this data.
- **Bootstrap CIs / multi-seed:** current results are directional, not
  significance tests.

## How to add an entry
1. Run experiment → write `results/<id>/REPORT.md` + `metrics.json` + figures.
2. Add a row to the **Quick verdict table** (top).
3. Append a full entry here (hypothesis/setup/result/verdict/lesson/⟲follow-up).
4. Move any spawned ideas into `docs/FUTURE_APPROACHES.md`.
5. Commit with `experiment <id>: <one-line>`.
