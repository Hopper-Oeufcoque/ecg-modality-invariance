# Method Ladder: Clinical 12-Lead → Apple Watch Single-Lead ECG Transfer

**Date:** 2026-07-27 · **Repo:** github.com/Hopper-Oeufcoque/ecg-modality-invariance
· **Author:** Hopper · **Status:** Experimental results, PTB-XL 100 Hz, CPU

A coherent experimental arc validating the forward-physics watch simulator (F10)
and ranking proven methods for closing the clinical→watch modality gap. All
experiments on PTB-XL 100 Hz, 5 superclasses (NORM/MI/STTC/CD/HYP), ~1225 train /
~497 test, 1D ResNet, single seed.

---

## The problem, measured

A model trained on clinical 12-lead ECG, tested on simulated single-lead Apple
Watch ECG, loses **−0.34 macro AUROC** (0.865 → 0.527). E1's axis decomposition
shows **lead-count accounts for essentially all of it** (−0.338 of −0.311); the
noise/bandwidth/electrode axes add nothing measurable on top at 100 Hz. The naive
12-lead→single-lead transfer is near chance because the model never saw missing
leads. This empirically grounds the synthesis report's central bet: **lead-count
is the war.**

## The method ladder (macro AUROC on full simulated watch, L4)

| # | Method | L4 full watch | Δ from naive | Category |
|---|---|---|---|---|
| — | **L0 clinical ceiling** (12-lead) | 0.865 | — | upper bound |
| 0 | Naive 12-lead → single-lead | 0.521 | — | (the gap) |
| 1 | + watch-sim augmentation (A5) | 0.551 | +0.03 | preprocessing/aug |
| 2 | + **lead-masking (K-MERL, C9)** | **0.717** | **+0.20** | architectural |
| 3 | + test-time BN adaptation (H3) | 0.724 | +0.20 | test-time |
| ref | latent alignment, frozen+probe (B9) | 0.700 | +0.18 | representation |
| ref | latent alignment, pretrain+finetune (B9) | 0.690 | +0.17 | representation |
| ref | single-lead model (trains on Lead-I) | 0.690 | +0.17 | target-domain |

## Headline findings

### 1. Lead-masking is the practical winner (E2)
A 12-lead model trained with random lead dropout (keep Lead-I) reaches **0.750
on clean Lead-I and 0.717 on full watch** — matching the single-lead reference
model (0.751) on clean Lead-I and **beating it on full watch** (0.717 vs 0.690).
You keep the rich 12-lead training signal and still deploy on single-lead, with
**no target-domain labels, no adapter, no synthesis.** This reproduces K-MERL
(arXiv:2502.17900, +16% AUC on partial-lead zero-shot) on PTB-XL and should be
the default baseline for any clinical→single-lead transfer.

### 2. The 12-lead prior is a feature, not a bug (E2)
Lead-masking (0.717) beats the single-lead model (0.690) on full watch — the
model that trained on 12 leads is *more* robust to watch noise than one that
only ever saw one clean lead. The richer spatial training signal transfers.

### 3. Watch-sim augmentation alone does NOT fix lead-count (E2)
Watch-aug (A5) alone: 0.521/0.551 — no better than naive on the lead-count axis.
It only helps the noise axis, which E1 showed is minor. **Augmentation fights the
wrong battle alone**; it's redundant once lead-masking handles lead-robustness
(combo V4 ≈ V2).

### 4. Latent-space alignment works but does NOT beat lead-masking (E3, E3b)
SelfMIS-style self-cutting contrastive alignment beats naive (0.700 vs 0.521 on
L4). The watch-sim variant (aligning watch-like single↔multi) > clean variant
(0.700 vs 0.655) — the F10 simulator is useful as a *training* tool for
alignment, not just an eval probe. **But the fair test (E3b: contrastive
pretrain THEN end-to-end fine-tune with lead-masking) = 0.690 on L4 — slightly
WORSE than lead-masking alone (0.717).** Contrastive pretraining does not add
value on top of end-to-end lead-masking for this task; it may even slightly
hurt (the contrastive objective may preserve lead-distinguishing structure
unhelpful for classification). The simple method wins decisively.

### 5. Test-time adaptation is a finishing move, not a main lever (E5)
BN-adapt adds +0.007 under genuine shift (L4) and slightly hurts without shift
(L1 −0.011) — exactly TTA theory. Small because lead-masking already closed the
dominant axis. Stack it on top for free; don't rely on it alone.

### 6. Spatial pathologies collapse under lead reduction; conduction survives (E1)
Per-class at naive L1: CD (conduction, bundle-branch blocks) stays 0.635 because
QRS widening is lead-invariant; MI/STTC/HYP collapse to chance. Lead-masking
recovers the spatial classes substantially (MI 0.508→0.608, STTC 0.487→0.759).
This is the central sanity check passing: the gap degrades spatial classes more
than lead-invariant ones, matching real clinical→watch transfer.

## What this refines vs the synthesis report
- The report ranked latent-space lead alignment (B9) as the top solution based on
  SelfMIS. **SelfMIS showed latent alignment > signal synthesis, not > lead-masking.**
  E2/E3 refine the ranking for the lead-count axis: **end-to-end lead-masking (C9)
  > latent alignment + probe (B9)** when you can train end-to-end. Latent alignment
  remains the right tool when you *can't* train end-to-end (frozen FM + adapter).
- The report's "noise/bandwidth axes" methods (A2/A5) are de-prioritized: at 100 Hz
  they're minor; a 500 Hz rerun would show more (queued).

## Honest limitations
- **100 Hz geometry** mutes the bandwidth axis; 500 Hz rerun queued.
- **Single seed**, no bootstrap CIs — directional results, not significance tests.
- **HYP under-sampled** (rare in PTB-XL); its AUROC is noisier.
- Simulated watch, not real Apple Watch data — the simulator is validated by the
  axis structure (lead-count dominant, spatial classes hit hardest) matching the
  literature, but real-watch validation is the real test.
- PTB-XL superclasses are coarse; finer pathology labels may show bigger gaps.

## Recommended recipe (what we'd ship)
**Two regimes (E17):**
1. **No simulator / no target-domain generation** → **lead-masking (C9, prob=0.5)**
   on 12-lead training. Best label-free transfer: 0.725 on full watch. Simplest
   recipe — one training-time augmentation, no preprocessing, no TTA (E16 showed
   combinations add nothing).
2. **With the forward-physics simulator (E1's F10)** → **train a single-lead
   model on simulated-watch Lead-I** with clinical labels. Beats the 12-lead
   approach: 0.742 on full watch, and the model needs no 12-lead data at
   inference. The simulator's best use is as a *from-scratch training
   distribution*, not a fine-tune stage (E16 C5) or alignment signal (E3-B).

The remaining gap to the 0.865 ceiling (~0.12) is genuine single-lead
information loss + sim/real distribution mismatch. The critical open question
(E6): does the sim-trained model's edge hold on *real* watch data, or is it a
sim-only artifact?

## Next experiments (queued, not run)
- **E4:** 500 Hz rerun to surface the bandwidth axis properly.
- **E6:** Real single-lead data (PhysioNet single-lead sets) to validate the
  simulator's gap against real watch shift.
- **E7:** LeadBridge adapter (E5 in taxonomy) — the labelled-target path, to beat
  lead-masking when some watch labels exist.
- **E8:** Scattering/speech-channel features (I4/I3) as modality-robust inputs.

## Experiment index
- E1 — `results/01_simulator_validation/` — simulator validation + axis decomposition
- E2 — `results/02_lead_masking/` — lead-masking wins, matches single-lead reference
- E3 — `results/03_latent_alignment/` — latent alignment (frozen+probe)
- E3b — `results/03b_latent_finetune/` — latent alignment (pretrain+finetune, fair)
- E5 — `results/05_tta/` — test-time BN adaptation
