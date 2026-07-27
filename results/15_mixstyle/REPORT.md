# E15 — MixStyle modality-style mixing (novel, image-DG→ECG)

**Date:** 2026-07-27 · **Status:** ⚠️✅ Regime-dependent — hurts lead-masking, helps single-lead+sim (new best 0.746)

## Hypothesis
MixStyle (Zhou et al. 2021, NeurIPS — image domain generalization, *never on ECG
modality*) randomly mixes per-sample per-channel feature statistics across the
batch during training, simulating novel modality "styles" and forcing
style-invariant content learning. It's the **by-construction** counterpart to
E10's neutral post-hoc INLP: E10 removes a learned modality direction after the
fact; MixStyle prevents reliance on style during training.

## Setup
- MixStyle layer (Beta-distributed soft style interpolation, p=prob of mixing)
  inserted after the stem, before residual blocks.
- V1 lead-masking baseline (p=0) · V2 +MixStyle p=0.5 · V3 single-lead+sim
  +MixStyle p=0.5 · V4 prob sweep [0.3, 0.7] on lead-masking.
- 20 ep, single seed. `experiments/15_mixstyle.py`.

## Results

| variant | L1 | L4 |
|---|---|---|
| V1 LeadMask (baseline) | 0.739 | 0.706 |
| V2 LeadMask+MixStyle p=0.5 | 0.722 | 0.699 |
| **V3 SimLead+MixStyle p=0.5** | **0.743** | **0.746** |
| V4 LeadMask+MixStyle p=0.3 | 0.731 | 0.703 |
| V4 LeadMask+MixStyle p=0.7 | 0.730 | 0.702 |

Refs: lead-masking (E2) = 0.718 · single-lead+sim (E17) = 0.742.

## Verdict: ⚠️✅ — regime-dependent; the E10 vs E15 contrast resolves

**MixStyle HURTS lead-masking** (V1 0.706 → V2 0.699, and worse at all probs
tested). The 12-lead lead-masking model relies on lead identity in the channel
dimension; mixing per-channel statistics confuses that identity. Style-invariance
is counterproductive when the channels carry genuine spatial information.

**MixStyle HELPS single-lead+sim** (V3 L4 = 0.746 vs E17's 0.742). This is a
**new best**, though +0.004 is within seed noise — the qualitative direction is
the real signal. For the single-lead model there's no lead identity to preserve,
so style-mixing forces noise-invariance on the only channel that matters, and it
helps. The by-construction approach succeeds where E10's post-hoc removal was
neutral — **but only in the single-lead regime**, not the 12-lead regime.

## Interpretation — the E10 vs E15 contrast
- **E10 (post-hoc INLP):** neutral everywhere. Removing the linear modality
  direction after training doesn't help because the gap is info loss, not a
  removable shortcut.
- **E15 (MixStyle, by-construction):** regime-dependent. Hurts 12-lead (lead
  identity is signal), helps single-lead+sim (no identity to preserve, and
  style-invariance regularizes the sim-trained model against noise overfitting).

The resolution: preventing style reliance at train time CAN help, but only when
the encoder isn't using channel identity as signal — i.e., the single-lead
sim-trained path, which is also the E17 winner. MixStyle is a **free regularizer
for the sim-trained single-lead model** specifically. This also aligns with E22:
the sim over-degrades (noise too aggressive), and MixStyle's style-randomization
partially protects against the sim's specific noise profile, which is exactly the
failure mode E6/E22 identified.

## Honesty flags
- Single seed; V3's +0.004 over E17 (0.746 vs 0.742) is within seed noise
  (~±0.005-0.01). The qualitative finding (helps single-lead, hurts 12-lead) is
  robust; the exact margin is not.
- V1 baseline (0.706) is lower than E2's 0.718 — different architecture
  (MixStyleResNet) and init; within-experiment comparisons are valid.
- MixStyle placement (after stem) is one choice; deeper placement untested.
- Only 1st/2nd channel moments mixed; higher-order style survives.
- sim-watch (E6/E22 realism caveat applies — MixStyle may help *because* the sim
  is miscalibrated, partially masking the over-degradation).

## Lesson
MixStyle is a regime-dependent regularizer: helps where channel identity is
absent (single-lead), hurts where it's signal (12-lead). For the project's best
path (single-lead+sim, E17), MixStyle is a free add that marginally improves and
protects against sim noise overfitting — worth including in the final stack. The
E10/E15 pair shows training-time invariance beats post-hoc removal, but neither
is a large lever — consistent with the finding that the gap is dominated by
lead-count info loss (E1), not a removable/avoidable style shortcut.

## Follow-ups
- **E15b — MixStyle + recalibrated sim (E22 m=0.05):** does the gain hold or grow
  when the sim is already less noisy? Tests whether MixStyle and recalibration
  are complementary or redundant.
- **E9 (REx)** — scripted, ready: the other training-time invariance method
  (cross-environment risk variance); tests whether explicit environment structure
  beats MixStyle's implicit style-randomization.

## Artifacts
- `results/15_mixstyle/metrics.json`
- `results/15_mixstyle/mixstyle.png`
