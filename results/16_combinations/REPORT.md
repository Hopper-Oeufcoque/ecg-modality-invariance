# Experiment 16 — Method Combinations on Top of Lead-Masking

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

E2 showed lead-masking is the decisive winner (0.717 on full watch). The
synthesis report's "recommended recipe" stacks lead-masking + matched-filter
(A2) + watch-aug (A5) + TTA (H3). E16 tests whether combinations add value over
lead-masking alone, plus a new two-stage combination and a prob sweep.

## Results (L4 full watch)

| Combo | AUROC | Δ vs LeadMask | Verdict |
|---|---|---|---|
| C1 LeadMask (baseline, prob=0.5) | **0.721** | — | ✅ best |
| C2 LeadMask + matched-filter (A2) | 0.690 | −0.031 | ❌ hurts |
| C3 LeadMask + TTA (BN-adapt) | 0.716 | −0.005 | ⚠️ neutral |
| C4 LeadMask + MF + TTA | 0.693 | −0.028 | ❌ MF drag dominates |
| C5 two-stage (lead-mask → sim-watch finetune) | 0.714 | −0.007 | ❌ slightly worse |
| C6 LeadMask prob=0.7 | 0.706 | −0.015 | ❌ 0.5 > 0.7 |

## Headline finding
**No combination beats lead-masking alone.** Every add-on either hurts or is
neutral. The simple method is robustly the winner — the synthesis report's
"recommended recipe" stack provides **no marginal value at 100 Hz**, and matched-
filter (A2) actively degrades performance.

## Why combinations fail to add value
- **Matched-filter (A2) hurts (−0.031):** applying the 0.3–40 Hz bandpass to all
  training leads removes content the small model can use, and at 100 Hz the
  bandwidth mismatch the matched-filter is supposed to fix is already minor (E1).
  The cure is worse than the disease here.
- **TTA is neutral (−0.005):** consistent with E5 — lead-masking already closed
  the dominant axis; the residual noise axis is too small for BN-adapt to help.
- **Two-stage sim-watch fine-tune (C5) slightly worse (−0.007):** domain-targeted
  fine-tuning on simulated watch doesn't help when lead-masking already handles
  lead-robustness; the simulated-watch distribution adds noise to training
  without closing the (already-closed) lead-count gap.
- **prob=0.7 < prob=0.5:** dropping more leads over-regularizes; 0.5 is near-
  optimal (a lower prob might do better — sweep not yet run downward).

## Implication
**Lead-masking alone is the recommended deployment, not the stacked recipe** —
at least at 100 Hz on PTB-XL superclasses. This simplifies the practical recipe
considerably: one training-time augmentation, no preprocessing dance, no
test-time adaptation. The stacked recipe may earn its keep at 500 Hz (E4, where
the bandwidth axis is real) or with real watch data (E6) — both queued.

## Limitations
- 100 Hz (bandwidth axis muted — the case where matched-filter would matter
  most is exactly the one we can't see yet).
- Single seed; the 0.005–0.031 deltas are within seed-noise territory for the
  smaller ones (C3/C5/C6) but C2/C4's matched-filter drag is consistent.
- prob sweep only went up (0.5→0.7); downward (0.3) untested.

## Files
- `experiments/16_combinations.py` · `results/16_combinations/{metrics.json,combinations.png,run.log}`
