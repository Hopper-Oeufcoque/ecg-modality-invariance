# E23 — Lead-masking validated on REAL single-lead (decisive real-deployment test)

**Date:** 2026-07-27 · **Status:** ⚠️⚠️⚠️ **Lead-masking catastrophically fails on real data — the 12-lead prior doesn't survive real shift**

## Hypothesis (pivoted by E6b)
E6b refuted the simulator strategy (clean Lead-I 0.753 > sim 0.737 on real). The
revised ranking put lead-masking (E2, trains on real clinical, no sim noise to
overfit) as the most realism-robust label-free method — but that claim was only
validated on SIMULATED watch (E2 L4=0.718 on sim). E23 tests it on REAL data:
does lead-masking's 12-lead prior beat clean single-lead training on REAL
single-lead, as it did on sim (E2 V2 0.717 > V5 0.690)?

## Setup
Binary NORM-vs-AF (same label alignment as E6b). All tested on REAL CinC 2017
(700 N / 700 A). In-experiment seeds for fair comparison.
- V1 lead-masking: 12-lead model w/ random lead dropout, test on real CinC
  (Lead-I in ch0, zero-pad — the E2 eval protocol)
- V2 clean Lead-I: 1-lead model trained on clean PTB-XL Lead-I
- V3 sim-trained: 1-lead model trained on sim-watch Lead-I
`experiments/23_leadmask_real_cinc.py`. 20 ep, single seed.

## Results (AUROC on REAL CinC 2017, binary NORM-vs-AF)

| variant | AUROC | what |
|---|---|---|
| V1 lead-masking (12-lead prior) | **0.557** | catastrophic — near chance |
| V2 clean Lead-I (1-lead) | 0.721 | robust |
| V3 sim-trained (1-lead) | 0.731 | robust |

E6b cross-references: V2 clean = 0.753, V3 sim = 0.737, V4 oracle = 0.946
(seed variance accounts for the small diffs; the ranking is robust).

## Verdict: ⚠️⚠️⚠️ — the 12-lead lead-masking model catastrophically fails on real

**V1 lead-masking (0.557) << V2 clean (0.721) and V3 sim (0.731).** The project's
sim-validated "winner" (lead-masking, 0.718 on sim) collapses to near-chance on
real single-lead data — a 0.161 drop. **The 12-lead prior does NOT survive real
single-lead domain shift.** In fact it's far WORSE than the simple single-lead
models it beat on sim.

## Why — the mechanism
The 12-lead lead-masking model was trained on PTB-XL with lead dropout, but the
Lead-I channel always carried *real PTB-XL Lead-I* statistics (clean clinical).
At test on real CinC:
1. **BN normalization mismatch:** the 11 zero-padded channels + 1 CinC Lead-I
   channel feed BN running stats (computed on PTB-XL). CinC Lead-I (different
   electrode/population) violates the assumed statistics → the 12-lead model's
   BN mis-normalizes, breaking features.
2. **Multi-lead structure reliance:** the 12-lead model learned inter-lead
   relationships that, with 11 channels zeroed, produce an out-of-distribution
   input the model never saw at the *real*-device statistics.

The single-lead models (V2/V3) don't have 11 zero channels disrupting BN, so
they're robust to the Lead-I domain shift — they only face the (smaller) shift of
Lead-I statistics, not the compounded 12-channel structure mismatch.

## What this means for the project (second major reframe, with E6b)
The sim-validated ranking is **inverted on real data:**

| method | on SIM | on REAL |
|---|---|---|
| lead-masking (12-lead) | 0.718 (best) | **0.557 (worst)** |
| single-lead + sim | 0.742 (best) | 0.731 (robust) |
| clean single-lead | 0.690 (ref) | 0.721 (robust) |

**The project's two sim-validated "winners" (lead-masking and sim-trained) do NOT
hold on real data.** Lead-masking fails catastrophically; sim-training ≈ clean
single-lead. The robust real-deployment finding: **train a SINGLE-LEAD model**
(clean Lead-I or lightly sim-augmented) — the simpler architecture is far more
robust to real domain shift than the 12-lead lead-masking prior.

Combined E6b + E23 conclusion: **all sim-validated results carry realism debt.**
The methods that looked best on sim (12-lead prior, aggressive sim) fail or don't
help on real. The methods that looked mediocre on sim (simple single-lead) are
the robust ones. This is a textbook sim-overfitting result: the simulator's
specific (PTB-XL-derived) statistics favored methods that exploited them, which
then fail on real (different-device) data.

## Honesty flags
- Single seed; binary AF/NORM only (spatial classes unmappable). The 0.557 vs
  0.72 gap is large enough to be robust to seed, but the exact margins aren't.
- CinC handheld lead-I (cleaner than wrist dry-electrode) — a cleaner target
  than Apple Watch, which makes lead-masking's failure here *more* damning.
- PTB-XL vs CinC population confound — but V1 vs V2/V3 are all PTB-XL-trained,
  so the population axis is held constant; the difference is the architecture
  (12-lead+zero-pad vs 1-lead), isolating the mechanism.
- The 0.557 may be partly a BN-mechanics artifact (not just "the prior is bad");
  either way the practical conclusion holds: 12-lead lead-masking is not
  real-deployable as-is. A fix (test-time BN recompute on real, E5-style) is queued.

## Follow-ups
- **E5b — test-time BN adaptation on real:** recompute BN running stats from the
  real CinC batch before predicting (E5 was marginal on sim, but E23's failure
  looks BN-driven — this might recover much of the 0.557→0.72 gap).
- **Single-lead + MixStyle (E15 V3) on real:** E15 helped single-lead+sim on sim
  (0.746); does the gain hold on real? The single-lead path is the robust one.
- **Real watch data** — the only path to the 0.946 oracle ceiling (E6b V4).

## Artifacts
- `results/23_leadmask_real_cinc/metrics.json`
- `results/23_leadmask_real_cinc/leadmask_real.png`
