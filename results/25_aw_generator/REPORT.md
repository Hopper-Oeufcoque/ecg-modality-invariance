# E25 — Apple-Watch generator validation (single-seed, superseded by E25b)

**Date:** 2026-07-27 · **Status:** ⚠️ Promising but single-seed & noisy — see E25b for the multi-seed verdict

## Goal
Test the north-star tool (`src/aw_generator.py`): does training on GENERATED
Apple-Watch-style data (clinical Lead-I → learned spectral transfer → AW-style)
transfer to real data BETTER than training on raw clinical Lead-I?

## Setup
Binary NORM-vs-AF, tested on REAL CinC 2017 (validated AW proxy, E6c dist 0.247).
Transfer function fit on a held-out CinC ref split (labels unused, no test leakage).
5 variants, single seed. `experiments/25_aw_generator.py`.

## Results (single seed, real CinC test n=700)

| variant | AUROC | ACC |
|---|---|---|
| V1 clean Lead-I | 0.622 | 0.571 |
| V2 old heavy sim | 0.810 | 0.610 |
| V3 GENERATED AW | 0.669 | 0.597 |
| V4 clean + generated | 0.707 | 0.660 |
| V5 oracle real→real | 0.928 | 0.850 |

Learned transfer function H(f): 501 bins, range [0.37, 2.42], mean 0.72 (net
attenuation — CinC has less high-freq energy than clinical, consistent with a
band-limiting recording chain).

## Verdict: ⚠️ Do not trust the single-seed numbers — E25b required
Two results point in useful directions but the seed is too noisy to conclude:

1. **V3 generated (0.669) > V1 clean (0.622), and V4 clean+generated (0.707) >
   both** — the generator cleared the acceptance bar *this seed*, and using it as
   augmentation (V4) helped most. Encouraging directional signal.

2. **BUT V1 clean = 0.622 here vs 0.753 in E6b** — this CinC test split (n=700, half
   the E6b set) is unrepresentative; and V2 old-sim = 0.810 contradicts E6b's 0.737.
   Single-seed variance on a small split is large. **Neither the encouraging V3>V1
   nor the surprising V2 win can be trusted from one seed.**

## Action taken
Launched **E25b** — the same comparison across 5 seeds with reshuffled splits and
re-init, reporting mean±std and paired Δ(V3−V1), Δ(V4−V1) per seed. That is the
statistically honest verdict on whether the generator works. **Read E25b's report
for the conclusion; treat E25's numbers as a noisy preview only.**

## Honesty flags
- Single seed, n=700 CinC (one unrepresentative split — V1 far below its E6b value).
- The generator preserves morphology by construction (zero-phase magnitude filter),
  so labels stay valid — but this was not separately spot-checked here.
- Transfer function fit toward CinC (real-AW proxy), not real AW directly.

## Artifacts
- `results/25_aw_generator/metrics.json`
- `results/25_aw_generator/generator_validation.png` (fig verified via PIL dims;
  vision tool errored this session)
- **Superseded by `results/25b_aw_generator_multiseed/`**
