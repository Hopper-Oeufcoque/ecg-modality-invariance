# E9 — REx (Risk Extrapolation) across simulated environments

**Date:** 2026-07-27 · **Status:** ⚠️ Near-neutral on sim — and on the (now-refuted) sim domain anyway

## Hypothesis
REx (Risk Extrapolation, Krueger et al. 2021 — the report's Solution-2 keystone,
*never on ECG modality*) penalizes cross-environment risk variance across 4
simulator variants (low/high noise, dry-electrode, motion-heavy) to force
training-time invariance to device noise. The by-construction counterpart to
E10's neutral post-hoc INLP — can preventing noise reliance at train time succeed
where post-hoc removal failed?

## Setup
4 environments via WatchSimConfig variants (low_noise, high_noise, dry_electrode,
motion_heavy). REx loss = mean_env[R] + λ·var_env[R]. λ∈{0(ERM), 0.5, 1.0, 2.0}.
Single-lead model, 5-class, 20 ep. `experiments/09_rex_environments.py`.

## Results

| λ | L1 | L4 |
|---|---|---|
| 0.0 (ERM) | 0.726 | 0.726 |
| 0.5 | 0.739 | 0.726 |
| 1.0 | 0.731 | **0.733** |
| 2.0 | 0.731 | **0.733** |

## Verdict: ⚠️ Near-neutral (and on the refuted sim domain)
REx at λ=1.0/2.0 gives +0.007 over ERM (0.733 vs 0.726) — within seed noise.
**Why it didn't help:** the cross-environment risk variance is consistently low
(var_risk ≈ 0.005-0.038 throughout training) — the 4 environments have *similar*
risk, so REx has little variance to penalize. The environments differ in noise
magnitude but not enough to create distinct shortcuts the model takes.

This is consistent with the broader finding (E10 neutral, E15 marginal): the gap
is dominated by lead-count info loss (E1), not an avoidable noise shortcut. REx
would only help if the model took device-specific shortcuts that fail across
environments — but the environments are too similar to surface such shortcuts.

**Critical caveat (post-E6b):** E9 is on *simulated* watch, which E6b showed does
NOT transfer to real (sim 0.737 ≈ clean 0.753 on real, and the 12-lead lead-masking
"winner" collapses to 0.557 on real). So E9's near-neutral result is on a domain
whose real-deployment relevance is now questionable. REx's value would need
re-testing on real data — but there, the environments would be real device variants,
which is the proper test.

## Honesty flags
Single seed; 4 environments are SIM variants (not clinical-vs-real); var_risk low
(environments too similar — a wider environment spread might surface more
shortcuts); on the now-refuted sim domain.

## Lesson
REx is the report's Solution-2 keystone, and it's near-neutral here — but the
more important lesson is methodological: a domain-generalization method's value
depends on the environments spanning a real shortcut axis. Sim-noise environments
are too similar to create one. The proper REx test would use REAL device variants
as environments (e.g., different Apple Watch generations, dry vs wet electrode)
— which requires real data, returning to the E6b conclusion that real data is the
binding constraint.

## Follow-ups
- Re-test REx with REAL device-variant environments if such data is obtained.
- The E9/E10/E15 training-time-invariance trio (REx/INLP/MixStyle) is uniformly
  near-neutral on sim — consistent with the gap being info loss, not a removable
  shortcut. Focus shifts to real data and lead recovery (B1 synthesis).

## Artifacts
- `results/09_rex_environments/metrics.json`
- `results/09_rex_environments/rex.png`
