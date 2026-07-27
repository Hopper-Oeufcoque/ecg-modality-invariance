# E6b — Classifier cross-over: train sim, test REAL (the definitive sim/real test)

**Date:** 2026-07-27 · **Status:** ⚠️⚠️⚠️ **Critical — the simulator HURTS real transfer; E17's edge does not survive contact with real data**

## Hypothesis (spawned by E6)
E6 found the simulator over-degrades at the distribution level. E6b is the
*task-level* test: does a model trained on simulated watch actually detect
disease on REAL single-lead ECG? Binary NORM-vs-AF (AF maps cleanly in both
PTB-XL and CinC 2017).

## Setup
4 variants, 1-lead ECGResNet1d, binary, 20 ep, single seed:
- **V1** sim→sim: train PTB-XL sim-watch, test PTB-XL sim-watch (favorable match)
- **V2** sim→real: train PTB-XL sim-watch, test REAL CinC 2017 (cross-over)
- **V3** clean→real: train PTB-XL clean Lead-I, test REAL CinC 2017 (control)
- **V4** real→real: train REAL CinC, test REAL CinC (oracle ceiling)

`experiments/06b_classifier_crossover.py`. PTB-XL: 524 train / 143 test. CinC:
1400 total (700 N / 700 A), 1120 train / 280 test (80/20 for V4).

## Results

| variant | AUROC | ACC | n | what it measures |
|---|---|---|---|---|
| V1 sim→sim | **0.993** | 0.951 | 143 | sim->sim (overfit sim artifacts) |
| V2 sim→real CinC | **0.737** | 0.694 | 1400 | sim→real (the cross-over) |
| V3 clean→real CinC | **0.753** | 0.686 | 1400 | clean→real (control) |
| V4 real→real (oracle) | **0.946** | 0.775 | 280 | real-data ceiling |

## Verdict: ⚠️⚠️⚠️ — the simulator HURTS real transfer

Three findings, in order of importance:

1. **The sim/real transfer debt is massive (0.256).** V1 sim→sim = 0.993 (near-
   perfect — the model overfit the simulator's specific noise/filter profile),
   collapsing to V2 sim→real = 0.737. The 0.993 is a red flag: the model learned
   sim-specific artifacts, not just ECG content. This is the direct task-level
   confirmation of E6's distribution-level over-degradation finding.

2. **Clean Lead-I training BEATS sim training on real data (V3 0.753 > V2 0.737).**
   Training on the simulator's over-degraded output is *actively counterproductive*
   for real-world deployment — it's worse than just training on clean clinical
   Lead-I. The simulator adds noise that (a) doesn't match real watch noise and
   (b) the model overfits, degrading real transfer. This refutes the E17 strategy
   (single-lead+sim as a from-scratch training distribution) for real deployment.

3. **The real-data ceiling (V4 0.946) is far above both sim/clean approaches.**
   There's a 0.209 gap between sim-trained (0.737) and the oracle (0.946) — most
   of the improvement potential requires real target-domain data. No amount of
   simulation closes this; the information isn't in the clinical source.

## What this means for the project (major reframe)

**The simulator strategy — the keystone of both Solution 1 and Solution 2 in the
synthesis report — is refuted for real deployment.** The chain of evidence:
- E6: simulator over-degrades (sim further from real than clinical is)
- E22: over-degradation is filter-bound; recalibration helps marginally (+0.012)
- **E6b: the over-degradation HURTS real transfer — clean Lead-I beats sim training**

**E17's "win" (0.742 on sim) does not transfer.** It was a sim→sim artifact —
the 0.993 V1 shows the model overfit the sim's noise profile, and the edge
vanishes on real data where V3 (clean) > V2 (sim).

**Revised project ranking:**
1. **Lead-masking (E2, 0.718 on sim)** — trains on real clinical, no sim noise to
   overfit; most realism-robust. Still the best *label-free* method.
2. **Clean Lead-I training (V3, 0.753 on real)** — actually better than sim for
   real deployment; the 12-lead prior via lead-masking may add more.
3. **Simulator** — valuable as an *evaluation probe* (E1 axis decomposition) and
   *alignment training signal* (E3-B), NOT as a from-scratch training distribution.
4. **Real watch data** — the only path to the 0.946 ceiling; no simulation closes
   the 0.209 gap.

## Honesty flags
- Single seed; binary AF/NORM only (spatial classes MI/STTC/HYP have no clean CinC
  mapping — this tests rhythm-pathology transfer specifically).
- CinC is handheld lead-I (cleaner than wrist dry-electrode) — a *cleaner* target
  than Apple Watch, which makes the sim's failure here *more* damning (if it can't
  transfer to a clean reference, it won't transfer to a noisier real watch).
- Different populations (PTB-XL clinical cohort vs CinC challenge cohort) confound
  the sim/real axis — some of V3>V2 may be population match, not just clean-vs-sim.
  V4 (same dataset train/test) controls this; the V4 ceiling is the clean comparison.
- V3 ACC (0.686) < V2 ACC (0.694) despite V3 AUROC > V2 — threshold/calibration
  difference; AUROC (threshold-independent, imbalance-robust) is the reliable metric.

## Follow-ups
- **E22b (scripted, ready)** — bandpass redesign to recover kurtosis; if a
  realistic sim is achievable, re-run E6b V2 with it. But E6b suggests the sim
  approach is fundamentally limited regardless of calibration.
- **Lead-masking on real CinC:** does E2's lead-masking (12-lead prior) beat V3
  clean Lead-I on real data? The definitive label-free real-transfer test.
- **Real watch data** — the only path to the 0.946 ceiling.

## Artifacts
- `results/06b_classifier_crossover/metrics.json`
- `results/06b_classifier_crossover/crossover.png`
