# E6 — Real single-lead validation of the forward-physics simulator

**Date:** 2026-07-27 · **Status:** ⚠️ **Important honesty flag — simulator over-degrades relative to real single-lead**

## Hypothesis
The forward-physics watch simulator (E1, keystone of Solution 1 & 2) produces
single-lead ECGs whose *distribution* matches real single-lead recordings. If
yes → the E17 sim-trained winner is trustworthy as a stand-in; if no → the sim
results carry a hidden realism debt.

## Setup
Compared three distributions on 256 clinical / 256 simulated / 300 real records:
- **Clinical Lead-I** — raw PTB-XL Lead-I (test split, max_per_class=200).
- **Simulated watch** — full forward-physics sim (bandpass + dry-electrode +
  noise + quantization) applied to PTB-XL Lead-I.
- **Real single-lead** — PhysioNet CinC 2017 (handheld lead-I electrode, 9–60 s,
  downsampled 300→100 Hz), closest public Apple-Watch analog.

Distribution-level metrics (no label alignment needed):
1. PSD band-energy fractions (<5 Hz drift / 5–40 Hz ECG / 40–50 Hz noise)
2. Baseline-wander amplitude (fraction of signal std)
3. Sample entropy (signal complexity / irregularity)
4. DFA α exponent (long-range correlation)
5. Kurtosis (morphology peakedness — sharp QRS + flat baseline ⇒ high)

Distance = mean abs z-score between two distributions (lower = closer).
The classifier cross-over (metric 6 in the docstring — train sim, test real) was
**not implemented** in this run; queued as E6b.

## Results

| metric | clinical Lead-I | simulated watch | real single-lead |
|--------|----------------:|----------------:|------------------:|
| baseline_wander | 0.307 | **0.435** | 0.190 |
| sample_entropy  | 0.480 | **0.818** | 0.282 |
| dfa_alpha       | 0.638 | 0.784 | 0.912 |
| kurtosis        | 16.07 | **4.77** | 17.71 |
| std             | 1.000 | 0.776 | 1.000 |
| PSD [lo,mid,hi] | [.32, .57, .002] | [.36, .36, .009] | [.56, .38, .0004] |

**Pairwise distances (mean abs z-score across 4 stats; lower = closer):**
- `sim_vs_real`      = **1.077**  ← largest
- `sim_vs_clinical`  = 0.995
- `real_vs_clinical` = **0.717**  ← smallest

## Verdict: ⚠️ The simulator over-degrades; real single-lead is closer to clinical than to our simulation

Three findings, in order of importance:

1. **The simulated distribution is FURTHER from real single-lead than raw
   clinical Lead-I is** (1.077 vs 0.717). The simulator's transformations push
   the signal *away* from the real-watch distribution, not toward it. This is the
   opposite of the intended purpose and a direct honesty flag on every sim-based
   result (E1–E17).

2. **The over-degradation is systematic and large:**
   - *sample_entropy* 2.9× too high (sim 0.818 vs real 0.282) — the added noise
     inflates signal complexity well beyond real watch.
   - *kurtosis* 3.7× too low (sim 4.77 vs real 17.71) — added noise flattens the
     sharp-QRS / flat-baseline morphology that defines real single-lead. Real
     watch is *peakier* than clinical, not smoother.
   - *baseline_wander* 2.3× too high (sim 0.435 vs real 0.190).

   The dry-electrode + motion-noise model is too aggressive. Real handheld
   single-lead is cleaner than the simulator produces.

3. **Real single-lead ≈ clinical Lead-I on morphology** (kurtosis 17.71 vs 16.07,
   distance 0.094 — nearly identical). The dominant real-watch characteristic is
   *peakiness*, which the clinical signal already has and the simulator destroys.

## What this means for the project (honest reframing)

- **E17's win carries a realism debt.** E17 (single-lead trained on the sim)
  beat lead-masking (0.742 vs 0.718) *on simulated watch*. But because the sim
  doesn't match real watch, E17 may be over-fitting to the simulator's
  over-aggressive noise profile — the edge may not transfer to real watch. This
  does NOT invalidate E17, but downgrades its claim from "new best method" to
  "best on a sim that needs recalibration."

- **Lead-masking (E2, 0.718) may be more robust to real watch**, paradoxically
  because it trains on real clinical + zero-pad (no simulated noise) and so
  cannot over-fit a wrong noise model. The label-free lead-masking result is the
  most *realism-robust* finding in the project so far.

- **The simulator direction is right, magnitude is wrong.** Watch *is* noisier
  than clinical (correct sign on baseline_wander, entropy), but the CinC
  reference is handheld lead-I (finger contact) — *cleaner* than wrist dry-
  electrode. So the simulator targeting more noise than CinC is partially
  defensible (Apple Watch wrist dry-electrode is genuinely noisier than handheld),
  but the 2.9–3.7× over-shoot exceeds even the dry-electrode justification.

## Honesty flags
- Single seed, 256 sim / 256 clinical / 300 real records.
- CinC 2017 is handheld lead-I, NOT wrist dry-electrode — a cleaner reference
  than the actual Apple Watch target. The sim/real gap on a *wrist dry-electrode*
  dataset could be smaller (simulator more justified) or larger (different noise
  color). **This is a reference-dataset limitation, not a simulator pass.**
- Distribution-level only; no task-level cross-over (E6b queued).
- No CIs; z-score distances are point estimates.

## Lessons
1. A forward-physics simulator must be *calibrated* against real target data, not
   just validated for axis structure (E1 did the latter, E6 does the former).
2. Kurtosis is the sharpest discriminator — real single-lead is peaky (high
   kurtosis); additive broadband noise destroys this. The simulator should
   preserve QRS peakedness, not flatten it.
3. "Closer to real than the source" is the right success criterion for a sim; the
   current sim fails it (real_vs_clinical < sim_vs_real).

## Follow-ups (spawned)
- **E6b — classifier cross-over:** train on sim, test on real CinC → quantify the
  sim/real gap at the *task* level (the missing metric 6). Determines whether the
  E17 edge survives contact with real data.
- **E-sim-calib — simulator recalibration:** reduce noise std and baseline-wander
  amplitude to match CinC stats (target kurtosis ≥ 15, entropy ≤ 0.4), then
  re-run E17 to see if the single-lead+sim edge holds, grows, or vanishes.
- **E-sim-dryelec:** obtain a wrist dry-electrode dataset (e.g. Apple Watch
  research data) to validate against the true target rather than the handheld
  proxy.

## Artifacts
- `results/06_sim_validation/metrics.json` — full per-domain means + distances
- `results/06_sim_validation/sim_validation.png` — 4-metric bar comparison
- `results/06_sim_validation/psd_bands.png` — PSD band-energy distribution
