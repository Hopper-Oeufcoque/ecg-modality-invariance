# E37 — CORRECTED sampling-rate analysis: RETRACTS E35/E36's bandwidth gap

**Bug found and fixed.** The stale E34 (n=20) notification prompted a
data-provenance check that uncovered a real error: HOME ships Apple Watch data
in TWO files at DIFFERENT native sampling rates, and E35/E36 mislabeled one.

- `data/ecg/` (20 recordings, 15000 samples) = **500 Hz** (README + HR check:
  94 bpm @500Hz ✓). E34 used this, correctly.
- `data-for-predicting/Apple_Watch_waveform.csv` (1000 patients, 6000 samples) =
  **200 Hz** (README explicit + HR check: 66 bpm @200Hz ✓ vs 100 @500Hz).
  **E35 and E36 wrongly processed this as 500 Hz** — a 2.5× frequency-axis
  stretch. A bin labeled "40 Hz" was physically at 16 Hz.

## Corrected profiles (all resampled from CORRECT native rate to common 100 Hz)

| axis | clinical | AW-B (200Hz) | AW-A (500Hz) |
|---|---|---|---|
| kurtosis | 11.80 | 9.13 | 9.39 |
| bw_energy | 0.120 | 0.213 | 0.127 |
| qrs_energy | 0.380 | 0.268 | 0.340 |
| **hf_energy** | **0.019** | **0.015** | **0.014** |
| mid_energy | 0.312 | 0.361 | 0.354 |

## Corrected distances

| pair | distance |
|---|---|
| clinical → AW-B (200Hz) | 0.502 |
| clinical → AW-A (500Hz) | 0.253 |
| AW-A → AW-B (consistency) | 0.200 |

## Verdict: ❌ E36 RETRACTED — there is NO high-frequency/bandwidth gap

1. **Real AW hf_energy is 0.014–0.015, ≈ clinical's 0.019 — NOT the 0.16 E36
   claimed.** E36's entire "clinical is over-filtered, watch keeps HF, bandwidth
   gap" conclusion was a **sampling-rate artifact**. Fully retracted. The E36
   spectral-transfer "fix" was solving a non-existent problem (injecting HF that
   real Apple Watch does not have — the same over-injection error as E34, in a
   new disguise).
2. **The two AW sources agree** (distance 0.200) once rates are correct — strong
   internal validation of the corrected numbers.
3. **The real gap is small and mild.** Real AW is close to clinical Lead-I
   (0.25–0.50), differing mainly by modestly higher baseline wander (bw 0.13–0.21
   vs 0.12) and slightly lower kurtosis. **This VINDICATES the original E6c
   finding: real Apple Watch is a clean signal, close to clinical Lead-I.** E34
   (n=20) was right; E35/E36 (mislabeled rate) were the error, not the small N.

## What stands and what falls

- **FALLS:** E35 ("gap is bandwidth"), E36 ("spectral bandwidth match is the
  real-AW fix"), and the E36 verdict-table/log claims. All rested on the 200-Hz-
  as-500-Hz bug. Retracted.
- **STANDS:** E6c's "real AW is clean & close to clinical." The CinC-proxy method
  results (E26–E33b) are unaffected by this bug (they never touched HOME) — but
  E34's separate point remains: CinC is a so-so proxy, and real AW being clean
  means **a LIGHT touch is right** — heavy domain randomization (calibrated to
  noisy CinC) over-perturbs relative to a clean real Apple Watch signal.

## The corrected takeaway

Real Apple Watch Lead-I is close to clinical Lead-I (small, mild gap — a bit more
baseline wander, slightly lower kurtosis; NO high-frequency excess). The right
clinical→AW transform is therefore **light and morphology-preserving** (gentle
baseline-wander robustness), NOT heavy noise injection (E33/CinC over-shoots) and
NOT HF bandwidth boosting (E36, retracted — spurious). This aligns with the E6/E6b
lesson we keep re-learning: **do not over-degrade; real AW is clean.**

## Honesty flags
- Distribution study only (no AUROC; HOME cohort unlabeled).
- AW-A is only 20 recordings (40 windows); AW-B is 1000 patients — the 200 Hz
  AW-B is the more robust profile and it confirms AW-A.
- Common 100 Hz (Nyquist 50 Hz); neither source has meaningful >50 Hz content
  once rates are correct, so nothing important is lost.
- Lesson logged: **always verify sampling rate from metadata AND a physiological
  sanity check (heart rate) before spectral analysis.** Added to AGENTS.md.

## Follow-ups
- Correct EXPERIMENT_LOG verdict table + AGENTS.md status (E35/E36 retracted).
- The right real-AW recipe is LIGHT DR (already have `StochasticAWAugmenter`
  strength≈0.5); no HF transform needed.
- Still need labeled real AW for any AUROC claim (unchanged data gap).
