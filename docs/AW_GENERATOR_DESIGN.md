# The Apple-Watch ECG Generator — Design & Rationale

> **North-star goal (user, 2026-07-27):** a tool that turns abundant clinical
> Lead-I ECGs into Apple-Watch-style training data, so models trained on it
> transfer to real Apple Watch BETTER than models trained on raw clinical Lead-I.
> We have many clinical ECGs but few Apple Watch ones — this bridges that gap.

## Why the old forward-physics simulator failed (and what it taught us)

The original `src/watch_simulator.py` was a first attempt at this tool. The
experiment chain showed exactly why it doesn't work, and how to fix it:

| finding | experiment | implication for the generator |
|---|---|---|
| Sim over-degrades vs real (sim furthest from real AW) | E6, E6c | Don't hand-code heavy noise |
| Over-degradation is FILTER-bound, not noise-bound | E22 | Fix the spectral transfer, not the noise |
| Sim-training HURTS real transfer (0.737 < clean 0.753) | E6b | Heavy noise makes it worse than doing nothing |
| Real Apple Watch is CLEAN (kurtosis 12, entropy 0.32) | E6c | Target a clean signal + light noise |
| Real gap = lead-count/spatial, not signal noise | E1, E24 | Generator can't invent leads; focus on single-lead-answerable tasks + recording-chain match |
| CinC handheld ≈ real Apple Watch (dist 0.247) | E6c | Train the generator toward CinC (open data), validate on it as a real-AW proxy |

## The design (learned, light-touch, morphology-preserving)

`src/aw_generator.py` implements a **learned spectral transfer function**:

1. **Empirically measure** the clinical→CinC magnitude-spectrum ratio
   `H(f) = mean|CinC| / mean|clinical|` (smoothed, clipped). This is the
   *recording-chain transfer function* — learned from real data, not assumed.
2. **Apply it as a zero-phase magnitude filter**: multiply the clinical Lead-I's
   rfft magnitude by `H(f)`, KEEP the phase. Phase carries P-QRS-T timing/shape —
   the diagnosis — so **the label stays valid by construction.** No hallucination
   risk (unlike an unconstrained GAN).
3. **Add light calibrated noise + gentle baseline wander** matched to the real-AW
   residual (E6c: real AW is clean, so noise_level ~0.03, not the sim's ~0.15-0.30).

## The acceptance test (honest bar)

A generator is only worth using if **train-on-generated beats train-on-clean-
clinical-Lead-I on REAL data.** E6b established the bar: clean Lead-I → real CinC
= **0.753** (and the old heavy sim = 0.737, i.e. *worse* than the bar). E25 tests
whether the learned generator clears it. E6c validated real CinC as a real-AW
proxy (dist 0.247), so this is a legitimate real-transfer test on open data.

## Roadmap

- **Phase A (E25, DONE/running):** learned spectral transfer function. Safe, fast,
  CPU-friendly, verifiable, zero hallucination risk. The baseline generator.
- **Phase B (if A underperforms or plateaus):** unpaired neural translation
  (CycleGAN/UNIT, clinical↔CinC) with cycle-consistency + a **pathology-preservation
  classifier loss** so it cannot destroy diagnostic content. Captures nonlinear
  morphological differences the linear transfer can't.
- **Phase C:** package the winner as the reusable tool — `AWGenerator.generate(clinical_leadI)`
  → AW-style training sample. Document + provide a batch API for generating large
  training sets from any clinical Lead-I corpus (PTB-XL, MIMIC-IV-ECG, CODE-15).

## Honest limitations (state these in any report/claim)

- **Cannot recover missing spatial leads.** For tasks needing 12-lead spatial
  information (Low EF, chamber markers — E24's high-gap tasks), no single-lead
  generator helps; those need real multi-lead or a fundamentally different approach.
  The generator targets *single-lead-answerable* tasks (rhythm, rate, some intervals).
- **Fit toward CinC, not real AW directly** (HOME is eval-only). CinC is a validated
  proxy (E6c 0.247) but not identical; a labeled real-AW set would allow direct fit.
- **Label validity rests on morphology preservation** — verified by construction
  (zero-phase), but should be spot-checked (does a NORM stay NORM-shaped?).
