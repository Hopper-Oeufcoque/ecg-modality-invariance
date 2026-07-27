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

## Result of Phase A (E25b, 5-seed) — it wasn't enough, and it told us why

Acceptance test **failed for the generator-alone path** and **corrected an
earlier claim**:

| arm | AUROC (5-seed) | Δ vs clean | verdict |
|---|---|---|---|
| V1 clean Lead-I | 0.681±0.052 | — | baseline / bar |
| V2 old heavy sim | 0.729±0.039 | +0.049 (4/5) | biggest winner (!) |
| V3 generated (Phase A) | 0.676±0.035 | −0.005 (3/5) | **NEUTRAL — fails bar** |
| V4 clean + generated | 0.709±**0.012** | +0.028 (4/5) | modest mean, **stability win** |
| V5 oracle (real) | 0.930±0.014 | — | ceiling |

**The decisive insight: fidelity ≠ utility.** The *faithful* spectral generator
(V3) tied clean, while the *crude* heavy sim (V2) gained the most. The benefit
of synthetic data here is **augmentation diversity / noise-robustness
regularization**, NOT making the signal look like the target. A near-
information-preserving transform can't add training signal that isn't already
in clinical Lead-I.

**Also corrected:** E6b's single-seed "sim HURTS transfer (0.737<0.753)" was
seed noise — clean Lead-I is high-variance (0.622–0.772); over 5 seeds sim
(0.729) > clean (0.681). Fidelity-based reasoning built partly on that single
seed is downgraded accordingly.

## The acceptance test (honest bar)

A generator is only worth using if **train-on-generated beats train-on-clean-
clinical-Lead-I on REAL data** (CinC, E6c-validated AW proxy, dist 0.247). The
honest 5-seed bar is clean = **0.681±0.052** — and Phase A did not clear it.

## Roadmap (revised after E25b)

- **Phase A (E25/E25b, DONE — ⚠️ insufficient alone):** learned spectral
  transfer function. Safe/fast/verifiable, but near-information-preserving →
  neutral as a training source. Retained only inside the V4 cocktail for its
  stability contribution.
- **Phase A′ (E26, NEXT):** **stochastic augmentation generator** — replace the
  deterministic transfer with randomized, label-preserving perturbations that
  clinical data lacks: electrode-contact dropouts, motion bursts, dry-electrode
  gain wander, variable baseline dynamics. Directly tests the E25b insight
  ("diversity > fidelity") on the same 5-seed harness. This is the most likely
  path to actually *beat* clean.
- **Phase B (if A′ plateaus):** unpaired neural translation (CycleGAN/UNIT,
  clinical↔CinC) with cycle-consistency + a **pathology-preservation classifier
  loss**. Captures nonlinear morphology the linear transfer can't — but only
  worth the complexity if stochastic augmentation isn't enough.
- **Phase C:** package the winner as the reusable tool —
  `AWGenerator.generate(clinical_leadI)` → AW-style training sample, with a batch
  API for large training sets from any clinical corpus (PTB-XL, MIMIC-IV-ECG,
  CODE-15).

## Honest limitations (state these in any report/claim)

- **Cannot recover missing spatial leads.** For tasks needing 12-lead spatial
  information (Low EF, chamber markers — E24's high-gap tasks), no single-lead
  generator helps; those need real multi-lead or a fundamentally different approach.
  The generator targets *single-lead-answerable* tasks (rhythm, rate, some intervals).
- **Fit toward CinC, not real AW directly** (HOME is eval-only). CinC is a validated
  proxy (E6c 0.247) but not identical; a labeled real-AW set would allow direct fit.
- **Label validity rests on morphology preservation** — verified by construction
  (zero-phase), but should be spot-checked (does a NORM stay NORM-shaped?).
