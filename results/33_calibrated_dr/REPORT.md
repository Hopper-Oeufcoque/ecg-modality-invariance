# E33 — Target-calibrated domain randomization: ✅ breaks the 0.80 plateau (zero-shot)

**Question:** hand-tuned augmentation plateaus ~0.80 (E27b) and all test-time
adaptation fails (E29/E32). Domain randomization only gives zero-shot transfer
if the training distribution actually COVERS the target. Does calibrating the
augmenter to the MEASURED real-target statistics (from unlabeled data) — so the
augmented training distribution envelopes the watch domain — push zero-shot past
the plateau?

**Setup:** measure per-axis signal stats (baseline-wander/QRS/HF band energy,
kurtosis) on unlabeled CinC ref; build `CalibratedAugmenter` whose perturbation
ranges cover clinical→target (cover=1.3×) on each axis. Train on clinical Lead-I
+ calibrated augmentation; test on real CinC AF/NORM. 5 seeds. Zero target labels
(calibration uses target signal *statistics* only).

## Results (AUROC mean±std, 5 seeds)

| Arm | AUROC | Δ vs hand-tuned | seeds + |
|---|---|---|---|
| clean Lead-I | 0.681 ± 0.052 | — | — |
| augment hand-tuned (E27b) | 0.783 ± 0.021 | — (bar) | — |
| **augment CALIBRATED** | **0.824 ± 0.016** | **+0.042** | **5/5** |
| oracle (real→real) | 0.931 ± 0.006 | — | — |

**Calibrated DR closes ~57% of the clean→oracle gap ZERO-SHOT** (clean 0.681 →
0.824, oracle 0.931), up from hand-tuned's ~41%. First method to break the 0.80
plateau without any target labels.

## The morphology-guard correction (important methodology fix)

The naive raw-Pearson morph guard flagged calibrated DR at **0.73 < 0.85** —
apparently "label-risky." A direct check proves that guard was measuring the
WRONG thing:

| morphology metric | value | interpretation |
|---|---|---|
| raw Pearson corr | 0.69 | LOW — but dominated by added baseline wander |
| **QRS-band corr (>1 Hz high-pass)** | **0.966** | QRS shape essentially intact |
| **R-peak location match (±50 ms)** | **0.976** | 98% of R-peaks preserved (rhythm intact) |

Calibrated DR deliberately injects **baseline wander** because the real target
genuinely has ~2.4× more low-freq energy (target bw_energy 0.242 vs clinical
0.100). That low-frequency content is **label-irrelevant** — it tanks raw
correlation but leaves the diagnostic signal (QRS morphology + R-R rhythm, where
the AF/NORM label lives) 97% preserved. **The label is valid.** This is genuine
target coverage, not corruption.

→ **Guard corrected:** the right morphology-preservation metric is QRS-band
correlation + R-peak preservation, NOT raw Pearson. E27b's strength-2.5
rejection should be re-examined under this better metric (it may have been
partly a false alarm too — logged as follow-up E27c).

## Why this worked where everything else failed

- **Test-time adaptation (E29/E32) failed** because you can't recover missing
  boundary information by adjusting a fixed model on unlabeled data.
- **Calibrated DR works** because it puts the target-domain variation INTO the
  training distribution — the model learns the right boundary *while training*,
  covering the real recording chain (measured, not guessed). This is the correct
  mechanism for zero-shot domain randomization: cover the target, don't adapt to
  it afterward.

## Honesty flags
- 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c proxy not real AW.
- Still a gap to oracle (0.824 vs 0.931) — calibrated DR does NOT fully close
  zero-shot; ~43% of the gap remains, likely needing real labels (E30) or
  richer coverage.
- Calibration uses unlabeled target signal statistics (realistic "we have some
  unlabeled watch recordings" assumption) — effectively zero-shot for LABELS,
  but not truly "target-blind." A fully target-blind version (wide priors, no
  measurement) is worth testing (E33b).
- Coverage is on summary stats (band energies, kurtosis), not a full-manifold
  guarantee — the residual gap may be un-measured axes.
- Morphology verified quantitatively (QRS corr 0.966, R-peak 0.976); figure
  verified via PIL. Vision tool down this session.

## Follow-ups spawned
- **New best zero-shot recipe: calibrated domain randomization (0.824).**
  Supersedes hand-tuned augmentation as the packaged tool's core.
- **E27c** — re-audit E27b's strength-2.5 rejection with the corrected QRS-band
  guard (was it a real mirage or a raw-Pearson false alarm?).
- **E33b** — stack calibrated DR + few-shot k=50 (E30): does 0.824 + 50 labels
  reach oracle? This is the likely path to ~0.90+ with minimal labels.
- **E33c** — target-blind wide-prior DR (no measurement) to test how much the
  measurement itself matters.
- Update `src/aw_generator.py` with `CalibratedAugmenter` and the QRS-band guard.
