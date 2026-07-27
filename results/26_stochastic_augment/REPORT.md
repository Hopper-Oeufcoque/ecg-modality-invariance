# E26 — Stochastic AW augmentation ("diversity > fidelity")

**Question:** E25b showed the *faithful* Phase-A spectral generator was neutral
vs clean, while the *crude* heavy sim gained most → the useful ingredient is
augmentation **diversity**, not target **fidelity**. E26 tests that directly:
does a **label-preserving stochastic augmenter** (contact dropouts, motion
bursts, dry-electrode gain wander, variable baseline, mild noise) beat clean
clinical Lead-I on real single-lead data?

**Setup:** identical harness to E25b for direct comparability — binary NORM vs
AF, PTB-XL Lead-I source (n_train=524) → CinC 2017 real target (n=1400, 350
test/seed), 5 seeds, 1D ResNet, 20 epochs. `StochasticAWAugmenter` (src/aw_generator.py)
verified label-preserving: augmented samples keep 0.94–0.98 correlation with the
source R-peak structure while differing from each other (pairwise 0.92 →
genuine diversity). Arms:
- **V1 clean** — raw clinical Lead-I (the bar)
- **V2 oldsim** — old heavy forward-physics sim (E25b's surprise winner)
- **V3s stochastic** — stochastic-augmented data *alone*
- **V4s clean+stochastic** — clean ∪ stochastic-augmented (the cocktail)
- **V5 oracle** — train on real CinC

## Results (AUROC, mean±std over 5 seeds; paired vs clean)

| Arm | AUROC | Δ vs clean | seeds + | paired t (p) |
|---|---|---|---|---|
| V1 clean Lead-I | 0.681 ± 0.052 | — | — | — |
| V2 old heavy sim | 0.731 ± 0.028 | +0.050 | 4/5 | — |
| V3s stochastic alone | 0.712 ± 0.048 | +0.031 | 3/5 | 1.05 (0.35) — n.s. |
| **V4s clean + stochastic** | **0.733 ± 0.033** | **+0.053** | **5/5** | **3.62 (0.022)** ✅ |
| V5 oracle (real→real) | 0.930 ± 0.014 | — | — | — |

## Verdict: ✅ First method to BEAT clean Lead-I on real data with significance

- **V4s (clean + stochastic augmentation) beats clean in 5/5 seeds, Δ+0.053,
  paired t=3.62, p=0.022 (two-sided; one-sided 0.011).** Even at N=5 this is
  significant. **This is the first generator/augmentation variant in the whole
  project to clear the honest acceptance bar convincingly.**
- **V3s (stochastic alone) is NOT reliable** (Δ+0.031 but 3/5 seeds, p=0.35).
  Augmentation must be *added to* clean data, not *replace* it — the cocktail is
  the mechanism, matching V4's behavior in E25b.
- **Confirms the E25b insight — diversity > fidelity.** Stochastic augmentation
  (Δ+0.053, significant) beat the faithful spectral transfer from E25b (Δ−0.005,
  neutral) despite making *no attempt* to look like Apple Watch. What helps is
  injecting label-preserving variation clinical data lacks, so the model learns
  features robust to wearable-recording artifacts.
- V4s (0.733) essentially matches the crude sim (0.731) but with **lower
  variance and a principled, tunable, morphology-guaranteed** construction — and
  unlike the sim it's a clean API on top of clinical Lead-I.

## What this means for the north-star tool

The Apple-Watch training-data generator now has a **working, validated core
recipe**: take a clinical Lead-I corpus, apply `StochasticAWAugmenter` to make
diverse label-preserving copies, and **train on clean ∪ augmented**. This
significantly improves real single-lead transfer over training on the raw
clinical data alone — exactly the north-star goal, using data we already have.

Gap to the real-data oracle (0.930) remains large — augmentation narrows the
modality gap, it doesn't close it. But it moves us from "no synthetic recipe
helps" to "a validated recipe helps, reproducibly."

## Honesty flags
- N=5 seeds, n=700 CinC (350 test/seed), single dataset pair — significant but
  small; p=0.022 is real yet not a large-scale confirmation.
- **AF vs NORM binary only.** Not tested on multi-label or spatial tasks (E24's
  high-gap tasks — Low EF, chambers — need multi-lead; augmentation won't fix
  those, per the standing limitation).
- CinC handheld is the E6c-validated proxy (dist 0.247), not real Apple Watch.
- Augmentation is label-preserving *by construction* (no time-warp/inversion;
  0.94–0.98 source corr) — but "label validity" is a morphology argument, not a
  cardiologist-verified one.
- V2 (old sim) matches V4s here — the win is "significant + principled +
  tunable," not "beats every alternative by a wide margin."
- Figure verified via PIL (1300×715); vision tool still down this session.

## Follow-ups spawned
- **E27** — sweep augmenter `strength` (0.5/1.0/1.5) and expansion factor
  (1×/2×/3×) to find the best cocktail ratio; does more diverse augmentation
  keep helping or saturate/hurt?
- **E28** — test V4s recipe on a *different pathology* (e.g. a broader rhythm/
  morphology label) to check the gain generalizes beyond AF/NORM.
- **Package (Phase C)** — the validated recipe is ready to wrap as the reusable
  tool: `AWTrainingSetBuilder(clinical_corpus) → clean+augmented training set`.
- **Phase B (neural CycleGAN)** deprioritized — a simple stochastic augmenter
  already clears the bar; only revisit if E27/E28 plateau below useful accuracy.
