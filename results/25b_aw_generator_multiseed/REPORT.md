# E25b — AW-Generator acceptance test (5-seed)

**Question:** Does training on *generated* Apple-Watch-style data (Phase-A
learned spectral transfer + light calibrated noise) beat training on raw
**clean clinical Lead-I** on real single-lead data (CinC handheld, the validated
real-AW proxy from E6c)? This is the honest north-star acceptance test: a
generator is only worth building if train-on-generated > train-on-raw.

**Setup:** binary NORM vs AF. PTB-XL Lead-I (n_train=524) as clinical source;
CinC 2017 (n=1400, 350 test/seed) as real-AW proxy. 5 seeds, mean±std, paired
per-seed deltas. 1D ResNet, 20 epochs. Five arms:
- **V1 clean** — train on raw clinical Lead-I
- **V2 oldsim** — train on old heavy forward-physics simulator output
- **V3 generated** — train on Phase-A AW-generator output *(the tool under test)*
- **V4 clean+gen** — train on clean ∪ generated (augmentation)
- **V5 oracle** — train on real CinC (upper bound)

## Results (AUROC, mean±std over 5 seeds)

| Arm | AUROC | Δ vs clean | seeds + |
|---|---|---|---|
| V1 clean Lead-I | 0.681 ± 0.052 | — | — |
| V2 old heavy sim | **0.729 ± 0.039** | **+0.049 ± 0.064** | 4/5 |
| V3 generated (Phase A) | 0.676 ± 0.035 | −0.005 ± 0.043 | 3/5 |
| V4 clean + generated | 0.709 ± **0.012** | +0.028 ± 0.058 | 4/5 |
| V5 oracle (real→real) | 0.930 ± 0.014 | — | — |

## Verdict: ⚠️ Phase-A generator FAILS as a standalone training source; modest + as augmentation

1. **V3 (generated-alone) is NEUTRAL** vs clean Lead-I: Δ = −0.005 ± 0.043,
   only 3/5 seeds positive. **Fails the acceptance bar.** A morphology-preserving
   spectral transfer + light noise does *not*, on its own, produce better
   training data than the raw clinical Lead-I it was built from. Makes sense:
   the transform is nearly information-preserving, so the trained model sees
   essentially the same content.

2. **V4 (clean + generated augmentation) gives a modest, noisy positive**
   (+0.028 ± 0.058, 4/5 seeds) — the mean gain overlaps zero, so *not*
   significant on effect size alone. **But its variance collapses to 0.012**
   (vs clean's 0.052 and oldsim's 0.039) — by far the most *stable/reproducible*
   arm. Practical value here is **training stability**, not headline AUROC.

3. **Surprise — the old heavy sim (V2) is the biggest single winner**
   (+0.049, 4/5 seeds), which **contradicts E6b** (single-seed: sim 0.737 <
   clean 0.753). Reconciliation below.

## Reconciling the E6b contradiction (important honesty correction)

E6b concluded "simulator training HURTS real transfer" from a **single seed**
where clean=0.753, sim=0.737. This 5-seed run shows clean Lead-I is
**extremely high-variance** (0.622 → 0.772 across seeds, std 0.052). E6b
happened to land on a *lucky clean split*. Averaged over 5 seeds, the picture
**inverts**: sim (0.729) beats clean (0.681). 

→ **E6b's "sim hurts" claim was single-seed noise and is hereby downgraded.**
The multi-seed truth is narrower and more useful: *noise/distribution
augmentation (whether the crude hand-coded sim or a learned transform) gives a
small robustness gain on a noisier real target — and the gain is dominated by
seed variance, not by simulator fidelity.*

## What this tells us about the north-star tool

- The win mechanism is **augmentation diversity / noise-robustness
  regularization**, NOT spectral fidelity. Making generated data *look more
  like* the target didn't beat crudely *perturbing* it. Fidelity ≠ utility here.
- **Phase A is not enough.** A near-information-preserving filter can't add
  training signal that isn't already in clinical Lead-I. To actually *help*, the
  generator must introduce *useful, label-preserving variation* the clinical
  data lacks (electrode-contact dropouts, motion bursts, dry-electrode
  distortion, baseline dynamics) — i.e. **stochastic augmentation, not
  deterministic transfer.**
- Best practical recipe emerging: **clean ∪ light-augmented** as a multi-source
  cocktail (V4's stability is real and valuable), with the augmentation tuned
  for *diversity* not *realism*.

## Honesty flags
- 5 seeds, n=700 CinC (350 test/seed) — small; deltas dominated by seed variance.
- AF/NORM binary only — not multi-label, not spatial tasks.
- CinC handheld is a *proxy* for real AW (E6c dist 0.247), not real AW itself.
- Generator fit toward CinC distribution; preserves QRS morphology (label-valid).
- No confidence intervals beyond ±std; effect sizes for V2/V4 overlap zero.
- Figure verified via PIL (dims 1300×715) — vision tool still down this session.

## Follow-ups spawned
- **E26** — replace deterministic transfer with **stochastic augmentation**
  (random contact dropouts, motion bursts, dry-electrode gain wander); re-run
  this exact 5-seed harness. Tests the "diversity > fidelity" hypothesis directly.
- **Phase B** — neural CycleGAN clinical↔CinC + pathology-preservation loss,
  only if E26's stochastic augmentation plateaus.
- Re-audit any other single-seed verdicts in the log for seed-variance fragility.
