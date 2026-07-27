# E29 — Unsupervised Domain Adaptation to close the gap: ❌ all methods HURT

**Question:** the 0.73→0.93 gap is pure recording-modality shift (single-lead
vs single-lead, not lead-count loss), so it *should* be attackable with domain
adaptation using UNLABELED real target data. Test AdaBN, Deep CORAL, and DANN
stacked on the E26 winning recipe (source = clinical Lead-I + stochastic aug),
aligning to unlabeled real CinC.

**Setup:** binary NORM/AF, PTB-XL Lead-I source → real CinC target, 5 seeds, 1D
ResNet. UDA uses cinc_ref **signals only** (no labels); eval on held-out
cinc_test. Feature = 32-d penultimate layer.

## Results (AUROC, mean±std over 5 seeds; Δ vs A1 = E26 recipe)

| Arm | AUROC | Δ vs A1 | seeds + |
|---|---|---|---|
| A0 clean Lead-I | 0.681 ± 0.052 | — | — |
| **A1 clean + stochastic (E26)** | **0.747 ± 0.040** | — (bar) | — |
| A2 + AdaBN | 0.706 ± 0.044 | −0.041 | 0/5 |
| A3 + Deep CORAL | 0.675 ± 0.035 | −0.072 | 0/5 |
| A4 + DANN | 0.638 ± 0.045 | −0.109 | 1/5 |
| V5 oracle (real→real) | 0.941 ± 0.004 | — | — |

## Verdict: ❌ Feature-space alignment does NOT close the gap — it widens it

- **All three UDA methods hurt**, monotonically with aggressiveness: AdaBN
  (mildest) −0.041, CORAL −0.072, DANN (most aggressive) −0.109. None positive
  in more than 1/5 seeds.
- **AdaBN hurting** is the most diagnostic result: recomputing BN statistics on
  the real target made things *worse*, which means the source (augmented) and
  target BN statistics are **already reasonably matched** — the augmentation
  brought first/second moments close, so overwriting them with the smaller
  target-ref set just adds estimation noise. The residual gap is **not** a
  normalization-statistics problem.
- **CORAL/DANN hurting** is the deeper signal: forcing source and target
  *feature distributions* to align **destroys class-discriminative structure**.
  Because AF-vs-NORM is genuinely harder in the real single-lead domain, pulling
  the clinical features to match the target manifold drags them off the
  decision boundary. Classic UDA failure mode when the label-conditional
  structure differs across domains (conditional shift, not just covariate shift).

## What this tells us about the gap

The 0.73→0.93 gap is **NOT** a feature-misalignment problem you can fix by
surgically matching distributions. The oracle (0.94) wins because it *learned
the real domain's class boundary directly* — and no amount of unsupervised
feature-aligning to the target reconstructs that boundary without labels. This
strongly implies:

1. The gap is closed by **data/label information**, not representation geometry.
2. The productive lever is making the model **see the right variation during
   training** (augmentation — E27 confirms this scales) or getting **a few real
   labels** (semi-supervised — the E30 direction), NOT unsupervised alignment.

## Honesty flags
- 5 seeds, n=700 CinC (350 test/seed), AF/NORM only, CinC = E6c proxy not real AW.
- UDA hyperparameters (CORAL weight 10×ramp, DANN λ schedule) were reasonable
  defaults, not exhaustively tuned — a heavily-tuned DANN *might* recover to
  neutral, but the consistent negative trend across three independent methods
  makes "UDA is the answer here" unlikely. Logged as a clear negative.
- Adaptation target is the proxy distribution (CinC), not real AW directly.
- Figure verified via PIL (1430×715); vision tool down this session.

## Follow-ups spawned
- **UDA deprioritized** for this problem — three methods, consistent negative.
- Redirects effort to (a) **E27** augmentation scaling (works, see its report)
  and (b) **E30** semi-supervised / few-shot: how many *labeled* real samples
  to close the gap — the honest data-collection-cost question.
