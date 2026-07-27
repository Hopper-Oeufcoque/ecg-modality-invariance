# E32 — TENT test-time entropy minimization: ❌ HURTS (zero-shot lever fails)

**Question:** the augment→oracle gap (0.80→0.93) is the zero-shot frontier. TENT
(Wang et al. ICLR 2021) is the strongest zero-label test-time method — adapt only
BN affine params (or all params) by minimizing prediction entropy on the
unlabeled target batch. Does it close the gap where AdaBN (E29) failed?

**Setup:** augment-pretrained model (s1.5, 5× = 0.783), then TENT-adapt on the
unlabeled real CinC test batch (transductive). affine mode (BN γ/β, lr 1e-3, 10
steps) and full mode (all params, lr 1e-4). 5 seeds.

## Results (AUROC mean±std, 5 seeds; Δ vs augment)

| Arm | AUROC | Δ vs augment | seeds + |
|---|---|---|---|
| clean | 0.681 ± 0.052 | — | — |
| **augment (bar)** | **0.783 ± 0.021** | — | — |
| augment + TENT (affine) | 0.737 ± 0.053 | −0.046 | 1/5 |
| augment + TENT (full) | 0.712 ± 0.056 | −0.071 | 0/5 |
| oracle | 0.914 ± 0.025 | — | — |

## Verdict: ❌ TENT HURTS — entropy minimization is the wrong objective here

- Both TENT variants **reduce** AUROC (affine −0.046, full −0.071) and **inflate
  variance** (std 0.021 → 0.053/0.056). More adaptation = more damage.
- **Why it fails:** TENT minimizes prediction entropy, i.e. it makes the model
  *more confident* on the target batch. That only helps if the decision boundary
  is roughly right and just needs sharpening. Here the augment-trained model's
  boundary is genuinely uncertain on real single-lead AF — so entropy
  minimization sharpens confidence around a **partly wrong** boundary,
  reinforcing errors (classic TENT failure under large/genuine shift with a
  small, imbalanced test batch).
- **Consistent with E29:** every unsupervised test-time / feature adaptation we
  have tried (AdaBN, CORAL, DANN, now TENT) HURTS. Four independent methods,
  same direction.

## The hardening conclusion

**Zero-label adaptation of a fixed clinical-trained model does not close the
modality gap — it consistently makes things worse.** The gap is not a
correctable "the features are right but miscalibrated" problem; the real
single-lead decision boundary carries information the clinical+augmented model
simply never learned. That information comes only from (a) covering the target
distribution *during training* (E33 tests this) or (b) real labels (E30).

TENT/entropy-style test-time adaptation is now **ruled out** for this problem.

## Honesty flags
- 5 seeds, n=700 CinC (350 test/seed), AF/NORM only, CinC = E6c proxy.
- TENT is transductive (adapts on the test batch) — the intended deployment
  mode, and still it hurts, so this isn't a leakage artifact in TENT's favor.
- TENT hyperparameters (lr, steps) are standard but not exhaustively swept; a
  heavily-tuned TENT *might* reach neutral, but the strong consistent negative
  across 4 adaptation methods makes "test-time adaptation is the answer"
  implausible. Logged as a clear negative.
- Note oracle here = 0.914 (this seed set), slightly below the 0.93 seen
  elsewhere — seed variance; the augment→oracle gap is still ~0.13.
- Figure verified via PIL (1300×715); vision tool down this session.

## Follow-ups spawned
- **Test-time adaptation (TENT/AdaBN/CORAL/DANN) all deprioritized** — 4/4 hurt.
- All remaining hope for *zero-shot* rests on **train-time target coverage**
  (E33 calibrated domain randomization). If that also plateaus, the honest
  answer is that this gap needs a modest number of real labels (E30).
