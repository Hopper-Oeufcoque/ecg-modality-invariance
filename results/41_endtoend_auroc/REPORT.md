# E41 — END-TO-END AUROC: clinical-train → real labeled single-lead transfer

**Date:** 2026-07-27
**Train:** PTB-XL Lead-I, AFIB vs NORM (real 12-lead clinical, ch0 only), n≈700/class.
**Test:** REAL held-out CinC 2017, AF vs Normal (AliveCor KardiaMobile dry-electrode
single-lead, 300→100 Hz). 5 seeds. Calibration uses ONLY unlabeled CinC-ref stats.

## The question
E40 proved the closed-loop calibrator *hits* the real modality profile
(coverage). E25b warned coverage ≠ utility. **E41 is the utility test:** does
calibrating clinical training data to the real single-lead distribution actually
lift downstream AUROC on real labeled data?

## Result

| arm | AUROC (5 seeds) | vs clean | seeds |
|---|---|---|---|
| clean (floor) | 0.681 ± 0.052 | — | 0.62/0.70/0.67/0.77/0.65 |
| light_DR | 0.712 ± 0.047 | +0.031 | 0.65/0.76/0.74/0.75/0.66 |
| **closed_loop** | **0.753 ± 0.060** | **+0.072 (5/5)** | 0.78/0.83/0.72/0.78/0.65 |
| clean+closed (cocktail) | 0.722 ± 0.064 | +0.041 (3/5) | 0.69/0.81/0.62/0.73/0.77 |
| oracle (train-on-real) | 0.930 ± 0.014 | — | 0.93/0.95/0.93/0.91/0.93 |

- **Closed-loop is the best zero-label arm: +0.072 over clean, wins 5/5 seeds.**
  Paired t=2.26, **p=0.086** — a consistent trend, NOT yet significant at 0.05.
- Closes ~29% of the clean→oracle gap (0.681→0.753 of 0.681→0.930) with **zero
  test labels** (calibration used only unlabeled CinC stats).
- light-DR helps a little (+0.031); the *calibration* is what adds most of the lift.
- Calibrated wander amp 0.549 → measured target bw 0.242 (matches E40).

## Honest flags (load-bearing)
- **p=0.086, not significant.** 5/5 directional wins + effect size 0.072 make it a
  strong trend, but this needs more seeds (10–20) before any hard claim. Logged
  as ⚠️ trend, not ✅ proven.
- **Cocktail REGRESSED (0.722 < 0.753 closed-loop-alone) — contradicts E26**,
  where clean∪augmented beat augmented-alone (that was on a WatchSim proxy target,
  not this real-distribution-calibrated augmenter). Here mixing in clean clinical
  *dilutes* the modality coverage. Lesson: the cocktail rule is target-dependent,
  not universal. Flagged for follow-up.
- **CinC KardiaMobile ≠ Apple Watch** — finger vs wrist dry-electrode. It is the
  closest LABELED real single-lead proxy; real AW (SJLIFE/HOME) has no labels.
- **AF vs NORM is the easy distinctive-rhythm task** (E31: harder tasks generalize
  worse). Mechanism demonstration, not a universal accuracy claim.
- Single-site clinical source; n capped 700/class.

## Verdict ⚠️✅ (trend, promising)
First real end-to-end evidence that **closed-loop modality calibration converts
abundant clinical data into better real-single-lead transfer** — coverage DID
translate to utility here (answering E25b's open question in the affirmative for
this task), but the statistical bar isn't cleared yet. This is the strongest
zero-label result the project has on REAL labeled data.

## Follow-ups
1. **E42: bump to 15–20 seeds** to settle p<0.05 (or refute).
2. Investigate the cocktail regression (mixing ratio sweep clean:closed).
3. Multi-axis closed-loop (E40 residual) → does closing qrs/mid energy lift further?

## Artifacts
`metrics.json`, `endtoend_auroc.png` (5-arm bar, clean floor + oracle ceiling lines).
