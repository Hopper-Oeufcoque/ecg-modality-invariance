# E33b — Calibrated DR + few-shot: the best-of-both recipe (closest to oracle yet)

**Question:** E33 gave the best zero-shot base (calibrated DR, 0.81–0.82). E30
showed ~50 labels help. Does stacking them — calibrated-DR pretrain + fine-tune
on k real labels — reach oracle with fewer labels than hand-tuned pretrain did?

**Setup:** calibrated-DR pretrain (E33 recipe) → fine-tune on k∈{25,50,100}
balanced real CinC; vs hand-tuned pretrain + k=50 reference. 5 seeds, real CinC
AF/NORM. Oracle = 0.937.

## Results (AUROC mean±std, 5 seeds; % of remaining gap closed vs calibrated k=0)

| Arm | AUROC | % of gap to oracle |
|---|---|---|
| calibrated zero-shot (k=0) | 0.811 ± 0.020 | 0% (base) |
| calibrated + k=25 | 0.836 ± 0.023 | +20% |
| **calibrated + k=50** | **0.874 ± 0.040** | **+50%** |
| **calibrated + k=100** | **0.894 ± 0.022** | **+66%** |
| hand-tuned + k=50 (reference) | 0.836 ± 0.024 | +19% |
| oracle (real→real) | 0.937 ± 0.011 | 100% |

## Verdict: ✅ The two levers stack — and a better zero-shot base halves the label budget

- **Calibrated-DR + k=50 (0.874) beats hand-tuned + k=50 (0.836) by +0.038.**
  The better zero-shot base carries through fine-tuning: at the same label
  budget, calibrated DR closes **50%** of the remaining gap vs hand-tuned's 19%.
  A better start ≈ **doubles the value of the same 50 labels.**
- **calibrated + k=100 → 0.894**, within **0.043 of oracle** (0.937) — ~66% of
  the gap closed, using only 100 labeled real samples.
- **Clear monotone ladder:** 0.68 clean → 0.82 calibrated zero-shot → 0.87 (+50
  labels) → 0.89 (+100 labels) → 0.94 oracle.

## The complete gap-closing picture (final synthesis)

The 0.68→0.94 clinical→watch modality gap decomposes into a validated,
actionable ladder — the deliverable answer to "how do we transfer clinical ECG
models to Apple Watch":

| approach | labels needed | AUROC | note |
|---|---|---|---|
| clean clinical Lead-I | 0 | 0.68 | naive baseline |
| hand-tuned augmentation | 0 | 0.78–0.80 | E26/E27b |
| **target-calibrated DR** | **0** | **0.82** | **best zero-shot (E33)** |
| calibrated DR + 50 labels | 50 | 0.87 | E33b |
| calibrated DR + 100 labels | 100 | 0.89 | E33b |
| oracle (train on real) | ~700 | 0.94 | ceiling |

**What works:** cover the target distribution during training (calibrated DR) +
a small number of real labels. **What doesn't:** unsupervised test-time
adaptation of a fixed model (E29/E32, 4/4 methods hurt).

## Honesty flags
- 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c proxy not real AW.
- **AF/NORM specific** — E31 showed this ladder does NOT cleanly generalize to a
  harder rhythm task; these numbers are for a distinctive-rhythm task.
- k=50 has high variance (std 0.040 — one seed at 0.795 drags it); k=100 is more
  stable (0.022). The k=50 knee is real but noisy at N=5.
- Full zero-shot to oracle (0.94) is NOT achieved — best zero-shot is 0.82.
  Reaching ~0.90 needs ~100 real labels. Honest bottom line: **near-oracle
  performance requires a modest label budget; pure zero-shot tops out ~0.82.**
- Morphology validated in E33 (QRS corr 0.966, R-peak 0.976). Figure via PIL.

## Follow-ups spawned
- **Deployment recipe FINALIZED (for distinctive rhythms):** calibrated-DR
  pretrain (zero-shot 0.82) + fine-tune on 50–100 real labels (0.87–0.89).
  Package as Phase C `AWTrainingSetBuilder` + fine-tune helper.
- **E33c** — richer target coverage (more stat axes: higher moments, PSD shape,
  RR-interval stats) to push the zero-shot base above 0.82.
- **E34** — validate the ladder on a cleanly-mapping second rhythm (address E31).
