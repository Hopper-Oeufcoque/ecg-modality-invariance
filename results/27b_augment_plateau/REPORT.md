# E27b — Augmentation plateau + morphology guard

**Question:** E27 showed augmentation scaling with the best at the grid corner
(strength 1.5, 3×). Push further — strength {1.5, 2.0, 2.5} × expansion {3,5,7},
5 seeds — to find the plateau, WITH a morphology-preservation guard (mean
augmented-vs-source R-peak correlation; flag < 0.85 as label-risk) so we don't
buy AUROC by corrupting the diagnostic content.

## Results (AUROC mean±std, 5 seeds; morphology corr by strength)

Morphology preservation drops with strength (label-validity proxy):
- strength 1.5 → corr **0.920** ✅ safe
- strength 2.0 → corr **0.875** ✅ safe
- strength 2.5 → corr **0.826** ⚠️ UNSAFE (below 0.85 guard)

| strength \ exp | 3× | 5× | 7× |
|---|---|---|---|
| 1.5 (safe) | 0.775 | **0.801** | 0.783 |
| 2.0 (safe) | 0.778 | 0.795 | 0.789 |
| 2.5 (⚠unsafe) | 0.828 | 0.814 | **0.830** |

## Verdict: ⚠️✅ Plateau found in the SAFE zone (~0.80); higher AUROC beyond it is label-risky

- **Within morphology-safe configs (strength ≤ 2.0), augmentation plateaus at
  ~0.80** (best safe = s1.5_×5 = 0.801, Δ+0.120 vs clean, 5/5 seeds). Expansion
  saturates at 5× (7× no better); strength 1.5≈2.0. So pure honest augmentation
  tops out around **0.80 — recovering ~48% of the clean→oracle (0.93) gap.**
- **strength 2.5 scores higher (0.828–0.830) but is a MIRAGE / partial cheat:**
  its morphology correlation (0.826) is below the 0.85 guard, meaning the
  perturbations are starting to distort QRS shape — some of that AUROC gain
  comes from label leakage / the model keying on artifacts, not genuine
  robustness. **We do NOT adopt strength 2.5.** This is exactly why the guard
  was built in.
- **Lock the packaged recipe at strength 1.5, expansion 5×** (0.801, label-safe,
  low variance). That's the honest ceiling of zero-label augmentation.

## Honesty flags
- 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c proxy not real AW.
- Morphology guard is a *correlation* proxy for label validity, not a
  cardiologist reading. strength 2.5's gain being partly spurious is inferred
  from the corr drop, not proven by clinical review — but the guard's whole
  point is to refuse that trade.
- Best-safe (s1.5_×5) beats E27's s1.5_×3 (0.775) — expansion 5× is the sweeter
  spot; consistent story.
- Figure verified via PIL (1690×715); vision tool down this session.

## Follow-ups
- **Recipe LOCKED: strength 1.5, 5× expansion.** Pure augmentation ceiling ≈0.80.
- To go beyond 0.80 needs *real labels* (E30) — augmentation alone can't, and
  going harder corrupts morphology. This bounds the zero-label approach honestly.
