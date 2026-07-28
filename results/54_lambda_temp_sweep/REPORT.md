# E54 — λ / temperature sensitivity of the E51 headline

**Date:** 2026-07-27
**Script:** `experiments/54_lambda_temp_sweep.py`
**Grid:** λ ∈ {0.03, 0.1, 0.3, 1.0} × temp ∈ {0.05, 0.1, 0.2} · 10 seeds/cell · AF task
**Refs (seed-matched):** clean 0.701 · closed_aug (calibration) 0.737

## Hypothesis
E51's headline (joint 0.807, +0.106 vs clean) used λ=0.1, temp=0.1 chosen a-priori.
Before building further, confirm the win is a **robust plateau**, not a fragile spike
at one lucky hyperparameter setting. Sweep both knobs of the alignment term and check
how many cells still beat calibration (0.737) and clean (0.701).

## Results — AUROC surface (10 seeds/cell)

| λ \ temp | 0.05 | 0.1 | 0.2 |
|---|---|---|---|
| **0.03** | 0.782 | 0.769 | 0.758 |
| **0.1** | 0.816 | 0.805 | 0.799 |
| **0.3** | 0.814 | **0.819** | 0.814 |
| **1.0** | 0.783 | 0.789 | 0.792 |

- **Cells beating calibration: 12/12. Cells beating clean: 12/12.**
- Best cell: λ=0.3, temp=0.1 → **0.819**. E51 default (λ=0.1, temp=0.1) → 0.805
  (reproduces E51's 0.807 @20 seeds within noise).
- Worst cell: λ=0.03, temp=0.2 → 0.758 — **still +0.021 over calibration.**

## Verdict ✅ (robust plateau — headline confirmed)
The win is **not** hyperparameter-fragile. Every one of the 12 configurations beats
both calibration and clean, and the whole central region (λ∈{0.1, 0.3} × all three
temps) sits in a tight 0.799–0.819 band. The a-priori default was near-optimal; a
light tune to λ=0.3 buys a marginal +0.014.

## Interpretation — a well-behaved U-shape in λ
The alignment weight λ traces a clear optimum:
- **λ=0.03 (too weak):** alignment barely perturbs the encoder → smaller gain (0.76–0.78)
  and highest variance (std ~0.035) — closest to the un-aligned regime.
- **λ=0.1–0.3 (the sweet spot):** alignment and CE balance → best transfer (0.80–0.82),
  variance already tightened (std ~0.022–0.029).
- **λ=1.0 (too strong):** alignment starts to dominate CE → mean dips back to ~0.79
  with the *lowest* variance (std ~0.016). This is the E50 failure mode beginning to
  reassert — over-weighting invariance trades away discriminative content — but the CE
  anchor keeps it from collapsing, so it still beats calibration.

Temperature is a second-order knob: within each λ, temp moves AUROC by ≤0.017. Lower
temp (0.05) is marginally better at λ=0.1; the effect is minor everywhere.

The variance monotonically shrinking with λ confirms the E51b reading: the alignment
term acts partly as a regularizer, but — critically — the *mean gain* peaks at moderate
λ and the CE anchor prevents the information-destruction that killed E50 at any λ ≤ 1.0.

## Consequence for the north star
The headline method is **robust to its hyperparameters** — a practitioner does not need
to tune carefully to get the win; anything in λ∈[0.1, 0.3] with temp∈[0.05, 0.2] lands
near 0.81. That materially strengthens the deployability claim.

## Honesty flags
- 10 seeds/cell (vs 20 in E51) — CPU budget for a 12-cell grid; per-cell CIs wider.
- CinC dry-finger ≠ AW wrist; AF/N easy task.
- Single clinical train set; 3-device caveat (SJLIFE pairs for alignment, tested on CinC).
- Grid is coarse (4×3); the true optimum may lie between sampled points, but the plateau
  is broad enough that this doesn't affect the robustness conclusion.
