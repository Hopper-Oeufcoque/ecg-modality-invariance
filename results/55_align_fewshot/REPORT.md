# E55 — Does label-anchored alignment stack with few-shot real labels toward oracle?

**Date:** 2026-07-27
**Script:** `experiments/55_align_fewshot.py`
**Seeds:** 10 · **k (real CinC fine-tune labels):** {0, 10, 25, 50, 100} · AF task

## Hypothesis
The north-star deployment recipe = abundant clinical data + unlabeled real paired set
+ a handful of real target labels. E46 mapped this curve for clean vs calibration
(both plateaued ~0.85 << oracle 0.923). E51 alignment is the confirmed k=0 winner
(0.807 vs calibration 0.742). Question: as we add k real labels, does alignment stay
ahead / reach oracle faster, or do the arms converge (labels wash out the advantage,
as they did for calibration)?

## Results — labels-to-target curves (AUROC real CinC, 10 seeds)

| k | clean | calibration | **alignment (joint)** |
|---|---|---|---|
| 0 | 0.701 | 0.735 | **0.805** |
| 10 | 0.742 | 0.770 | 0.772 ⬇ |
| 25 | 0.807 | 0.824 | **0.836** |
| 50 | 0.843 | 0.849 | **0.863** |
| 100 | 0.845 | 0.853 | **0.866** |
| oracle | | | **0.930** |

**joint − calibration at each k:** k0 +0.069 (9/10, p=0.003) · k10 +0.002 (n.s.) ·
k25 +0.012 · k50 +0.014 (9/10, p=0.05) · k100 +0.013.
**Labels to reach 0.85:** clean never (≤100) · calibration 100 · **alignment 50.**
**Labels to reach 0.90:** none of the three within k≤100.

## Verdict ✅ (alignment is the best recipe at every budget — with two honest caveats)
Alignment dominates the entire curve in absolute AUROC: at every label budget,
joint ≥ calibration ≥ clean. It **halves the labels needed to reach 0.85** (50 vs
calibration's 100; clean never gets there). The biggest advantage is at **k=0**
(+0.069) — exactly the zero-label regime that matters most for deployment.

**Caveat 1 — the advantage shrinks as labels arrive** (+0.069 → ~+0.013). Alignment
and real labels are partial *substitutes*, the same pattern E46 found for calibration:
once enough real labels are present, they supply directly what alignment supplied
indirectly. Alignment keeps a small, persistent edge (~+0.013 at k=100) but not the
k=0 dominance.

**Caveat 2 — a real k=10 dip.** Joint at k=10 (0.772) drops *below* its own k=0
(0.805). Fine-tuning the well-aligned encoder on just 10 labels **hurts** — the tiny-k
fine-tune instability flagged in E30. Ten labels are enough to perturb a good
representation but too few to improve it (a mild catastrophic-forgetting effect).
Calibration and clean don't dip because their k=0 baselines are lower (less to lose).

## Practical deployment guidance (falls straight out of the curve)
- **< ~25 real labels available → use the aligned model ZERO-SHOT (k=0, 0.805).** Do
  not naively fine-tune; it can hurt (k=10 dip). This is a concrete, actionable rule.
- **~50 real labels → fine-tune the aligned model** → 0.863, the best achievable here,
  crossing 0.85 at half the label cost of calibration.
- **Approaching oracle (0.93) needs many more than 100 labels** regardless of method —
  the last ~0.06 is not reachable by any zero/few-shot recipe in this setup.

## Consequence for the north star
This completes the practical picture: the confirmed method gives the best transfer at
**every** label budget, and is most valuable exactly where labels are scarcest (k=0).
The realistic deployable operating point is **alignment + ~50 real labels → 0.863**,
or **alignment zero-shot → 0.805** when no labels exist. Near-oracle performance
remains gated on substantial real labeling — an honest ceiling, not closed by any
method tested.

## Honesty flags
- CinC dry-finger ≠ AW wrist; AF/N easy task.
- Tiny-k fine-tune is high-variance (E30) — the k=10 dip and wide k≤25 spreads reflect this.
- Joint base uses SJLIFE pairs (3-device caveat: align on SJLIFE, test on CinC).
- 10 seeds; single clinical train set; single architecture. oracle 0.930 is CinC
  train-on-real (the proxy oracle, not the 0.77–0.84 real-AW HOME benchmark).
