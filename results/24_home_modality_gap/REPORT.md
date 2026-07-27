# E24 — Modality-gap meta-analysis from HOME's real Apple Watch baselines

**Date:** 2026-07-27 · **Status:** ✅ Real-device, task-level modality gap — the first REAL Apple Watch evidence in the project

## Hypothesis
HOME publishes, for each of 9 clinical tasks, predictions from two models on the
same 1000 REAL Apple Watch ECGs: a "12-lead model" (trained on Lead-I from resting
12-lead ECGs — exactly the clinical→watch transfer this project studies) and a
"fine-tuning model" (trained on real Apple Watch Lead-I). The *agreement* between
them on real watch data measures the modality gap per task on the TRUE target
device: high agreement → clinical model already transfers well (fine-tuning barely
changes it); low agreement → large gap, clinical-only transfer insufficient.

## Setup
License-compliant meta-analysis: **no training, fine-tuning, or domain-adaptation
on HOME data** — only analysis of HOME's own published baseline predictions.
Per task, 1000 real Apple Watch subjects, compute Pearson r, Spearman rho, mean
abs difference (normalized), and (binary tasks) decision agreement @0.5.
`experiments/24_home_modality_gap.py`.

## Results (12-lead vs fine-tuning agreement on REAL Apple Watch)

| task | Pearson r | Spearman rho | dec. agree | verdict |
|---|---|---|---|---|
| Low Hb | 0.857 | **0.856** | 0.820 | TRANSFERS WELL |
| Death | 0.699 | 0.703 | 0.777 | MODERATE |
| Gender | 0.579 | 0.691 | 0.711 | MODERATE |
| Low eGFR | 0.414 | 0.658 | 0.446 | MODERATE |
| Age | 0.623 | 0.611 | — | MODERATE |
| High PASP | 0.592 | 0.607 | 0.824 | MODERATE |
| High NT-proBNP | 0.526 | 0.496 | 0.730 | LARGE GAP |
| High LA | 0.512 | 0.429 | 0.882 | LARGE GAP |
| Low EF | 0.526 | 0.421 | 0.900 | LARGE GAP |

Mean Spearman rho = 0.608, median = 0.611.

## Verdict: ✅ The modality gap is real, task-dependent, and moderate-to-large on real Apple Watch

1. **No task fully survives the shift.** Even the best (Low Hb, rho=0.856) shows
   meaningful fine-tuning-induced change; most tasks are in the 0.42-0.70 range.
   The mean rho of 0.61 means fine-tuning on real Apple Watch substantially
   reordered predictions for the average task — the clinical→watch modality gap
   is genuine and significant on the true target device, corroborating the whole
   project's premise with REAL data (not sim).

2. **Task-dependence is strong and interpretable.** The tasks with the LARGEST
   gaps are the spatial/structural cardiac ones — **Low EF (0.42), High LA (0.43),
   NT-proBNP (0.50)** — heart-failure/chamber markers whose ECG signatures live in
   spatial lead relationships (QRS morphology, axis, voltage) that a single watch
   lead captures poorly. This mirrors E1's sim finding (spatial classes MI/STTC/HYP
   collapse under lead reduction) — now confirmed on REAL Apple Watch. The
   smallest gap (Low Hb, 0.86) is a diffuse metabolic marker less dependent on
   spatial ECG structure.

3. **Decision agreement can mask ranking disagreement.** Low EF has 0.900 decision
   agreement @0.5 but only 0.421 rank agreement — because it's a rare-positive task
   (both models predict "negative" for most), decisions agree while the *ranking*
   (what AUROC measures) diverges badly. This is exactly why rho, not accuracy, is
   the right lens — and a caution for anyone reading the decision-agreement column.

## The critical honesty caveat
**This measures AGREEMENT, not ACCURACY.** HOME withholds labels, so I cannot
compute either model's true performance. High 12-lead/fine-tuned agreement means
fine-tuning changed little — NOT that either model is correct. A task where both
models are equally wrong would show high agreement but poor real utility. So:
- LARGE-gap tasks (Low EF, High LA, NT-proBNP) are robustly flagged: fine-tuning
  materially changed predictions → clinical-only transfer is insufficient there.
- TRANSFERS-WELL tasks (Low Hb) are only weakly supported: agreement is necessary
  but not sufficient for good transfer. Needs the labels (submission process) to confirm.

## How this connects to the project
- **Corroborates the core premise on REAL data:** the clinical→watch modality gap
  is real and task-dependent on actual Apple Watch — not a sim artifact. The sim
  experiments (E1) predicted spatial tasks suffer most; E24 confirms it on real AW
  (Low EF / High LA / NT-proBNP = largest gaps).
- **Prioritizes the deployment target:** modality-invariance methods matter MOST
  for spatial/structural tasks (Low EF, chamber markers), least for diffuse markers
  (Low Hb). Future method work should be evaluated on the high-gap tasks where it
  matters, not gender/age where transfer is easier.
- **Validates fine-tuning as the reference:** HOME's fine-tuning model (trained on
  real device data) is the practical answer — echoing E6b's oracle (0.946) finding
  that real target data is the binding constraint. Where fine-tuning diverges most
  from 12-lead is where target data is most necessary.

## Honesty flags
- Agreement, not accuracy (labels withheld) — see the critical caveat above.
- HOME's models are the authors' architecture/training, not ours — this measures
  THEIR modality gap, a strong external reference but not our pipeline.
- Single set of published predictions; no seed/CI (they're point predictions).
- HOME cohort is health-records-linked clinical patients — may differ from general
  Apple Watch users (selection toward disease).

## Follow-ups
- If an evaluation account is obtained: submit our single-lead models' predictions
  on the high-gap tasks (Low EF, High LA) to get true AUROC vs HOME's baselines.
- E6c (running): distribution-level sim-vs-real-AW — pairs with this task-level view.
- Focus future modality-invariance experiments on the high-gap spatial tasks.

## Artifacts
- `results/24_home_modality_gap/metrics.json`
- `results/24_home_modality_gap/modality_gap_per_task.png`
