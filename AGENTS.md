# ECG Modality Invariance — Agent Working Notes

Context auto-loaded for any Hermes session working in this directory.

## Project
Research + experimentation on transferring clinical ECG AI models to Apple
Watch single-lead ECGs, focused on **recording-modality invariance**. Owner:
Luke (US-based). Repo: github.com/Hopper-Oeufcoque/ecg-modality-invariance
(**public** since 2026-07-27).

## Deliverables (living)
1. **Method taxonomy** — `docs/method_taxonomy.md`, 9 categories (A–I), ~50
   methods tagged `[P]`/`[A]`/`[N]`.
2. **Evidence** — what has been tried in the literature, with citations
   (`references/`, 155-paper corpus).
3. **Frontier** — `notes/idea_log.md` (13 frontier ideas) +
   `docs/FUTURE_APPROACHES.md` (actionable experimental backlog).
4. **Experiments** — `results/` holds per-experiment REPORT.md + metrics.json +
   figures. **`docs/EXPERIMENT_LOG.md` is the living lab notebook** (what was
   tried, what worked, what didn't — negative results logged with equal
   weight). `results/EXPERIMENT_SYNTHESIS.md` ties them into a method ladder.
5. **Code** — `src/` (watch_simulator, dataset, model) +
   `experiments/` (numbered scripts). Python venv at `.venv` (torch CPU,
   wfdb, sklearn, scipy).

## Current experimental status (as of 2026-07-27, real-data phase E38–E56)
- **HEADLINE (confirmed): label-anchored real-paired modality alignment.** Train the
  single-lead encoder to classify clinical Lead-I AND align same-patient
  clinical↔Apple-Watch pairs (real SJLIFE) in the same step — the CE loss anchors
  invariance so it can't destroy pathology. **0.807 alone / 0.820 +calibration
  (+0.078 vs calibration, +0.119 vs clean, 20/20 seeds, p<1e-8)** on real CinC AF.
  Falsification-tested (E51b: shuffled pairs → null p=0.76; only same-patient
  correspondence works, +0.101 p=3e-10) → genuine cross-modality invariance, not
  regularization. Robust (E54: 12/12 λ×temp cells win), general (E53: +0.034 on
  morphology where calibration is null), label-efficient (E55: ½ the labels-to-0.85),
  device-safe (E56: no OOD collapse, halves variance).
- **Lever 2: closed-loop calibration** — +0.041 zero-label (E42), gap-proportional
  (E44), rhythm-specific (E47), worth ~10-15 labels (E46). Lighter; no paired data.
- **Both obey a dose-response law: benefit ∝ modality gap.** Real AW wrist is large-gap
  (SJLIFE bw~0.20) → predicts large lift there (a PREDICTION — no labeled real-AW data).
- **The E48–E50 negatives** (learned-invariance null, clinical-distillation −0.060,
  unanchored-contrastive −0.073) identified the failure mode — unanchored invariance
  destroys pathology — whose fix (the CE anchor) is the headline.
- **Canonical writeup: `docs/FINDINGS.md`.** Method ladder: `results/EXPERIMENT_SYNTHESIS.md`.
  Full notebook: `docs/EXPERIMENT_LOG.md`. Superseded simulator-phase reframe in
  `docs/SESSION_HANDOFF.md`.
- **Open (gated on external resources): HOME eval-portal frozen-model submission** — the
  only path to convert the AW prediction into a measurement (application in progress).

## Domains of shift (12-lead clinical → 1-lead watch)
lead count, electrode physics (wet vs dry), noise, sampling/bandwidth,
population/context.

## Working conventions
- Primary sources only for load-bearing claims (PubMed, arXiv, IEEE). Cite with
  DOI/arXiv ID.
- **Every experiment → append to `docs/EXPERIMENT_LOG.md`** (hypothesis/setup/
  result/verdict ✅⚠️❌/lesson/follow-up) + a `results/<id>/REPORT.md`. Update
  the Quick verdict table at the top. Commit with `experiment <id>: <one-line>`.
- Spawned ideas go into `docs/FUTURE_APPROACHES.md`; mark ✅/❌ + link the
  experiment ID once run.
- Phased work with checkpoint reporting — report at each phase boundary, ask
  before deviating from spec. (User has waived permission checkpoints for this
  project — proceed autonomously through experiment + iteration cycles.)
- No GPU on this host; CPU-only. Keep models modest (~0.5M params, 20 ep on
  ~1200 records is feasible in minutes on 4 cores).

## Pitfalls (recurring — read before launching runs)
- **mkdir race:** when launching `python3 exp.py > results/<id>/run.log 2>&1`,
  the shell opens the redirect BEFORE the script creates its output dir → dies
  instantly with `No such file or directory` (exit 1). ALWAYS prepend
  `mkdir -p results/<id> &&` in the launch command. Bit us twice on 2026-07-27.
- Launch long chains as separate background jobs; avoid `(a) & (b) & wait`
  subshells (dropped a job once).
- **Sampling-rate provenance:** HOME ships Apple Watch data at TWO rates —
  `data/ecg/*.csv` (15000 samp) = 500 Hz; `data-for-predicting/Apple_Watch_waveform.csv`
  (6000 samp, 1000 patients) = 200 Hz. ALWAYS verify fs from metadata AND a
  heart-rate sanity check (R-peaks → 50-100 bpm) before spectral analysis. A
  mislabeled fs silently invalidated E35/E36 (retracted by E37).

## Datasets
- PTB-XL 100 Hz subset at `~/data/ptbxl/` (downloaded, ~21k records).
  500 Hz subset (`filename_hr`) available for E4. AFIB n=1514, NORM n=16748.
- **CinC 2017** at `~/data/cinc2017/training2017/` — REAL labeled single-lead
  (AliveCor KardiaMobile, DRY FINGER electrode, 300 Hz). 738 AF / 5050 N / 2456 O
  / 284 noisy. Large baseline-wander gap (bw~0.25) → the best LABELED Apple-Watch
  proxy we have. Primary real-transfer test set (E41/E42/E43).
- **SJLIFE paired** at `~/projects/ecg-modality-invariance/data/sjlife/` (243
  patients, PAIRED clinical 12-lead 500 Hz + Apple Watch 512 Hz, same person).
  Public/trainable. NO disease labels (age/sex/HR/time-gap only). Real AW = DRY
  WRIST, bw~0.20 (~6× clinical). ~8.25× amplitude gain vs clinical → per-record
  z-score mandatory. Used for E38 (measure real modality profile).
- **Icentia11k** at `~/projects/ecg-modality-invariance/data/icentia/` (mined
  subset: 200 AF + 200 Normal windows, 21 patients). CardioSTAT CHEST-PATCH
  single-lead, 250 Hz, labeled beats+rhythms. Electrically ≈ clinical-clean
  (bw~0.016) → LOW modality gap → 2nd-device external-validity test (E44).
  CAVEAT: few patients × many windows = within-patient leakage; treat absolutes
  as optimistic (oracle hit 1.000). Full set = 11k patients if more needed.
- **HOME benchmark** at `~/data/HOME/` — 1000 AW ECGs, EVALUATION-ONLY (license
  forbids training/adaptation; labels withheld; centralized scoring only). Two
  files at DIFFERENT rates (see sampling-rate pitfall above).
- CODE-15, MIMIC-IV-ECG (clinical 12-lead) — not yet downloaded.

## Key result (as of 2026-07-27, E38–E56)
**Label-anchored real-paired modality alignment** is the confirmed headline
(`experiments/51_label_anchored_align.py::train_joint`): joint `CE(clinical Lead-I) +
λ·InfoNCE(same-patient clinical↔Apple-Watch SJLIFE pairs)`, λ∈[0.1,0.3]. **0.807 alone
/ 0.820 with calibration** on real CinC AF (+0.078 vs calibration, +0.119 vs clean,
20/20 seeds, p<1e-8) — ~52% of the clean→oracle(0.93) gap, zero wearable labels.
Falsification control (E51b) confirms genuine same-patient cross-modality invariance,
not regularization (shuffled pairs → null, p=0.76). General beyond rhythm (E53), robust
(E54), device-safe (E56), halves transfer variance.

**Closed-loop modality calibration** is the lighter, no-paired-data lever
(`src/aw_generator.py::ClosedLoopCalibrator`): measure unlabeled target baseline-wander
→ binary-search a coloured-wander augmentation to hit it, QRS band untouched → **+0.041
(p=0.009, 20 seeds)** on CinC. Both levers are GAP-PROPORTIONAL (E44/E56: help ∝ the
modality gap; idle on clean chest-patch). Real AW wrist is large-gap → predicts a large
lift, but real AW has NO public labels → the AW number is a PREDICTION, not proven
(HOME eval-portal submission is the pending confirmation path). Canonical: `docs/FINDINGS.md`.
