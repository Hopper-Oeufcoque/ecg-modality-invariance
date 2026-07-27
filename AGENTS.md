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

## Current experimental status (as of 2026-07-27)
- **HEADLINE: all sim-validated results carry realism debt — they don't transfer
  to real single-lead data.** See `docs/SESSION_HANDOFF.md` for the full reframe.
- E1: lead-count = dominant axis. E2: lead-masking = sim-best 0.718 (BUT fails
  on real — E23).
- E6: simulator OVER-DEGRADES vs real (sim_vs_real 1.077 > real_vs_clinical 0.717;
  kurtosis 4.8 vs 17.7). E22: over-degradation is FILTER-bound (bandpass).
- E6b (decisive): sim training HURTS real transfer (sim→real 0.737 < clean→real
  0.753); 0.993 sim→sim is overfit artifacts. Oracle (real→real) = 0.946.
- E23 (decisive): lead-masking 12-lead CATASTROPHICALLY fails on real (0.557 vs
  sim 0.718); single-lead (clean 0.721, sim 0.731) robust. **Revised real
  ranking: single-lead models > 12-lead lead-masking. Train single-lead.**
- Novel methods (E10 INLP, E15 MixStyle, E9 REx, E20 DeepSet, E8 speech, E18
  scattering): all near-neutral or regime-dependent. Gap is info loss, not a
  removable/avoidable shortcut.
- **Real Apple Watch data: HOME benchmark at `~/data/HOME` (1000 AW ECGs,
  evaluation-only). E6c (sim vs REAL AW) + E5b (BN-adapt salvage) are the
  immediate next experiments.** See `docs/SESSION_HANDOFF.md`.

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
  500 Hz subset (`filename_hr`) available for E4.
- CODE-15, MIMIC-IV-ECG (clinical 12-lead) — not yet downloaded.
- Apple Heart Study, PhysioNet single-lead sets — for E6 (real watch validation).
