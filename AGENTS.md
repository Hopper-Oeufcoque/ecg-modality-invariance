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
- E1 validated the forward-physics simulator (lead-count = dominant axis).
- E2 found **lead-masking (K-MERL C9) is the decisive winner** (0.521→0.717 on
  full watch, matches/beats single-lead reference, zero target labels).
- E3/E3b: latent alignment beats naive but adds NO value over end-to-end
  lead-masking — only helps when encoder is frozen (FM+adapter path).
- E5: test-time BN adaptation marginal.
- Next queued: E4 (500 Hz), E6 (real single-lead), E7 (LeadBridge) — see
  `docs/FUTURE_APPROACHES.md`.

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

## Datasets
- PTB-XL 100 Hz subset at `~/data/ptbxl/` (downloaded, ~21k records).
  500 Hz subset (`filename_hr`) available for E4.
- CODE-15, MIMIC-IV-ECG (clinical 12-lead) — not yet downloaded.
- Apple Heart Study, PhysioNet single-lead sets — for E6 (real watch validation).
