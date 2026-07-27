# Session Handoff — ECG Modality Invariance Project

> **Read this FIRST in a new session.** This file captures the exact state at
> the last handoff so a fresh session picks up immediately. Living docs
> (EXPERIMENT_LOG.md, FUTURE_APPROACHES.md) have the full history.
>
> **Project dir:** `~/projects/ecg-modality-invariance/` (auto-loads AGENTS.md)
> **Repo:** github.com/Hopper-Oeufcoque/ecg-modality-invariance (public)
> **Venv:** `.venv` (torch CPU, wfdb, sklearn, scipy) — `source .venv/bin/activate`
> **User preference:** autonomous driving, no permission checkpoints, Opus-5 only
> (0 OpenRouter — config locked Jul 27).

## The big picture (as of 2026-07-27, end of session)

**Phase 2 = experiments. The headline finding is a major honesty flag:**
**all sim-validated results carry realism debt — they do not transfer to real
single-lead data.** The chain of evidence:

1. **E6** — the forward-physics simulator OVER-DEGRADES relative to real single-
   lead (CinC 2017 handheld): sim_vs_real distance (1.077) > real_vs_clinical
   (0.717). Sim's noise flattens QRS peakedness (kurtosis 4.8 vs real 17.7).
2. **E22** — the over-degradation is FILTER-bound (bandpass), not noise-bound;
   recalibration helps marginally (+0.012) but kurtosis stays stuck ~5.
3. **E6b** (decisive) — sim training HURTS real transfer: sim→real CinC = 0.737,
   but clean Lead-I→real = 0.753. The 0.993 sim→sim AUROC is a red flag (model
   overfit sim artifacts). Oracle (real→real) = 0.946.
4. **E23** (decisive) — lead-masking (12-lead prior, "winner" on sim at 0.718)
   CATASTROPHICALLY fails on real CinC: 0.557. Single-lead models (clean 0.721,
   sim 0.731) far more robust. The 12-lead prior doesn't survive real shift.

**Revised real-deployment ranking (INVERTED from sim):**
- single-lead models (clean Lead-I or sim-augmented): ~0.72-0.73 on real ← best label-free
- 12-lead lead-masking: 0.557 on real ← catastrophic (was sim-best!)
- oracle (real watch data): 0.946 ← the ceiling, unreachable by simulation

**Robust conclusion: train a SINGLE-LEAD model for real deployment.** The 12-lead
prior and aggressive simulator both fail on real data. Real watch data is the
binding constraint (the 0.946 gap is unreachable by sim).

## Novel methods tried (this session) — all near-neutral or regime-dependent
- **E10 INLP** (NLP fairness → ECG, novel): modality linearly separable (0.999)
  but scrubbing it NEUTRAL (gap = info loss, not a removable shortcut). ❌
- **E15 MixStyle** (image DG → ECG, novel): regime-dependent — HURTS lead-masking,
  HELPS single-lead+sim (new sim-best 0.746, within noise). ⚠️✅
- **E9 REx** (report Solution-2 keystone, novel): near-neutral on sim (+0.007,
  within noise); environments too similar. ⚠️
- **E20 DeepSet** (point-cloud → ECG, novel): 0.708, competitive but < leaders. ⚠️
- **E8 speech features** (novel cross-domain): ❌ destroys ECG morphology.
- **E18 scattering** (novel): 0.661 training-free, complement not replacement. ⚠️

## New real Apple Watch data (user-provided 2026-07-27)
- **HOME benchmark** cloned to `~/data/HOME` — 1000 REAL Apple Watch patient ECGs
  (200 Hz, 30s, Lead I) in `data-for-predicting/Apple_Watch_waveform.csv`.
  Evaluation-only (no training/fine-tuning/domain-adapt — license-compliant for
  distribution analysis + inference-only, NO labels). Plus 20 labeled training
  examples (data/ecg + train.csv, LVD task) and baseline-model predictions.
- PhysioNet ecg-capable-smartwatches — transfer-function characterization (METRON
  PS-440 simulator, not patient ECG; restricted access). Noted, not downloaded.

## IMMEDIATE NEXT ACTIONS (do these first)
1. **Re-run E6c** (was mid-run when session ended — background process killed by
   restart). It's the DEFINITIVE realism test: simulator vs REAL Apple Watch.
   ```
   cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
   mkdir -p results/06c_real_applewatch
   python3 experiments/06c_real_applewatch.py > results/06c_real_applewatch/run.log 2>&1
   ```
   - If sim over-degrades vs real AW too → E6 holds on true target; sim fundamentally
     miscalibrated for wrist dry-electrode.
   - If sim matches AW better than CinC → aggression was justified for wrist; CinC was
     the wrong reference; sim more defensible.
   - **Write E6c REPORT.md + EXPERIMENT_LOG entry when done.**

2. **Run E5b** (scripted, ready) — test-time BN adaptation on real CinC. Tests if
   E23's 0.557 catastrophe is BN-driven (recoverable) or fundamental.
   `python3 experiments/05b_bn_adapt_real.py`

3. **Inference-only test on real Apple Watch (HOME)** — run the single-lead models
   (E17 sim-trained, V2 clean) on the 1000 real AW waveforms. No labels → compute
   prediction distribution / compare to HOME baseline-model predictions (correlation
   = captures similar signal). This is the closest we get to real-AW task eval
   without the submission process.

4. Continue exploring novel architectures per FUTURE_APPROACHES.md (E14 group-
   equivariant, B1 signal synthesis — the "recover missing lead info" lever E10
   pointed to).

## Running convention reminder
- Every experiment → `results/<id>/REPORT.md` + `metrics.json` + figures (visually
  verify before reporting) + EXPERIMENT_LOG.md entry (verdict table + full entry)
  + FUTURE_APPROACHES.md update. Commit `experiment <id>: <one-line>`, push.
- Negative results logged with EQUAL weight. Honesty flags (small-N, single seed,
  no CIs, sim-vs-real) stated in every report.
- Sign off 🐇 (rabbit), call user "Hop".

## Key files to reorient
- `docs/EXPERIMENT_LOG.md` — the living lab notebook (full verdict table + entries)
- `docs/FUTURE_APPROACHES.md` — actionable backlog (E6b done, E10 done, E15 done,
  E9 done, E22 done; E5b/E6c/E22b/B1 open)
- `results/EXPERIMENT_SYNTHESIS.md` — consolidated method ladder (needs refresh
  with E6b/E23 real-deployment reframe)
- `src/watch_simulator.py` — the (now-refuted-for-training) forward-physics sim
- `experiments/` — all numbered scripts
