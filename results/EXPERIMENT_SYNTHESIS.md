# Clinical → Wearable Single-Lead ECG Transfer: Synthesis of Findings

**Repo:** github.com/Hopper-Oeufcoque/ecg-modality-invariance · **Author:** Hopper
**Status (2026-07-27):** real-data results (E38–E47). This supersedes the earlier
simulator-phase synthesis (E1–E5), most of which was later **refuted on real
data** — see "What we retracted" below.

---

## The goal
Leverage abundant **clinical** ECG data to train models usable on **Apple Watch**
(and other wearable single-lead) ECG for downstream detection — ideally with
**zero or minimal** target labels, approaching a train-on-target oracle.

## TL;DR — what actually works, and where it stops
**Closed-loop modality calibration** (measure the unlabeled target's baseline-wander
level, then binary-search a morphology-preserving coloured-wander augmentation on
clinical data to hit it) gives a **real, significant, zero-label** transfer gain
— but **only for rhythm-detectable tasks**, and only on devices that actually
have a modality gap.

| dimension | finding | evidence |
|---|---|---|
| **Zero-label lift (AF)** | +0.041 AUROC (0.701→0.742), p=0.009, 20 seeds | E42 |
| **Label-equivalent value** | worth ~10–15 real labeled examples | E46 |
| **Best minimal-tuning recipe** | calibrate-on-unlabeled + ~50 real labels → 0.855 | E46 |
| **Gap-proportional** | helps on high-wander dry-electrode (CinC); idles on clean chest-patch (Icentia) | E42 vs E44 |
| **Rhythm-specific** | vanishes on morphological task (−0.002) | E47 |
| **Augmentation ceiling** | ~+0.041; residual gap is in-band (QRS/mid). Confirmed **information-bound**: learned invariance hits the same wall (E48) | E43, E48 |
| **Oracle (train-on-real, AF)** | 0.93 | E42 |

**One-line honest claim:** *For rhythm-detectable wearable tasks (AF confirmed),
closed-loop modality calibration converts clinical data into a zero-label
transfer gain worth ~10–15 labels; it does not extend to morphological diagnosis,
where single-lead transfer is intrinsically weak.*

---

## How we got here (the real-data arc)

### The mechanism
1. **Real single-lead is CLEAN and low-pass** (E37 spectral, E38 paired-hardware,
   E43 downstream) — the retracted "high-frequency bandwidth gap" was a
   sampling-rate artifact. The real clinical→wearable gap is **baseline wander**
   (dry electrode + motion), not HF noise.
2. **Fidelity ≠ utility** (E25b): a faithful spectral generator was neutral as
   training data; crude diversity-injecting augmentation helped. But…
3. **…open-loop calibration overshoots** (E39): a `sqrt(gap)` heuristic injected
   2× too much wander and collapsed kurtosis → worse than clean.
4. **Closed-loop fixes it** (E40): binary-search the injection until the *measured*
   output matches the target; use 1/f coloured wander (not sinusoids) to preserve
   peak structure. Coverage distance 1.06→0.66, morphology intact (QRS-corr 0.99).
5. **Coverage → utility, confirmed** (E41→E42): the calibrated model lifts real
   labeled-CinC AUROC +0.041 (p=0.009), zero test labels. First properly-powered
   real-data win. (E41's 5-seed +0.072 was small-sample optimism — corrected.)

### The boundaries (equally important)
- **Gap-proportional** (E44): on Icentia chest-patch (electrically ≈ clinical, no
  wander gap) calibration correctly idles (−0.003). The benefit scales with the
  gap. Apple Watch is a dry *wrist* electrode with a LARGE wander gap (SJLIFE E38:
  ~6× clinical) → dose-response *predicts* calibration should help on AW (unproven
  — real AW has no public labels).
- **HF axis is a no-op** (E43): adding a second calibration axis self-zeroed
  because real single-lead has ~no HF. Confirms the clean-signal physics end-to-end.
- **Rhythm-specific** (E47): on a morphological task (Normal-vs-Other) the lift
  vanishes (−0.002), because morphology lives in the in-band QRS/mid frequencies
  calibration deliberately leaves untouched. Also, clinical→real morphological
  transfer is intrinsically weak (oracle only 0.753).
- **Labels dominate eventually** (E46): calibration and labels are substitutes;
  above ~50 labels the calibration edge shrinks into the noise.

---

## The reusable tool
`src/aw_generator.py`:
- `ClosedLoopCalibrator.fit(target_bw, clinical_probe_sigs)` — the winner. Measures
  target from **unlabeled** data, morphology-preserving, gap-proportional.
- `MultiAxisClosedLoopCalibrator` — bw+hf; self-zeroes unused axes (kept for
  targets that DO have an HF gap; no-op on clean single-lead).
- `signal_modality_stats`, `measure_distribution`, `qrs_morphology_preserved`
  (QRS-band corr + R-peak match — the label-validity guard).

**Recipe:** z-score per record (SJLIFE showed ~8× clinical/AW gain difference) →
measure target bw from unlabeled wearable data → `ClosedLoopCalibrator.fit` →
train clinical Lead-I with the calibrated augmentation → (optional) fine-tune on
~50 real labels.

---

## Datasets used
- **PTB-XL** (clinical 12-lead, Lead-I) — training source. AFIB n=1514.
- **CinC 2017** (AliveCor KardiaMobile, dry FINGER, labeled AF/N/O) — primary real
  test; big wander gap; 1 record/patient (leakage-free). E41/E42/E43/E46/E47.
- **SJLIFE paired** (243 pts, clinical + Apple Watch, same person) — real modality
  profile measurement (E38). No disease labels.
- **Icentia11k** (CardioSTAT chest-patch, labeled) — 2nd-device dose-response
  (E44). Low gap. AF too sparse for a patient-disjoint re-mine (E45 abandoned).
- **HOME** (real Apple Watch, 1000) — EVALUATION-ONLY (license forbids training).

---

## What we retracted / corrected (honesty ledger)
- **E35/E36 "bandwidth gap"** → RETRACTED (E37): 200 Hz file processed as 500 Hz.
- **Early simulator synthesis (E1–E5)** → superseded: lead-masking/sim-training
  looked good on simulated watch but **failed on real single-lead** (E6b, E23).
  The simulator over-degrades; sim-trained models don't transfer.
- **E41 +0.072 (5 seeds)** → corrected to +0.041 (20 seeds, E42): small-sample
  optimism.
- **E44 Icentia absolutes** → untrusted (patient leakage, oracle=1.000); only the
  qualitative dose-response (which rests on the leakage-free CinC gap) stands.
- **Cocktail rule (E26)** → target-dependent, not universal (E41 cocktail regressed).

## Standing limitations
- Best real proxy is CinC finger-electrode, **not** Apple Watch wrist. Real AW
  (SJLIFE/HOME) has **no usable labels** → the AW lift is a dose-response
  *prediction*, not a measurement.
- All positive transfer results are **AF/rhythm**; morphology does not transfer.
- CPU-only, modest ResNet, single fixed clinical train set per experiment (seed
  CIs omit clinical-cohort variance).

## Open levers (next)
1. ~~**Representation-level rhythm-invariance**~~ — **ANSWERED (E48, ❌).** An
   explicit feature-consistency invariance loss does *not* beat implicit
   augmentation (−0.010, p=0.47); both families hit the same +0.041 wall. The
   ceiling is **information-bound**, not a formulation artifact. Do not chase
   cleverer single-lead invariance objectives.
2. **INFORMATION INJECTION (the pivot)** — the residual gap needs *new
   information*, not a better loss:
   (a) **multi-lead → single-lead distillation** — teach a single-lead student to
   mimic a 12-lead teacher's logits/features, so clinical multi-lead structure is
   compressed into the Lead-I student (available now: PTB-XL 12-lead + Lead-I);
   (b) **auxiliary channels** watches actually carry (accelerometer/PPG) as extra
   modality-invariant inputs;
   (c) **real target labels** — the proven closer (E46: ~50 → 0.855).
3. **Other arrhythmias** (PVC/PAC, flutter) — does the rhythm-transfer claim hold
   beyond AF? (Icentia has beat labels if we mine more patients)
4. **A same-taxonomy labeled single-lead morphological set** — to cleanly separate
   "calibration doesn't help morphology" from "task+label mismatch" (E47 caveat).

## Experiment index (real-data era)
- E37 `results/37_corrected_sampling/` — clean-signal physics, retracts E35/E36
- E38 `results/38_paired_transform/` — real paired modality profile (SJLIFE)
- E39 `results/39_recalibrated_augment/` — open-loop overshoots (negative)
- E40 `results/40_closed_loop_calib/` — closed-loop hits the target
- E41 `results/41_endtoend_auroc/` — first real end-to-end AUROC (5 seeds)
- E42 `results/42_seeds20_significance/` — significance (+0.041, p=0.009)
- E43 `results/43_multiaxis_calib/` — HF axis no-op
- E44 `results/44_icentia_seconddevice/` — dose-response (2nd device)
- E46 `results/46_fewshot_bridge/` — minimal-tuning bridge
- E47 `results/47_harder_task/` — rhythm-specific (morphology null)
- E48 `results/48_representation_invariance/` — learned invariance = augmentation (ceiling is info-bound)
