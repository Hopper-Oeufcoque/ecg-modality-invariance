# E36 — Morphology-preserving bandwidth match: ✅ the correct real-AW fix

> ⛔ **RETRACTED (2026-07-27, by E37).** This experiment treated the HOME
> `data-for-predicting/Apple_Watch_waveform.csv` file as 500 Hz, but it is
> **200 Hz** (confirmed by README + heart-rate sanity check). The 2.5×
> frequency-axis error created a spurious "high-frequency / bandwidth gap." With
> correct sampling rates (E37), real AW hf_energy ≈ clinical (0.015 vs 0.019) —
> **there is no bandwidth gap and no need for this spectral transfer.** Read
> `results/37_corrected_sampling/REPORT.md` instead. Kept below for the record.

---


**Question:** E35 found the real clinical→Apple-Watch gap is a BANDWIDTH mismatch
(clinical over-filtered, hf_energy 0.018; real AW keeps HF, 0.161) and that
broadband noise injection matches the stat but destroys QRS (corr 0.594). Can a
**zero-phase spectral transfer** (magnitude reshaping toward real AW, phase kept)
close the HF gap while preserving morphology?

**Setup:** native **500 Hz** (no resampling away the HF band). Learn magnitude
transfer H(f) = |realAW| / |clinical| on a held-out half of the 1000-waveform
HOME cohort; apply to clinical PTB-XL 500 Hz NORM (phase preserved); profile
against the other AW half. Morphology via QRS-band corr + R-peak match.
License-compliant: transfer built from signal spectra only, NO training on HOME.

## Profiles at 500 Hz (mean per axis)

| axis | clinical | REAL AW | spectral-transfer | light DR |
|---|---|---|---|---|
| kurtosis | 11.26 | 11.82 | 12.84 | 11.75 |
| bw_energy | 0.136 | 0.014 | 0.032 | 0.144 |
| qrs_energy | 0.343 | 0.227 | 0.193 | 0.341 |
| **hf_energy** | 0.018 | **0.161** | **0.124** | 0.018 |
| mid_energy | 0.321 | 0.314 | 0.444 | 0.317 |

## Distance to real AW + morphology

| strategy | dist to AW | QRS corr | R-peak match |
|---|---|---|---|
| clinical (raw) | 2.431 | — | — |
| **spectral transfer** | **0.478** | **0.900** | **0.985** |
| light DR | 2.448 | 0.987 | 0.964 |
| (E35 broadband noise) | — | 0.594 ❌ | 0.882 |

## Verdict: ✅ Phase-preserving spectral transfer is the right mechanism

- **Closes 74% of the HF-energy gap** (0.018 → 0.124 toward real AW's 0.161)
  and cuts overall distance-to-real-AW **5×** (2.431 → 0.478).
- **Preserves the label:** R-peak match 0.985, QRS-band corr 0.900 — vs
  broadband noise injection (E35) which hit a similar HF stat but collapsed QRS
  to 0.594. This is the crux: **match the spectrum with a filter (phase kept),
  not with noise.** The magnitude envelope moves to the target while P-QRS-T
  timing/shape (the diagnosis) survives.
- **light DR does nothing for the bandwidth axis** (hf stays 0.018) — confirms
  the gap needs a *targeted spectral* transform, not generic perturbation.

## Honest limitations

- Stats + morphology study — **no AUROC** (would need labels/training on HOME,
  which we don't do). This shows the transform makes clinical *look like* real
  AW in the frequency domain while keeping morphology; it does NOT yet prove
  downstream AF/NORM transfer improves. That requires a labeled real-AW test set.
- QRS corr 0.900 is good but below light-DR's 0.987 — the aggressive HF boost
  does mildly roughen the waveform; a gentler transfer (lower clip, or blend
  with clinical) may trade a little HF coverage for higher morphology fidelity.
- bw_energy slightly over-suppressed (0.032 vs AW 0.014 — actually good
  direction); mid_energy overshoots (0.444 vs 0.314) — the transfer isn't
  perfect across all axes, HF is where it wins.
- 500 Hz Nyquist = 250 Hz; watch content above that (native ~512 Hz) is out of
  scope. Distance metric still HF-axis-weighted (read per-axis + QRS).
- Single transfer fit on NORM clinical; per-class/per-pathology transfer untested.

## The corrected real-AW recipe (supersedes CinC-calibrated DR for deployment)

**For real Apple Watch, the right zero-shot transform is a morphology-preserving
spectral bandwidth match** (clinical → real-AW magnitude envelope, phase kept) —
NOT the CinC-calibrated noise-injection DR (E33), which aimed at a mediocre proxy
and over-added baseline wander. E36 brings clinical 5× closer to real AW on the
true modality axes while keeping the label valid.

## Follow-ups spawned
- **E37 (needs labeled real AW):** train NORM/AF on spectral-transfer-matched
  clinical, test on a labeled real-AW set — the missing AUROC validation. HOME
  prediction cohort is unlabeled; would need Apple Heart Study or a labeled AW
  subset. Flag as data-gap.
- **Update `src/aw_generator.py`:** add `build_bandwidth_transfer` +
  `apply_spectral_transfer` (500 Hz, phase-preserving) as the real-AW recipe.
- **E36b:** gentler transfer (clip 0.3–4, blend α) to push QRS corr >0.95 while
  keeping most HF coverage.
- Correct README/SESSION_HANDOFF: real-AW deployment = spectral bandwidth match,
  not CinC-calibrated DR.
