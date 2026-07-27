# E35 — Recalibrating toward REAL Apple Watch (n=1000): the honest picture

**Setup:** the full HOME real Apple Watch cohort — **1000 patient waveforms**
(vs E34's 20), 500→100 Hz, eval-only (statistics only, NO training). Profile
clinical / CinC / real AW and test 3 augmentation strategies for coverage +
morphology preservation.

## Modality profiles (mean per axis, n_AW=1000)

| axis | clinical | CinC | **REAL AW** | cinc-cal DR | aw-cal DR | light DR |
|---|---|---|---|---|---|---|
| kurtosis | 11.80 | 8.31 | 8.12 | 4.67 | 4.69 | 12.50 |
| bw_energy | 0.120 | 0.256 | **0.016** | 0.511 | 0.060 | 0.127 |
| qrs_energy | 0.380 | 0.264 | 0.259 | 0.207 | 0.190 | 0.377 |
| **hf_energy** | 0.019 | 0.004 | **0.165** | 0.010 | 0.394 | 0.019 |
| mid_energy | 0.312 | 0.384 | 0.357 | 0.181 | 0.142 | 0.309 |

## Morphology preservation (QRS-band corr / R-peak match)

| strategy | QRS corr | R-peak |
|---|---|---|
| cinc-calibrated DR | 0.967 | 0.977 |
| **aw-calibrated DR** | **0.594** ❌ | 0.882 |
| light DR | 0.985 | 0.973 |

## Verdict: ⚠️❌ n=1000 overturns n=20 — and reveals a bandwidth gap that can't be stat-matched

Four honest findings:

1. **n=20 (E34) was unreliable — the full n=1000 profile is different.** E34's
   subset showed real AW with LOW hf_energy (0.014, "clean like clinical"). The
   full cohort shows the opposite: **real AW hf_energy = 0.165 (~8× clinical) and
   very LOW baseline wander (0.016)**. A small-N profile misled us; this is a
   direct lesson to trust the n=1000 numbers, not the n=20 snapshot.

2. **The dominant clinical→watch difference is HIGH-FREQUENCY energy — and it's
   a FILTERING difference.** Clinical PTB-XL has *artificially* low HF (0.019)
   because clinical ECGs are heavily bandpass-filtered (~0.5–40 Hz). Real Apple
   Watch retains genuine broadband HF content (diagnostic check: the 30–50 Hz
   energy is smoothly decaying, **no 50/60 Hz powerline spike** → genuine signal
   character, not mains artifact). This directly confirms E22's "the gap is
   filter-bound" finding, now on real hardware.

3. **You CANNOT close it by matching the summary stat with noise.** The
   aw-calibrated augmenter successfully pushed hf_energy up (0.394) to cover real
   AW's 0.165 — but in doing so **destroyed QRS morphology (corr 0.594)**. Naive
   broadband HF injection is just noise; it wrecks the diagnostic waveform. The
   summary-stat-coverage idea (E33) has a hard limit: matching a moment ≠
   matching the structured content, and over-injecting corrupts the label.

4. **Light DR is the only morphology-safe option (QRS 0.985)** but doesn't add
   the HF content, so it doesn't cover the real-AW axis either.

## Honest limitations of this analysis

- **The standardized distance metric is dominated by the hf_energy axis** (where
  clinical std is tiny, so any gap is huge in standardized units) — all
  strategies land ~2.1–3.1, and "cinc closest (2.128)" is basically an artifact
  of that one axis. The distance number is NOT a reliable ranking here; the
  per-axis profile + morphology table are the trustworthy outputs.
- Still stats-based (5 axes), 500→100 Hz resampled (the >50 Hz watch content is
  gone — real HF gap may be even larger at native rate), HOME eval-only.

## The corrected takeaway (load-bearing)

The real clinical→Apple-Watch modality gap, measured on true hardware, is
dominated by a **bandwidth/filtering mismatch** (clinical is over-filtered; watch
keeps HF), NOT by the baseline-wander/noise our CinC-calibrated method targeted.
The right approach is **structured bandwidth matching that preserves morphology**
(e.g. gentle high-pass on clinical to de-emphasize the over-filtered look, or a
learned filter — NOT broadband noise injection). This supersedes the CinC-tuned
calibrated-DR direction for real-AW deployment.

## Follow-ups spawned
- **E36 — morphology-preserving bandwidth match:** apply a *filter* transform
  (match clinical→AW magnitude response) rather than noise injection; verify QRS
  corr stays >0.95 while hf_energy moves toward 0.165. This is the mechanistically
  correct fix and ties to the original E33 spectral-transfer idea (which was
  morphology-safe) — the spectral-transfer generator may have been right after
  all for THIS axis; combine it with light DR.
- **Correct claims:** README/SESSION_HANDOFF must state the deployment gap is
  bandwidth-dominated on real AW; CinC-calibrated numbers are proxy-only.
- Re-profile at native 500 Hz (don't resample away the HF) for a true picture.
