# E42 — Significance at 20 seeds: closed-loop calibration is real but smaller

**Date:** 2026-07-27
**Setup:** identical to E41 (train PTB-XL Lead-I AFIB/NORM → test held-out real
CinC 2017 AF/N, calibrate to unlabeled CinC-ref bw), rerun at **20 seeds** on the
two decisive arms (clean, closed_loop) + oracle. Purpose: settle E41's p=0.086.

## Result

| arm | AUROC (20 seeds) |
|---|---|
| clean | 0.701 ± 0.048 |
| closed_loop | 0.742 ± 0.047 |
| oracle | 0.931 ± 0.014 |

- **Δ closed−clean = +0.041**, wins **15/20** seeds.
- **paired t=2.92, p=0.0088** · Wilcoxon p=0.0094 · Cohen dz=0.65 (medium).
- **Significant at 0.05 ✅** — both parametric and non-parametric agree.

## The honest correction to E41
E41's 5-seed headline (+0.072, p=0.086) was **optimistic small-sample noise**.
At 20 seeds the effect **halves to +0.041** and clean drifts up (0.681→0.701).
This is textbook regression-to-the-mean — the 20-seed number is the trustworthy
one. **The effect is genuine and significant, but modest (~half the E41 claim).**
The verdict table and synthesis are updated to the E42 numbers.

## What stands
- Closed-loop modality calibration — fit with **zero test labels** (unlabeled
  target distribution only) — gives a **real, statistically significant** lift in
  clinical→real-single-lead transfer AUROC (+0.041, p=0.009, dz=0.65).
- It closes ~18% of the clean→oracle gap (0.701→0.742 of 0.701→0.931).
- 15/20 directional wins → consistent, not a few-seed fluke.

## What this does NOT say (honesty flags)
- Effect is **modest** (+0.041). Not a magic transfer fix; the bulk of the
  clean→oracle gap (0.19) remains — real labels still dominate.
- **CinC finger-electrode ≠ Apple Watch wrist** (closest LABELED real proxy).
- **AF vs NORM = easy distinctive-rhythm task** (E31: harder tasks generalize
  worse — this lift may not carry to subtle morphology tasks).
- Single-site clinical source; single fixed clinical training set across seeds
  (only CinC split + init vary), so the CI reflects test/init variance, not
  clinical-cohort variance.

## Verdict ✅ (significant, modest)
Confirms the E41 mechanism at proper power: measuring the real target
distribution (unlabeled) and closing the calibration loop to it produces a real,
significant AUROC lift on real labeled single-lead data. Magnitude corrected
down to +0.041. Strongest properly-powered zero-label result in the project.

## Follow-ups
- Vary the clinical training subset across seeds → get a CI that includes
  cohort variance (current CI is optimistic on that axis).
- Multi-axis closed-loop (E40 residual qrs/mid) — can it push past +0.041?
- Harder task (Normal-vs-Other on CinC) → does significance survive E31's warning?

## Artifacts
`metrics.json`, `significance.png` (bar + paired per-seed scatter).
