# E60b — Higher-seed CONFIRMATION of real-AW sex transfer (15 seeds)

**Branch:** `explore/novel-methods` · **Date:** 2026-07-27 · **Seeds:** 15 (vs E60's 5) · **Test:** real SJLIFE Apple Watch, n=243 patients, female=1
**Purpose:** E60 showed aligned +0.031 vs clean at 5/5 seeds but only p=0.085. This run adds seeds to test whether the effect is real or small-sample luck.

## Result (Sex AUROC, female=1) — 5-seed vs 15-seed side by side

| Arm | E60 (5 seeds) | E60b (15 seeds) | shift |
|-----|---------------|-----------------|-------|
| clinical_self (PTB-XL internal) | 0.684 | 0.686 ± 0.023 | stable |
| clean → real AW | 0.680 | **0.695 ± 0.025** | +0.015 (floor rose) |
| closed_aug (calibration) | 0.697 | 0.705 ± 0.022 | stable |
| band_scramble | 0.679 | 0.677 ± 0.032 | stable (null) |
| **aligned** | 0.712 | **0.709 ± 0.030** | stable |
| oracle | 0.692 | 0.692 ± 0.017 | identical |

**Comparisons (15 seeds):**
- aligned_vs_clean: **Δ=+0.014, 10/15 wins, p=0.080** (was +0.031 at 5 seeds)
- closed_aug_vs_clean: Δ=+0.010, 10/15, p=0.12
- band_scramble_vs_clean: Δ=−0.018, 4/15, p=0.075 (trends *negative*)
- aligned_vs_closed_aug: Δ=+0.004, 9/15, p=0.60 (**alignment ≈ calibration here**)
- oracle_vs_clean: Δ=−0.003, 6/15, p=0.75 (**oracle = clean, no better**)

## Verdict — ⚠️ PARTIAL RETRACTION of E60's enthusiasm; the direction holds, the magnitude shrank
Honest reassessment. Adding seeds moved three things:

**1. The clean floor rose (0.680→0.695) and the aligned gain halved (+0.031→+0.014).**
The 5-seed +0.031 was partly small-sample luck on the clean baseline. At 15 seeds
aligned is still the **numerically top transfer arm (0.709)** and still beats clean in
2/3 of seeds (10/15), but the effect is **+0.014, p=0.080 — not significant**, and it is
now **statistically indistinguishable from calibration** (Δ+0.004, p=0.60). The clean
separation between alignment and the cheaper levers that we saw on CinC does **not**
reproduce on real-AW sex.

**2. aligned no longer clearly beats oracle.** E60's headline surprise (aligned 0.712 >
oracle 0.692) is real in means (0.709 vs 0.692) but within noise at 15 seeds. I should
not have leaned on it as hard as I did — it's a soft trend, not a demonstrated result.
oracle ≈ clean (p=0.75) is itself the more robust finding: **training on 243×3 real-AW
sex labels buys essentially nothing over naive clinical transfer** — the watch set is
too small to learn from directly, which is consistent, but the "alignment beats it"
framing needs the bigger effect it doesn't have.

**3. The one thing that held rock-solid: sex morphology transfers nearly intact.**
clinical_self 0.686 ≈ clean 0.695 across 15 seeds. This is the durable real-data
finding — an *in-band* morphological signal crosses clinical→real-AW with no measurable
loss, unlike out-of-band rhythm. Everything modality-invariance-method-related is a
second-order effect *on top of* an already-mostly-transferring signal, which is exactly
why the lever gains are small here (little gap left to close — E44 dose-response again).

## What this means honestly
- **Alignment is directionally best on real Apple Watch, but the real-AW sex effect is
  small and not yet significant (p≈0.08 at 15 seeds).** It is NOT the +0.10-scale win we
  measured on CinC AF. Claiming a confirmed real-AW alignment win would be overreach.
- The likely reason is mechanistic, not a failure: **sex is in-band and already transfers
  (0.68→0.70), so there is little modality gap for any invariance lever to recover** — the
  dose-response law (E44/E56) predicts small gains exactly here. This would also predict
  that a *disease* endpoint with real out-of-band modality sensitivity would show a larger
  alignment gain — but we cannot test that without disease labels.
- **band_scramble trends negative on real AW** (−0.018) — the free lever that tied
  calibration on CinC does not help (maybe hurts) on this in-band task. Consistent: scramble
  perturbs the low band, which carries no sex signal, adding noise without benefit.

## Consequence for the goal
Real-AW sex is a **weak test bed for modality-invariance methods** precisely because the
signal is modality-robust to begin with — good news for "does clinical morphology reach the
watch" (yes), unhelpful for "do our levers add value on the watch" (can't tell from sex).
The measured-real-AW win we wanted still hinges on a **disease endpoint with genuine
out-of-band modality sensitivity** (rhythm/conduction), i.e. real disease labels on a paired
cohort. E60's core correction stands: we can now *measure* on real AW, but sex isn't the
outcome that discriminates the methods.

## Honesty flags
n=243 real-AW test; 15 seeds; population shift (PTB-XL old/cardiac vs SJLIFE young/cancer-
survivor) confounds modality; ~3 windows/patient; aligned_vs_clean p=0.080 NOT significant;
E60's aligned>oracle claim softened to a within-noise trend; sex is in-band → weak method
discriminator; leakage-safe patient-disjoint folds.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/60b_real_aw_sex_confirm && \
  python3 experiments/60b_real_aw_sex_confirm.py > results/60b_real_aw_sex_confirm/run.log 2>&1
```
Figure: `results/60b_real_aw_sex_confirm/real_aw_sex_confirm.png`
