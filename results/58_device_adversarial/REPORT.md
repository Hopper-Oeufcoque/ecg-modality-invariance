# E58 — Device-adversarial content/style disentanglement (DANN + label anchor)

**Branch:** `explore/novel-methods` · **Idea:** N2 (`docs/NOVEL_IDEATION.md`)
**Date:** 2026-07-27 · **Seeds:** 20 · **Task:** real CinC 2017 AF vs NORM

## Question
Same goal as E57 — recover the alignment gain **without same-patient pairing** —
but through a different force. Instead of matching feature moments (CORAL) or
transporting mass (Sinkhorn), train a **device discriminator** (clinical-Lead-I
vs Apple-Watch) and use a **gradient-reversal layer (GRL)** so the encoder learns
features the discriminator *cannot* classify by modality — active adversarial
device-invariance (DANN, Ganin & Lempitsky 2015), plus the mandatory CE anchor.

Two variants:
- **dann** — GRL scrubs modality from the *whole* 32-d feature; classifier reads all 32.
- **dann_style** — content/style split: device head reads only an 8-d *style*
  sub-vector, classifier reads only the 24-d *content* sub-vector, GRL scrubs
  device from content by construction.

Watch pool = SJLIFE real Apple-Watch windows, **unpaired** (correspondence discarded).

## Result (AUROC on real CinC, 20 seeds)

| Arm | AUROC | vs clean | vs calibration | vs paired |
|-----|-------|----------|----------------|-----------|
| clean | 0.701 ± 0.048 | — | — | — |
| closed_aug (calibration) | 0.742 ± 0.047 | +0.041 | — | — |
| **dann** | **0.661 ± 0.047** | **−0.040 (p=0.005) HURTS** | −0.081 (p<1e-4) | −0.147 (p<1e-4) |
| **dann_style** | **0.707 ± 0.041** | +0.006 (p=0.66, null) | −0.035 (p=0.020) | −0.100 (p<1e-4) |
| joint_paired (E51) | 0.807 ± 0.023 | +0.106 | +0.065 | — |

## Verdict — ❌ NEGATIVE (plain DANN backfires; style-split neutral)
- **Plain DANN actively hurts** — 0.661, *below* the clean floor (−0.040, 3/20
  seeds, p=0.005). Forcing the whole feature vector to be modality-indistinguishable
  destroys discriminative morphology — the **E50 information-destruction failure
  mode**, now via an adversarial objective instead of a contrastive one. The CE
  anchor alone doesn't save it: the adversarial gradient is strong enough to drag
  the shared representation off the pathology manifold.
- **The content/style split repairs the damage but adds nothing** — quarantining
  the adversary to an 8-d style stub brings it back to 0.707 (≈ clean, p=0.66).
  Protecting the 24-d content channel stops the bleeding, but no positive transfer
  appears — there is no free modality-invariant signal to harvest from unpaired
  data adversarially.
- **Both lose decisively to paired** (0/20, p<1e-4) and to calibration.

## Interpretation — triangulates the E57 conclusion
E57 (passive distribution matching) recovered ~⅓ of the gain; E58 (active
adversarial matching) recovers **none**, and unconstrained even goes negative.
Two mechanistically opposite unpaired approaches both fail to reach calibration,
let alone paired. Together they make a strong triangulated case:

> The alignment gain lives in **same-patient correspondence** (E51b), not in any
> modality-marginal property of watch data. No unpaired objective — moment
> matching, optimal transport, or adversarial scrubbing — substitutes for
> knowing *which clinical trace and which watch trace are the same heart*.

And a second lesson reinforced: **invariance pressure not anchored to content is
destructive** (E50 contrastive collapse; E58 adversarial collapse). The CE anchor
is necessary but, for a *strong* adversary on the full feature, not sufficient —
you must also structurally protect the content channel (style-split), and even
then you only recover to neutral.

## Consequence for the goal
Kills N2 as a deployability shortcut. Confirms the paired set is load-bearing and
that our effort is better spent on (a) getting more/better paired data and
(b) methods that *use* correspondence more efficiently, than on trying to
manufacture invariance from unpaired watch traces.

## Honesty flags
- Watch pool is SJLIFE real AW but **unpaired** (pairing discarded).
- CinC dry-finger ≠ SJLIFE dry-wrist ≠ target AW wrist.
- AF-vs-NORM easy axis; λ=0.1 a priori; standard GRL schedule (2/(1+e^−10p)−1).
- Style-dim=8 chosen a priori (not swept); single clinical train set; 20 seeds.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/58_device_adversarial && \
  python3 experiments/58_device_adversarial.py > results/58_device_adversarial/run.log 2>&1
```
Figure: `results/58_device_adversarial/device_adversarial.png`
