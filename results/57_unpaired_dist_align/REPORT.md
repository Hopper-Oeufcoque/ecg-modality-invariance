# E57 — Can unpaired distribution alignment replace paired alignment?

**Branch:** `explore/novel-methods` · **Idea:** N1 (`docs/NOVEL_IDEATION.md`)
**Date:** 2026-07-27 · **Seeds:** 20 · **Task:** real CinC 2017 AF vs NORM

## Question
The confirmed headline (E51) needs **paired** same-patient clinical↔Apple-Watch
data (SJLIFE) for its InfoNCE alignment term. Paired cross-hardware ECG is the
rarest ingredient in the whole pipeline. **Can we recover the alignment gain from
an UNPAIRED pile of watch traces instead** — matching feature *distributions*
rather than instance *correspondences*? If yes, the method deploys against any
bag of unlabeled wearable ECG, no pairing required.

Two unpaired distribution-alignment losses, both borrowed from vision domain
adaptation, both with our mandatory CE anchor (same recipe as E51, only the
alignment term swapped):
- **CORAL** — match the 2nd-order statistics (feature covariance) of the
  clinical-Lead-I and watch feature clouds.
- **Sinkhorn** — entropy-regularized optimal transport between the two feature
  clouds (matches the full marginal, not just covariance).

Watch pool = SJLIFE real Apple-Watch features, but with **pairing discarded**
(the model never sees which clinical trace matches which watch trace).

## Result (AUROC on real CinC, 20 seeds)

| Arm | AUROC | vs clean | vs calibration | vs paired |
|-----|-------|----------|----------------|-----------|
| clean | 0.701 ± 0.048 | — | — | — |
| closed_aug (calibration) | 0.742 ± 0.047 | +0.041 | — | — |
| **coral (unpaired)** | **0.715 ± 0.049** | +0.014 (p=0.31, n.s.) | −0.027 (p=0.14) | −0.092 (p=2e-7) |
| **sinkhorn (unpaired)** | **0.730 ± 0.055** | +0.029 (p=0.053, borderline) | −0.012 (p=0.52) | −0.077 (p=1e-5) |
| joint_paired (E51) | 0.807 ± 0.023 | +0.106 | +0.065 | — |

## Verdict — ⚠️ PARTIAL / mostly NEGATIVE
Unpaired distribution matching recovers **only a fraction** of the paired gain:
- Sinkhorn gets a whiff of signal (+0.029 over clean, but only borderline
  p=0.053, and statistically indistinguishable from plain calibration).
- CORAL is effectively null (+0.014, p=0.31).
- **Both lose decisively to paired alignment** (0/20 seeds, p<1e-4). Paired
  keeps its tight 0.023 std; the unpaired arms stay at the wide ~0.05 clean-band
  variance — i.e. they don't even buy the variance-halving that paired does.

**Interpretation.** The value of the SJLIFE alignment is in the *same-patient
correspondence*, not in the marginal watch distribution. This is the E51b
falsification result seen from the other side: E51b shuffled *which* watch trace
pairs with *which* clinical trace and the gain vanished (p=0.76); here we throw
away pairing entirely and only ~⅓ of the gain survives — and that residue is no
better than the far cheaper calibration lever. Aligning the *shape of the cloud*
is not the same as aligning *the same heart across two devices*. The mechanism
is genuinely relational, not distributional.

## Consequence for the goal
The rarest ingredient (paired cross-hardware data) stays load-bearing — we cannot
swap it for cheap unpaired watch traces without collapsing to calibration-tier
gains. **BUT** it also sharpens the value proposition: if a lab can collect even a
few hundred same-patient clinical↔watch pairs (no disease labels needed, as
SJLIFE shows), that paired set is worth far more than a large unpaired watch
corpus. Directs future data-collection effort toward pairing, not volume.

## Honesty flags
- Watch pool is SJLIFE real AW but **unpaired** (pairing deliberately discarded).
- CinC electrode (dry finger) ≠ SJLIFE electrode (dry wrist) ≠ target (AW wrist).
- AF-vs-NORM is the easy morphological axis.
- λ=0.1 chosen a priori (not tuned for CORAL/Sinkhorn — possible they'd do
  marginally better tuned, but the paired gap is far too large to close).
- Single clinical train set, 20 seeds, single split family.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/57_unpaired_dist_align && \
  python3 experiments/57_unpaired_dist_align.py > results/57_unpaired_dist_align/run.log 2>&1
```
Figure: `results/57_unpaired_dist_align/unpaired_dist_align.png`
