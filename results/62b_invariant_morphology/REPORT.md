# E62b — Invariant handcrafted features on MORPHOLOGY (N5 falsification control)

**Branch:** `explore/novel-methods` · **Date:** 2026-07-27 · **Seeds:** 20 · **Test:** real CinC N-vs-O (morphology), the E47/E53 task
**Controls:** E62 (same pipeline, AF/timing task)

## Question
E62 found invariant handcrafted features hit 0.908 on CinC AF (+0.208 vs deep clean), but
78% of that was CV-of-RR and AF *is* rhythm irregularity → the win might be timing-tautological.
This control runs the **identical** feature pipeline on a **morphology** task (NORM vs Other,
where signal lives in wave shape, not timing). Prediction: the margin should collapse if E62
was timing-specific.

## Result (AUROC on real CinC N-vs-O morphology, 20 seeds)

| Arm | E62 (AF/timing) | E62b (morphology) | margin shift |
|-----|-----------------|-------------------|--------------|
| clean (deep) | 0.701 | 0.561 ± 0.022 | task much harder |
| closed_aug | 0.742 | 0.558 ± 0.017 | calibration null here (as E47/E53) |
| **handcrafted** | **0.908** | **0.610 ± 0.012** | — |
| ensemble | 0.883 | 0.607 ± 0.022 | — |
| **handcrafted vs clean** | **+0.208** | **+0.049 (20/20, p<1e-4)** | **4× smaller** |
| ensemble vs handcrafted | −0.025 | −0.003 (p=0.45, null) | no complementarity |

## Verdict — ✅ CONTROL CONFIRMS E62 IS TIMING-SPECIFIC (but a small real gain survives)
**The main finding: the handcrafted margin collapses 4×** on morphology (+0.208 → +0.049).
This confirms the E62 headline was overwhelmingly a **timing/rhythm** effect — the huge AF
number came from RR-irregularity being a near-direct AF readout, exactly as flagged. On a
morphology task where amplitude/shape carries the signal, the amplitude-invariant features
that dominated E62 lose almost all their power. The honest boundary is now mapped: **invariant
handcrafted features are a rhythm-family lever, not a general-purpose one.**

**The nuance worth keeping: a small but unanimous +0.049 survives even on morphology**
(20/20 seeds, p<1e-4). Two candidate explanations, both honest:
1. The spectral-shape + Hilbert-phase features carry *some* morphology signal that is also
   modality-robust (normalized PSD shape, phase structure survive amplitude/baseline shift).
2. On this hard task the deep model is near-floor (clean 0.561, calibration null 0.558 — the
   modality gap plus a weak catch-all label leave it barely above chance), so even a weak but
   modality-robust feature set edges it. Note the ABSOLUTES are low (0.61) — this is "least-bad
   on a hard task," not a strong classifier.

**Complementarity fails again (consistent with E62):** ensemble ≈ handcrafted (−0.003, p=0.45).
Adding the deep model neither helps nor clearly hurts here (both are near-floor, so mixing is
neutral rather than the E62 drag). The SignalMC-MED F6 "handcrafted complementary to learned"
claim does not hold on either task in our setting.

## Synthesis of N5 (E62 + E62b together)
- **Invariant-by-construction features dodge the modality gap** — mechanically clean, verified
  bit-exact under the SJLIFE transform. That principle is real.
- **Their VALUE is task-dependent and tracks how much of the label is timing vs morphology:**
  AF (pure timing) → +0.208 dominant win; morphology → +0.049 marginal. This is the SAME
  rhythm-vs-morphology axis seen in E47 (calibration rhythm-specific) and E53 (alignment
  generalizes but weaker on morphology). The dose-response intuition extends: **a method's
  gain tracks how much of the target signal lives in the axis the method protects.**
- **Deployment recommendation (refined):** timing/rhythm AW tasks (AF, PVC burden, HRV) →
  invariant handcrafted features are the strongest, cheapest, most robust route. Morphology
  tasks (sex, MI, hypertrophy, LVEF) → deep model + alignment/calibration; handcrafted adds
  little. Don't ensemble across a strong+weak pair.

## Honesty flags
CinC-O is a weak catch-all label (E53 oracle only ~0.75); morphology intrinsically harder;
absolutes here are low (0.56–0.61, barely above chance) so treat the +0.049 as "least-bad on a
hard task," not a good classifier; same feature set as E62; 20 seeds; CinC ≠ real AW wrist.

## Repro
```
cd ~/projects/ecg-modality-invariance && source .venv/bin/activate
mkdir -p results/62b_invariant_morphology && \
  python3 experiments/62b_invariant_morphology.py > results/62b_invariant_morphology/run.log 2>&1
```
Figure: `results/62b_invariant_morphology/invariant_morphology.png`
