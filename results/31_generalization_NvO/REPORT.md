# E31 — Does the gap ladder GENERALIZE? Mostly NO (sobering negative)

**Question:** E26/E27b/E30 all used AF-vs-NORM (a very distinctive rhythm). The
biggest honesty risk: is "augmentation recovers ~48% of the gap, few-shot knee
at k~50" specific to AF's strong signature? E31 re-runs the key rungs on a
harder task — **Normal vs Other** (heterogeneous non-AF abnormal rhythm).

**Setup:** source = PTB-XL NORM vs non-NORM/non-AF abnormal (generic abnormal);
target = real CinC N-vs-O; 5 seeds; 1-lead ResNet.

## Results (AUROC mean±std, 5 seeds; Δ vs clean)

| Arm | AUROC | Δ vs clean | seeds + |
|---|---|---|---|
| clean Lead-I | 0.560 ± 0.033 | — | — |
| augment (s1.5, 5×) | 0.567 ± 0.018 | +0.007 | 3/5 |
| aug-pretrain + finetune k=50 | 0.597 ± 0.027 | +0.037 | 4/5 |
| oracle (real→real) | 0.758 ± 0.018 | +0.198 | 5/5 |

## Verdict: ❌⚠️ The ladder does NOT cleanly transfer to the harder task

- **The whole problem is harder and everything is lower.** Even the oracle only
  reaches **0.758** (vs 0.93 on AF/NORM) — Normal-vs-Other is genuinely hard
  even *with* real training data, because "Other" is a heterogeneous grab-bag.
- **Augmentation's benefit nearly vanishes: +0.007** (was +0.12 on AF/NORM).
  On a task without a crisp morphological signature, injecting recording-
  perturbation diversity does little — there's no robust class feature for it
  to protect.
- **Few-shot k=50 still helps directionally (+0.037, 4/5)** but far less than on
  AF/NORM, and it's nowhere near oracle (0.597 vs 0.758).

## What this means (honest reframe)

The impressive AF/NORM ladder was **partly a property of AF being an easy,
distinctive single-lead rhythm.** The modality-transfer methods work best
exactly where the task is already easy. On a harder, fuzzier label:
- the clinical→watch *task* gap and the *label difficulty* gap entangle;
- augmentation (which protects an existing strong feature) has little to grab;
- the honest conclusion is **our methods are task-dependent, not universal.**

This does NOT invalidate the AF/NORM results — it bounds their claim. For
distinctive rhythms (AF, and likely brady/tachy, pauses), the augmentation +
few-shot recipe is genuinely useful. For subtle/heterogeneous targets it is not
sufficient, and the bottleneck is task difficulty + label alignment, not just
recording modality.

## Honesty flags
- The clinical↔CinC label alignment for "abnormal/Other" is **approximate** —
  PTB-XL generic-abnormal ≠ CinC "Other" exactly. Some of the low numbers are
  label-mismatch, not pure modality gap. This is a stress test, not a clean
  benchmark (stated up front in the script).
- 5 seeds, CinC = E6c proxy not real AW.
- The right follow-up is a task where the label maps cleanly across datasets
  (e.g. a different but still well-defined rhythm), to separate "harder rhythm"
  from "label noise."
- Figure verified via PIL (1170×715); vision tool down this session.

## Follow-ups spawned
- **Claim scope corrected** across the log/README: the recipe is validated for
  distinctive rhythms (AF/NORM), NOT yet shown to generalize to subtle/
  heterogeneous tasks.
- **E34** — repeat on a cleanly-mapping non-AF rhythm to disentangle task
  difficulty from label mismatch.
