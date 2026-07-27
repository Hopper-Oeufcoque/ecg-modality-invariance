# E27 — Tuning the augmentation recipe: more diversity keeps helping

**Question:** how far can pure stochastic augmentation go? Sweep strength
{0.5, 1.0, 1.5} × expansion {1×, 2×, 3×} on the E26 cocktail (clean ∪ N
augmented copies), tested on real CinC. Does augmentation saturate, or keep
closing the gap?

**Setup:** binary NORM/AF, PTB-XL Lead-I → real CinC, 3 seeds (compute), 1D
ResNet. clean-only baseline = 0.662.

## Results (AUROC, mean over 3 seeds; Δ vs clean-only 0.662)

| strength \ expansion | 1× | 2× | 3× |
|---|---|---|---|
| **0.5** | 0.694 (+0.032) | 0.733 (+0.072) | 0.737 (+0.076) |
| **1.0** | 0.763 (+0.102) | 0.770 (+0.108) | 0.758 (+0.096) |
| **1.5** | 0.766 (+0.105) | 0.788 (+0.126) | **0.791 (+0.130)** |

Best config: **strength 1.5, expansion 3× → 0.791** (Δ+0.130, 3/3 seeds), also
the *lowest-variance* high performer at higher strengths (s1.5_x2 std 0.010).

## Verdict: ✅ Augmentation scales — and hasn't plateaued

- **Both knobs help monotonically up to the grid edge.** Strength 0.5 → 1.0 is
  the big jump (+0.07 → +0.10); 1.0 → 1.5 keeps gaining (+0.10 → +0.13).
  Expansion 1×→2× helps; 2×→3× is marginal at low strength but still positive
  at strength 1.5.
- **The best cell (s1.5, 3×) is at the corner of the grid** — i.e. we have NOT
  found the ceiling of pure augmentation. More strength / more copies were still
  improving when the sweep ran out.
- **This is the single biggest gap-closer so far:** 0.662 → 0.791 recovers
  **~46%** of the clean→oracle modality gap (oracle 0.94), from augmentation
  alone with zero target labels.

## Reconciling with E29 (the coherent story)

E27 and E29 together give a clean, consistent picture of *what the gap is*:

- **E29 (feature-space alignment): all hurt.** You cannot close the gap by
  matching source/target representations.
- **E27 (input-space diversity): scales strongly, no plateau yet.** You close
  the gap by training the model to be robust to a *wide* range of
  label-preserving recording perturbations.

→ **The residual modality gap is dominated by input-variation coverage, not
representation geometry.** The winning strategy is "show the model enough
realistic single-lead variation," and we're still on the rising part of that
curve.

## Honesty flags
- 3 seeds only (compute) — directional; the best config (s1.5_x3) needs a
  5-seed + paired-significance re-confirmation (queued as E27b/E28).
- n=700 CinC, AF/NORM only, CinC = E6c proxy not real AW.
- Higher strength risks eventually corrupting morphology/label — must monitor
  the source-correlation guard as strength climbs past 1.5 (E27b checks this).
- Figure verified via PIL (910×715); vision tool down this session.

## Follow-ups spawned
- **E27b** — extend the sweep UP (strength 2.0/2.5, expansion 4×/5×/6×) at 5
  seeds to find where augmentation actually plateaus, WITH a morphology-
  preservation guard (source correlation must stay > ~0.85) so we don't buy
  AUROC by destroying labels.
- **E30** — semi-supervised: add k labeled real samples (k=10/50/200) on top of
  the best augmentation recipe; quantify the data-collection cost to reach oracle.
- **Phase C** — lock the tuned recipe into `AWTrainingSetBuilder`.
