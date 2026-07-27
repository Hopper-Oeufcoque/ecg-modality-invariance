# E20 — Set-invariant DeepSet over leads

**Date:** 2026-07-27 · **Status:** ⚠️ competitive, does not beat lead-masking

## Hypothesis
Treating the 12 ECG leads as an **unordered set** with permutation-invariant
pooling (DeepSets, Zaheer et al. 2017 — a point-cloud / set-encoding architecture
*never applied to ECG modality invariance*) makes the model robust to lead count
by construction: a single lead is just a 1-element set, so the same network
handles 12-lead, partial-lead, and 1-lead inputs without retraining. The hope was
that set-invariance beats lead-masking (which hacks lead-count robustness via
dropout) by being *architecturally* invariant rather than data-augmentation invariant.

## Setup
- **Architecture:** DeepSet — per-lead 1D ResNet encoder (shared weights across
  leads) → mean-pool over the lead dimension → classifier. ~0.5M params.
- **V1 (+lead-masking):** train on 12-lead with random lead masking (prob=0.5),
  evaluate on L1 (clean Lead-I) and L4 (full sim-watch).
- **V2 (+sim-watch):** train the single-lead path on sim-watch Lead-I with
  clinical labels (the E17 recipe, but through the DeepSet architecture).
- `experiments/20_deepset_leads.py`, 20 ep, single seed, PTB-XL 100 Hz.

## Result
- V1 DeepSet+leadmask @ L1 = — · @ L4 = **0.698**
- V2 DeepSet+sim @ L1 = 0.737 · @ L4 = **0.708**

Comparison: lead-masking (E2-V2) = 0.718 · single-lead+sim (E17) = 0.742.

## Verdict: ⚠️
Competitive (0.708, within noise of 0.718) but does not beat either leader.
Mean-pooling across leads is the right invariance for *lead count*, but it
discards **lead identity** — and lead identity carries real signal when leads are
present (the 12-lead prior that made E2-V2 beat the single-lead reference). The
architecture is invariant at the cost of throwing away the spatial information
that makes 12-lead training valuable in the first place.

## Lesson
Set-invariance is elegant but **over-invariants** for the watch-only transfer
task. When a lead IS available at inference (watch = always Lead-I), its identity
is fixed and informative; forcing permutation-invariance over a singleton set
gains nothing and loses the multi-lead prior. The DeepSet's genuine value is for
a **unified clinic+watch model** where the number of available leads varies at
inference (some inputs 12-lead, some 1-lead, some partial) — there it gives a
single architecture that gracefully degrades, which lead-masking + zero-pad
approximates but less cleanly. For the fixed-target watch task, lead-masking
remains superior because it keeps lead identity in the channel dimension.

## Honesty flags
Single seed, single architecture config (mean-pool; max-pool or attention-pool
untested — attention could recover lead identity and is the natural next step).
No CIs; 0.708 vs 0.718 is within expected seed noise. sim-watch not real-watch
(see E6 caveat).

## Follow-ups
- **E20b — attention-pool DeepSet:** replace mean-pool with attention over leads
  so the model can weight lead identity when present and still pool a singleton.
  This may recover the 12-lead prior the mean-pool discarded.
- The unified clinic+watch variable-lead-count use case is the better home for
  this architecture — queued as a deployment-architecture experiment.

## Artifacts
- `results/20_deepset_leads/metrics.json`
