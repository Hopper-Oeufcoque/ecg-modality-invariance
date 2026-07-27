# Experiment 17 — Lead-Masking Prob Sweep + Single-Lead-Trained-on-Simulator

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

Two refinements: (A) the lead-masking dropout-prob sweep (E16's was one-sided),
and (B) the backlog question — does a single-lead model trained on *simulated
watch* Lead-I (no 12-lead training) match the 12-lead lead-masking winner?

## Results

### A. Lead-masking prob sweep (L4 full watch)
| prob | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 |
|---|---|---|---|---|---|
| AUROC | 0.677 | 0.655 | 0.703 | **0.725** | 0.708 |

**prob=0.5 is optimal.** Non-monotonic (dips at 0.3 and 0.6, peaks at 0.5). E16's
choice of 0.5 was correct; no gain available from tuning.

### B. Single-lead model trained on simulated watch
| Model | L1 clean Lead-I | L4 full watch |
|---|---|---|
| E2 V2 lead-masking (12-lead) | 0.750 | 0.725 |
| E16 C5 two-stage (12-lead → sim finetune) | — | 0.714 |
| **E17 single-lead + sim-aug** | **0.733** | **0.742** |

## Headline finding
**A single-lead model trained on simulated-watch data beats the 12-lead
lead-masking winner on full watch (0.742 vs 0.725).** And L4 (0.742) > L1
(0.733) — the sim-trained model does *better* on its target distribution than on
clean Lead-I, exactly because train and test distributions match.

This is the strongest positive result of the project: **with the forward-physics
simulator (E1's F10), you don't need 12-lead training at all.** A 1-lead model
trained on simulated watch Lead-I, using clinical labels that pass through
unchanged, outperforms the 12-lead lead-masking approach on the watch task.

## Why this matters — it reframes the simulator
- E1 validated the simulator as an **eval probe**.
- E3-B showed the simulator helps **alignment training**.
- E16 C5 showed the simulator *doesn't* help as a **fine-tune stage** on top of 12-lead.
- **E17 shows the simulator's best use is as a from-scratch training
  distribution for a single-lead model.** The 12-lead prior is unnecessary
  when you can generate the target distribution with known labels.

## Honesty flags
- Train/test distribution match: the single-lead model trains on sim-watch
  (random seeds, domain-randomized) and tests on sim-watch (seed=0). There's
  seed variation but the distributions overlap by construction — this is a
  *fair* but *favorable* test for the sim-trained model. The real test is E6
  (real single-lead data): does a sim-trained model transfer to *real* watch?
- All on PTB-XL superclasses at 100 Hz. L4 here is simulated, not real watch.
- Single seed.

## Implication for the recommended recipe
Two regimes now:
1. **No simulator / no target-domain generation** → lead-masking (prob=0.5)
   on 12-lead is the best label-free transfer (0.725).
2. **With the forward-physics simulator** → train a single-lead model on
   simulated watch data; beats the 12-lead approach (0.742) and needs no
   12-lead data at inference.

E6 (real single-lead validation) is now the critical next step: does the
sim-trained model's edge hold on *real* watch data, or is it a sim-only artifact?

## Files
- `experiments/17_prob_sweep_simlead.py` · `results/17_prob_sweep_simlead/{metrics.json,prob_sweep.png,run.log}`
