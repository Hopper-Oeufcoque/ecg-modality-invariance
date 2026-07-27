# E34 — Does calibrated-DR cover REAL Apple Watch? ❌ NO — the proxy misled us

**The deepest caveat, tested on real hardware.** Every result E26–E33b used CinC
2017 as an Apple Watch proxy, and our best method (E33 calibrated DR) calibrates
the augmenter toward CinC. E34 checks this against **20 real Apple Watch Lead-I
ECGs** (HOME dataset, 500 Hz→100 Hz, eval-only, license-compliant: distribution
statistics only, NO training on HOME).

## Modality profiles (mean per axis)

| axis | clinical | CinC (proxy) | **REAL AW** | calibrated-aug |
|---|---|---|---|---|
| kurtosis | 11.80 | 8.31 | **9.39** | 4.67 |
| bw_energy | 0.120 | 0.256 | **0.127** | 0.511 |
| qrs_energy | 0.380 | 0.264 | **0.340** | 0.207 |
| hf_energy | 0.019 | 0.004 | **0.014** | 0.010 |
| mid_energy | 0.312 | 0.384 | **0.354** | 0.181 |

## Key distances (mean standardized axis gap to real AW)

| from → real AW | distance |
|---|---|
| **clinical** | **0.253** |
| CinC (proxy) | 0.477 |
| calibrated-aug | **1.136** |

## Verdict: ❌ Three uncomfortable, important findings

1. **Clinical Lead-I is CLOSER to real Apple Watch (0.253) than CinC is
   (0.477).** On these modality axes, real AW looks *more like a clean clinical
   Lead-I* than like CinC handheld. CinC is noisier/more baseline-wander-heavy
   than a real Apple Watch, which is a genuinely clean signal (kurtosis 9.4,
   bw_energy 0.127 ≈ clinical's 0.120).

2. **CinC is a MEDIOCRE proxy for real Apple Watch** on these axes — worse than
   the raw clinical signal. This partially walks back E6c's "CinC ≈ AW (0.247)"
   claim: E6c used a different distance (whole-signal z-stats); on interpretable
   band-energy/kurtosis axes CinC is clearly *not* the closest thing to AW.

3. **Calibrated-DR (calibrated toward CinC) moved AWAY from real AW (1.136).**
   Because it calibrated to CinC's inflated baseline-wander (0.256) with a
   cover=1.3 margin, it pumped bw_energy to 0.511 — **4× past real AW's 0.127**.
   It optimized toward the wrong target. 2/5 axes covered, and mid/qrs energy
   went the wrong way.

## What this means (honest, load-bearing correction)

**The strong E33/E33b zero-shot numbers (0.82–0.89) are measured against CinC —
and CinC is not a faithful Apple Watch stand-in.** The calibrated-DR *mechanism*
is still sound (cover the target during training), but **it was calibrated to
the wrong distribution.** Calibrating toward CinC helps on CinC; it does not
necessarily help on real Apple Watch, and by these stats it likely *overshoots*
noise/wander that a clean Apple Watch signal doesn't have.

The single most consequential implication: **for real Apple Watch, a lighter
touch is probably better** — real AW is clean and close to clinical Lead-I, so
heavy domain randomization calibrated to noisy CinC is miscalibrated. This
echoes the original E6/E6b lesson (the old sim over-degraded because AW is
clean) — we partially re-made that mistake by trusting CinC.

## Honesty flags
- **n=20 real AW recordings** (40 windows) — a COARSE profile; means are
  indicative, not definitive. Needs the full HOME 1000-waveform set for a solid
  estimate (only 20 waveforms are in the repo's `data/ecg/`; the 1000-col files
  are the prediction cohort).
- Stats-based coverage on 5 axes — not full-manifold; other axes may differ.
- HOME is eval-only; NO training/fine-tuning done (license-compliant).
- This does NOT overturn the *relative* method findings (calibrated > hand-tuned
  > clean, all on CinC) — it overturns the *proxy's fidelity to real AW*, i.e.
  the external validity of the absolute numbers.

## Follow-ups spawned (high priority — this reorders the roadmap)
- **E35 — recalibrate DR toward REAL AW stats** (measured here) instead of CinC,
  and re-profile. Hypothesis: a lighter, AW-calibrated DR (low bw, low noise)
  covers real AW far better. This is now the most important experiment.
- **E36 — re-evaluate the proxy choice:** is clinical-Lead-I-with-light-DR the
  right zero-shot recipe for real AW, given AW ≈ clean clinical? Possibly the
  whole "add lots of variation" thrust is wrong for *clean* Apple Watch and
  right only for noisy handheld.
- **Correct external-validity claims** in README/SESSION_HANDOFF: CinC results
  are a proxy with limited AW fidelity; real-AW-calibrated methods pending.
- Get the full HOME waveform set (1000) for a trustworthy real-AW profile.
