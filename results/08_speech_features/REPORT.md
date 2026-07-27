# Experiment 8 — Speech-Channel-Robustness Features for ECG (Novel Cross-Domain)

**Date:** 2026-07-27 · **Repo:** ecg-modality-invariance · **Author:** Hopper

Borrowed from microphone-channel-robust speech recognition — decades of channel-
invariance research, **never applied to ECG.** Insight: ECG and speech are both
quasi-periodic signals; "channel" in speech (microphone/env) ≈ "electrode/recording
chain" in ECG. Ported the toolkit: MFCC-like cepstral features + CMVN (baseline
subtraction) + RASTA filtering (removes slow drift).

## Results (L4 full watch, single-lead)

| Variant | AUROC | Δ vs raw |
|---|---|---|
| V1 raw waveform (E17 reference) | 0.733 | — |
| V2 cepstral features (no channel-robustness) | 0.691 | −0.042 |
| V3 cepstral + CMVN (baseline subtraction) | 0.589 | −0.144 |
| V4 cepstral + CMVN + RASTA (full speech stack) | 0.539 | −0.194 |

## Verdict: ❌ — the speech-channel analogy does NOT transfer to ECG

The cepstral representation is *worse* than raw waveform, and **the more
speech-robustness I add, the worse it gets** (monotonic degradation: 0.733 →
0.691 → 0.589 → 0.539). CMVN and RASTA each *hurt* rather than help.

## Why it fails (the mechanism)
Speech and ECG differ in one critical way the analogy missed: **speech's
diagnostic content is in the spectral envelope (formants), which cepstral
features preserve; ECG's diagnostic content is in time-domain morphology and
phase (P-QRS-T shape, ST elevation, intervals), which cepstral features
destroy.** CMVN subtracts the mean cepstrum (per-patient baseline) — but that
baseline *is* the morphology. RASTA filters the cepstral trajectory — but ECG
"trajectory" carries rhythm info RASTA's high-pass removes. The speech toolkit
is optimized for a channel-invariance problem where the *channel* and the
*content* live in cleanly separable cepstral components; in ECG the channel
(modality) and content (pathology) are entangled in time-domain features, so
the toolkit removes the wrong thing.

## Lesson (valuable negative)
Not all cross-domain analogies transfer. The "ECG ≈ speech" intuition is
structurally real (both quasi-periodic), but the **representation matters**:
speech channel-robustness assumes a spectral-envelope representation where
content is separable from channel. ECG needs time-domain/phase representations
where they are not. This is a principled rejection of the I3 frontier idea —
save future researchers the same dead end. The right adjacent field is not
speech; it's **time-warp-stable signal processing** (scattering, E18) where
deformation invariance is proven *without* abandoning the time domain.

## Files
- `experiments/08_speech_features.py` · `results/08_speech_features/{metrics.json,speech_features.png,run.log}`
