"""Apple-Watch ECG generator (clinical Lead-I -> AW-style training data).

THE PROJECT'S NORTH-STAR TOOL. We have abundant clinical 12-lead ECGs but few
Apple Watch ECGs. This module turns a clinical Lead-I signal into an
Apple-Watch-style signal usable for TRAINING models that target Apple Watch.

Design principles (derived empirically from experiments E6/E6b/E6c/E22/E23/E24):
  1. REAL Apple Watch is a CLEAN signal (E6c: kurtosis ~12, entropy ~0.32),
     NOT noisy. The old forward-physics simulator over-noised and HURT transfer
     (E6b: sim-train 0.737 < clean-train 0.753 on real). So: light touch only.
  2. The clinical->watch difference is dominated by the RECORDING-CHAIN SPECTRAL
     TRANSFER (filter/electrode response), not additive noise. We LEARN this
     transfer function empirically from a real single-lead reference distribution
     instead of hand-coding Apple's bandpass (E22: the hand-coded bandpass was the
     kurtosis-killer).
  3. PRESERVE MORPHOLOGY = PRESERVE THE LABEL. The transfer is applied as a
     zero-phase magnitude filter (multiply rfft magnitude by H(f), keep phase),
     so P-QRS-T timing/shape — where the diagnosis lives — is preserved. The
     generated sample keeps the source clinical label valid by construction.
  4. TARGET = CinC 2017 handheld single-lead, which E6c validated as an excellent
     proxy for real Apple Watch (distance 0.247) and is fully open (PhysioNet) —
     so we can train the generator toward it WITHOUT touching HOME's eval-only data.

Usage:
    from src.aw_generator import build_transfer_function, AWGenerator
    H = build_transfer_function(clinical_leadI_sigs, cinc_sigs, fs=100)
    gen = AWGenerator(H, fs=100, noise_level=0.02)
    aw_style = gen.generate(clinical_leadI_signal)   # ready for training
"""

from __future__ import annotations

import numpy as np


def _fix_norm(x, siglen=1000):
    """Truncate/pad to siglen, per-record z-normalize."""
    x = np.asarray(x, dtype=np.float64)
    if x.size >= siglen:
        x = x[:siglen]
    else:
        x = np.concatenate([x, np.zeros(siglen - x.size)])
    return (x - x.mean()) / (x.std() + 1e-9)


def _smooth(v, k=7):
    """Simple moving-average smoothing of a 1-D array (odd k)."""
    if k <= 1:
        return v
    kern = np.ones(k) / k
    return np.convolve(v, kern, mode="same")


def build_transfer_function(clinical_sigs, target_sigs, fs=100.0, siglen=1000,
                            smooth_k=9, clip=(0.2, 5.0)):
    """Empirical magnitude-spectrum transfer function H(f) = |target| / |clinical|.

    Averages the magnitude spectrum over each domain, takes the ratio, smooths,
    and clips to avoid extreme gains. Applying H to a clinical Lead-I reshapes its
    spectral envelope to match the target (real-single-lead) distribution while
    preserving phase (morphology).

    clinical_sigs / target_sigs: iterables of 1-D signals (any length; fixed here).
    Returns H: real array of length siglen//2 + 1 (rfft bins).
    """
    def mean_mag(sigs):
        acc = None; n = 0
        for s in sigs:
            s = _fix_norm(s, siglen)
            mag = np.abs(np.fft.rfft(s))
            acc = mag if acc is None else acc + mag
            n += 1
        return acc / max(n, 1)

    Hc = mean_mag(clinical_sigs)
    Ht = mean_mag(target_sigs)
    H = Ht / (Hc + 1e-9)
    H = _smooth(H, smooth_k)
    H = np.clip(H, clip[0], clip[1])
    return H


class AWGenerator:
    """Clinical Lead-I -> Apple-Watch-style generator.

    H: transfer function from build_transfer_function().
    noise_level: std of additive light noise as a fraction of signal std. Real AW
        is clean (E6c) so keep this SMALL (~0.02-0.05). Default calibrated to the
        real-AW residual (much lighter than the old simulator's ~0.15-0.30).
    baseline_boost: optional gentle <0.7 Hz baseline-wander boost (real AW has more
        low-freq energy than clinical, E6c PSD). Fraction of signal std.
    """

    def __init__(self, H, fs=100.0, siglen=1000, noise_level=0.03,
                 baseline_boost=0.04, seed=None):
        self.H = np.asarray(H, dtype=np.float64)
        self.fs = fs
        self.siglen = siglen
        self.noise_level = noise_level
        self.baseline_boost = baseline_boost
        self.rng = np.random.default_rng(seed)

    def generate(self, clinical_leadI, rng=None):
        """Transform one clinical Lead-I signal into an AW-style signal.

        Zero-phase magnitude filtering (preserves morphology/phase) + light noise.
        Returns a z-normalized 1-D array of length siglen.
        """
        rng = rng or self.rng
        x = _fix_norm(clinical_leadI, self.siglen)
        X = np.fft.rfft(x)
        mag = np.abs(X); phase = np.angle(X)
        # apply learned spectral transfer to the MAGNITUDE, keep PHASE (morphology)
        mag_new = mag * self.H[: mag.size]
        Y = mag_new * np.exp(1j * phase)
        y = np.fft.irfft(Y, n=self.siglen)
        # gentle low-freq baseline boost (real AW has more <0.7 Hz energy, E6c)
        if self.baseline_boost > 0:
            t = np.arange(self.siglen) / self.fs
            f = rng.uniform(0.15, 0.6); phi = rng.uniform(0, 2 * np.pi)
            y = y + self.baseline_boost * np.std(y) * np.sin(2 * np.pi * f * t + phi)
        # light broadband noise (real AW is CLEAN — keep small)
        if self.noise_level > 0:
            y = y + self.noise_level * np.std(y) * rng.standard_normal(self.siglen)
        return (y - y.mean()) / (y.std() + 1e-9)

    def generate_batch(self, clinical_sigs, rng=None):
        return np.stack([self.generate(s, rng=rng) for s in clinical_sigs])


class StochasticAWAugmenter:
    """Phase-A' generator: label-preserving STOCHASTIC augmentation.

    E25b lesson: a faithful, near-information-preserving spectral transfer is
    NEUTRAL as training data (fidelity != utility). The utility of synthetic
    data came from augmentation DIVERSITY / noise-robustness, and the crude
    heavy sim actually gained the most. So instead of trying to look like the
    target, we inject the kinds of *label-preserving* variation that real
    single-lead wearable recordings have but clinical Lead-I lacks:

      - electrode-contact DROPOUTS (brief low-amplitude / flat segments)
      - MOTION bursts (transient localized high-amplitude wander)
      - dry-electrode GAIN WANDER (slow multiplicative amplitude drift)
      - variable BASELINE dynamics (multi-tone low-freq wander)
      - mild broadband noise + random global amplitude scale
      - optional light spectral shaping via a supplied transfer function H

    Every perturbation is morphology-order-preserving in the QRS band (we never
    time-warp or invert), so the source clinical label stays valid. Each call is
    RANDOM -> use to expand a small clinical set into a diverse training corpus.

    Parameters are randomized per-sample within ranges; `strength` scales them.
    """

    def __init__(self, fs=100.0, siglen=1000, H=None, strength=1.0, seed=None):
        self.fs = fs
        self.siglen = siglen
        self.H = None if H is None else np.asarray(H, dtype=np.float64)
        self.strength = strength
        self.rng = np.random.default_rng(seed)

    def _apply_H(self, x):
        X = np.fft.rfft(x)
        mag = np.abs(X) * self.H[: X.size]
        return np.fft.irfft(mag * np.exp(1j * np.angle(X)), n=self.siglen)

    def generate(self, clinical_leadI, rng=None):
        rng = rng or self.rng
        s = self.strength
        x = _fix_norm(clinical_leadI, self.siglen)
        t = np.arange(self.siglen) / self.fs

        # optional light spectral shaping toward the wearable band
        if self.H is not None and rng.random() < 0.5:
            x = self._apply_H(x)

        # 1) dry-electrode GAIN WANDER: slow multiplicative amplitude drift
        n_tones = rng.integers(1, 4)
        gain = np.ones(self.siglen)
        for _ in range(n_tones):
            f = rng.uniform(0.05, 0.5); phi = rng.uniform(0, 2 * np.pi)
            gain = gain + s * rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * f * t + phi)
        x = x * np.clip(gain, 0.3, 1.8)

        # 2) variable BASELINE wander (multi-tone low-freq additive)
        base = np.zeros(self.siglen)
        for _ in range(rng.integers(1, 4)):
            f = rng.uniform(0.1, 0.8); phi = rng.uniform(0, 2 * np.pi)
            base = base + s * rng.uniform(0.05, 0.3) * np.sin(2 * np.pi * f * t + phi)
        x = x + base * np.std(x)

        # 3) electrode-contact DROPOUTS: brief attenuated segments
        if rng.random() < 0.5 * min(s, 1.0) + 0.2:
            for _ in range(rng.integers(1, 3)):
                w = int(rng.uniform(0.02, 0.12) * self.siglen)
                i0 = rng.integers(0, max(1, self.siglen - w))
                atten = rng.uniform(0.05, 0.5)
                ramp = np.ones(self.siglen)
                seg = np.linspace(atten, atten, w)
                # smooth edges so we don't inject step artifacts
                edge = max(1, w // 6)
                seg[:edge] = np.linspace(1.0, atten, edge)
                seg[-edge:] = np.linspace(atten, 1.0, edge)
                ramp[i0:i0 + w] = seg
                x = x * ramp

        # 4) MOTION bursts: transient localized high-amplitude wander
        if rng.random() < 0.4 * min(s, 1.0) + 0.15:
            for _ in range(rng.integers(1, 3)):
                w = int(rng.uniform(0.03, 0.15) * self.siglen)
                i0 = rng.integers(0, max(1, self.siglen - w))
                bf = rng.uniform(1.0, 8.0); phi = rng.uniform(0, 2 * np.pi)
                env = np.hanning(w)
                burst = s * rng.uniform(0.3, 1.2) * env * np.sin(
                    2 * np.pi * bf * (np.arange(w) / self.fs) + phi)
                x[i0:i0 + w] = x[i0:i0 + w] + burst * np.std(x)

        # 5) mild broadband noise + random global amplitude scale
        x = x + s * rng.uniform(0.01, 0.06) * np.std(x) * rng.standard_normal(self.siglen)
        x = x * rng.uniform(0.8, 1.25)

        return (x - x.mean()) / (x.std() + 1e-9)

    def generate_batch(self, clinical_sigs, rng=None):
        return np.stack([self.generate(s, rng=rng) for s in clinical_sigs])
