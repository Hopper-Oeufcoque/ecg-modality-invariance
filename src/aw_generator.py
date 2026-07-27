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


# ----------------------------------------------------------------------------
# Target-CALIBRATED domain randomization (E33 — the zero-shot best recipe).
# Domain randomization only gives zero-shot transfer if the training
# distribution COVERS the target. Instead of hand-tuned strengths, we MEASURE
# the target's per-axis signal statistics (from unlabeled target recordings) and
# size the perturbations to envelope the target. E33: 0.824 zero-shot on real
# CinC (vs hand-tuned 0.783, clean 0.681), label preserved (QRS-band corr 0.966,
# R-peak match 0.976).
# ----------------------------------------------------------------------------

def signal_modality_stats(sig, fs=100.0):
    """Per-signal summary stats characterizing the recording modality."""
    from scipy import stats as _sstats
    x = np.asarray(sig, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    P = np.abs(np.fft.rfft(x)) ** 2
    Ptot = P.sum() + 1e-9
    def band(lo, hi): return P[(f >= lo) & (f < hi)].sum() / Ptot
    return {
        "kurtosis": float(_sstats.kurtosis(x)),
        "bw_energy": float(band(0, 1)),     # baseline wander <1 Hz
        "qrs_energy": float(band(5, 15)),   # QRS band
        "hf_energy": float(band(30, 50)),   # high-freq noise
        "mid_energy": float(band(1, 5)),
    }


def measure_distribution(sigs, fs=100.0, n=300):
    """Mean/std of modality stats over a set of signals (the 'target profile')."""
    ks = [signal_modality_stats(s, fs) for s in sigs[:n]]
    keys = ks[0].keys()
    return {k: (float(np.mean([d[k] for d in ks])),
                float(np.std([d[k] for d in ks]))) for k in keys}


def qrs_morphology_preserved(orig, aug, fs=100.0):
    """CORRECTED label-validity guard (E33): raw Pearson is fooled by baseline
    wander (a legitimate, label-irrelevant recording effect). Measure instead:
      - QRS-band correlation (>1 Hz high-pass removes baseline wander)
      - R-peak location preservation (rhythm = the diagnostic signal)
    Returns (qrs_corr, rpeak_match). Label is valid when both are high (>~0.9).
    """
    from scipy.signal import butter, filtfilt, find_peaks
    o = np.asarray(orig, float); a = np.asarray(aug, float)
    m = min(len(o), len(a)); o = o[:m]; a = a[:m]
    b, c = butter(2, 1.0 / (fs / 2), "high")
    oh = filtfilt(b, c, o); ah = filtfilt(b, c, a)
    qrs_corr = float(np.corrcoef(oh, ah)[0, 1])
    po, _ = find_peaks(oh, height=np.std(oh) * 1.5, distance=20)
    pa, _ = find_peaks(ah, height=np.std(ah) * 1.5, distance=20)
    if len(po) == 0:
        rpeak = float("nan")
    elif len(pa) == 0:
        rpeak = 0.0
    else:
        rpeak = float(np.mean([np.min(np.abs(pa - p)) <= 5 for p in po]))
    return qrs_corr, rpeak


class CalibratedAWAugmenter:
    """Target-coverage-calibrated domain randomization (E33 zero-shot best).

    Given clinical and target modality profiles (from measure_distribution), set
    perturbation ranges so the augmented distribution ENVELOPES the target on
    each axis (baseline wander, HF noise, kurtosis via bursts). `cover` > 1
    over-covers to be safe. Morphology-preserving (validate with
    qrs_morphology_preserved, NOT raw Pearson).
    """

    def __init__(self, clin_stats, tgt_stats, fs=100.0, siglen=1000, seed=None, cover=1.3):
        self.fs = fs; self.siglen = siglen; self.rng = np.random.default_rng(seed)
        self.cover = cover
        bw_gap = max(tgt_stats["bw_energy"][0] - clin_stats["bw_energy"][0], 0.0)
        hf_gap = max(tgt_stats["hf_energy"][0] - clin_stats["hf_energy"][0], 0.0)
        self.bw_amp = float(np.sqrt(bw_gap) * 4.0 * cover)
        self.hf_amp = float(np.sqrt(hf_gap) * 4.0 * cover)
        self.tgt_kurt = tgt_stats["kurtosis"][0]

    def generate(self, clinical_leadI, rng=None):
        rng = rng or self.rng
        x = np.asarray(clinical_leadI, dtype=np.float64)
        if x.shape[0] < self.siglen:
            x = np.concatenate([x, np.zeros(self.siglen - x.shape[0])])
        x = x[:self.siglen]; x = (x - x.mean()) / (x.std() + 1e-9)
        t = np.arange(self.siglen) / self.fs
        for _ in range(rng.integers(1, 4)):
            f = rng.uniform(0.1, 0.9); a = rng.uniform(0, self.bw_amp)
            x = x + a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        if self.hf_amp > 0:
            hf = rng.standard_normal(self.siglen)
            hf = np.diff(hf, prepend=hf[0])
            x = x + rng.uniform(0, self.hf_amp) * hf
        if rng.random() < 0.5:
            w = int(rng.uniform(0.03, 0.12) * self.siglen)
            i0 = rng.integers(0, self.siglen - w)
            x[i0:i0 + w] *= rng.uniform(1.2, 2.2)
        g = 1.0 + rng.uniform(0, 0.3) * np.sin(
            2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 6.28))
        x = x * g
        return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)

    def generate_batch(self, clinical_sigs, rng=None):
        return np.stack([self.generate(s, rng=rng) for s in clinical_sigs])


class ClosedLoopCalibrator:
    """Closed-loop target-calibrated augmenter (E40 — beats the clean coverage
    floor on REAL paired SJLIFE Apple data, unlike the open-loop
    CalibratedAWAugmenter which overshoots and collapses kurtosis).

    Key differences from CalibratedAWAugmenter:
      - wander model = <1 Hz COLOURED (1/f-ish) noise, NOT multi-tone sinusoids
        (sinusoids crush kurtosis; coloured wander preserves peak structure).
      - amplitude is BINARY-SEARCHED so the MEASURED output bw_energy matches the
        target, instead of set open-loop from a sqrt(gap) heuristic.
      - QRS band untouched -> morphology preserved (validate with
        qrs_morphology_preserved; E40 got QRS-corr 0.988, R-peak 0.963).

    Usage:
        clc = ClosedLoopCalibrator.fit(target_bw_energy, clinical_probe_sigs,
                                       fs=100.0, siglen=1000, seed=7, n_probe=40)
        aug = clc.generate(clinical_leadI)          # single
        batch = clc.generate_batch(clinical_sigs)   # many

    E40 result: distance-to-real-Apple 1.059 (clean) -> 0.659 (38% closer);
    bw_energy 0.217 vs target 0.230; kurtosis 7.50 preserved (open-loop -> 0.08).
    """

    def __init__(self, amp, fs=100.0, siglen=1000, seed=None):
        self.amp = float(amp)
        self.fs = fs
        self.siglen = siglen
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _low_wander(n, fs, rng):
        from scipy.signal import butter, filtfilt
        w = rng.standard_normal(n)
        b, a = butter(2, 0.9 / (fs / 2), "low")
        w = filtfilt(b, a, w)
        return w / (w.std() + 1e-9)

    def _add(self, x, a, rng):
        x = np.asarray(x, float)
        if x.shape[0] < self.siglen:
            x = np.concatenate([x, np.zeros(self.siglen - x.shape[0])])
        x = x[:self.siglen]
        x = (x - x.mean()) / (x.std() + 1e-9)
        return x + a * self._low_wander(self.siglen, self.fs, rng)

    @classmethod
    def fit(cls, tgt_bw, probe_sigs, fs=100.0, siglen=1000, seed=0, n_probe=40):
        """Binary-search wander amplitude so measured bw_energy == tgt_bw."""
        probe = list(probe_sigs)[:n_probe]
        obj = cls(0.0, fs=fs, siglen=siglen, seed=seed)

        def measured_bw(amp):
            vals = []
            for i, s in enumerate(probe):
                y = obj._add(s, amp, np.random.default_rng(int(amp * 1e6) % 99991 + i))
                vals.append(signal_modality_stats(y, fs)["bw_energy"])
            return float(np.mean(vals))

        lo, hi = 0.0, 3.0
        for _ in range(18):
            mid = (lo + hi) / 2
            if measured_bw(mid) < tgt_bw:
                lo = mid
            else:
                hi = mid
        obj.amp = (lo + hi) / 2
        return obj

    def generate(self, clinical_leadI, rng=None):
        rng = rng or self.rng
        y = self._add(clinical_leadI, self.amp, rng)
        t = np.arange(self.siglen) / self.fs
        g = 1.0 + 0.15 * np.sin(
            2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
        y = y * g
        return ((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)

    def generate_batch(self, clinical_sigs, rng=None):
        return np.stack([self.generate(s, rng=rng) for s in clinical_sigs])
