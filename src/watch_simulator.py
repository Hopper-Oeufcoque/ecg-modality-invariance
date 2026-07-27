"""Forward-physics Apple Watch ECG simulator (Method F10 / I7).

Transforms clinical 12-lead ECG into a realistic single-lead Apple-Watch-like
recording by applying, in order, the physical transfer function that separates
a clinical recording from a wrist-worn dry-electrode recording:

    12-lead  ->  Lead-I extraction          (lead-count axis)
             ->  Apple bandpass + resample   (sampling/bandwidth axis)
             ->  dry-electrode contact model (electrode axis)
             ->  motion + EMG + baseline     (noise axis)
             ->  ADC quantization            (sampling axis)

Each stage is independently toggleable so experiments can *decompose* the
modality gap by axis (lead-count vs bandwidth vs noise) — the keystone
validation this module exists to support.

Design notes / honesty flags
----------------------------
- Apple does not publish the exact ECG transfer function. The defaults below
  are drawn from the published analyses we cite in docs/method_taxonomy.md
  (Chan et al. 2020, arXiv:2012.00110 structured noise model; Apple Watch ECG
  app disclosures). Where a parameter is uncertain it is marked [CONFIGURABLE]
  and should be swept, not trusted as ground truth.
- This is a *forward* model (clinical -> watch), not a generative model. It
  cannot synthesize pathologies the source signal lacks; ground-truth labels
  pass through unchanged, which is exactly why it is useful for augmentation.
- Domain randomization: every stochastic stage samples fresh per call so a
  model trained on simulated watch sees the *distribution* of plausible watch
  conditions, not a single deterministic transform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class WatchSimConfig:
    """Parameters of the Apple-Watch-like transfer function.

    Defaults reflect the published Apple Watch ECG characteristics where known
    and are marked [CONFIGURABLE] where the exact value is unpublished.
    """

    # --- Target signal geometry ---
    fs_in: float = 100.0          # input sampling rate (Hz). PTB-XL 100Hz subset.
    fs_watch: float = 512.0       # Apple Watch output rate (Hz, published).

    # --- Bandwidth axis (Apple bandpass) ---
    # Apple states the ECG app records lead I; the passband is commonly cited
    # as ~0.3-40 Hz. [CONFIGURABLE: sweep 0.3-0.67 / 35-40]
    highpass: float = 0.3
    lowpass: float = 40.0
    filter_order: int = 4

    # --- Electrode axis (dry contact) ---
    # Dry stainless-steel wrist+finger electrodes have higher, variable
    # impedance -> a mild high-pass coupling effect + gain uncertainty.
    contact_gain_sigma: float = 0.15   # multiplicative gain jitter (log-normal)
    contact_hp: float = 0.05           # coupling high-pass corner (Hz), small

    # --- Noise axis ---
    baseline_wander_sigma: float = 0.15  # frac of signal std, spline drift
    motion_sigma: float = 0.10            # frac of signal std, low-freq colored
    emg_sigma: float = 0.05              # frac of signal std, high-freq
    motion_cutoff: float = 8.0           # Hz, motion artifact dominant band
    emg_band: tuple = (20.0, 150.0)      # Hz, EMG dominant band

    # --- Quantization axis ---
    adc_bits: int = 12                  # Apple ADC resolution (published-ish)
    adc_vref_uv: float = 4000.0         # full-scale ~4 mV input range

    # --- Toggles for axis decomposition ---
    apply_lead_reduction: bool = True   # lead-count axis
    apply_bandwidth: bool = True        # sampling/bandwidth axis
    apply_electrode: bool = True        # electrode physics axis
    apply_noise: bool = True           # noise axis
    apply_quantization: bool = True     # quantization axis

    # --- Reproducibility ---
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _design_bandpass(lo, hi, fs, order=4):
    """Zero-phase bandpass via cascaded second-order sections."""
    ny = fs / 2.0
    return butter(order, [lo / ny, hi / ny], btype="bandpass", output="sos")


def _butter_highpass(cutoff, fs, order=2):
    ny = fs / 2.0
    return butter(order, cutoff / ny, btype="highpass", output="sos")


def _butter_bandpass(lo, hi, fs, order=4):
    ny = fs / 2.0
    return butter(order, [lo / ny, hi / ny], btype="bandpass", output="sos")


def _colored_noise(n, fs, lo, hi, rng, sigma=1.0):
    """Band-limited colored noise in [lo,hi] Hz (for motion/EMG)."""
    # build in freq domain: white spectrum then band-mask
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    mask = (f >= lo) & (f <= hi)
    spec[~mask] = 0.0
    sig = np.fft.irfft(spec, n=n)
    sig = sig - sig.mean()
    if sig.std() > 1e-12:
        sig = sig / sig.std()
    return sig * sigma


def _baseline_wander(n, fs, rng, sigma):
    """Smooth cubic-spline-like drift from a few low-freq sinusoids."""
    t = np.arange(n) / fs
    out = np.zeros(n)
    for _ in range(4):
        f = rng.uniform(0.1, 0.6)          # respiratory + postural band
        a = rng.uniform(0.3, 1.0) * sigma
        phi = rng.uniform(0, 2 * np.pi)
        out += a * np.sin(2 * np.pi * f * t + phi)
    return out


# ---------------------------------------------------------------------------
# Core simulator
# ---------------------------------------------------------------------------

def simulate_watch(ecg: np.ndarray, fs_in: float, cfg: WatchSimConfig,
                   lead_names: Optional[list] = None,
                   rng: Optional[np.random.Generator] = None) -> dict:
    """Apply the forward watch transfer function to a clinical ECG.

    Parameters
    ----------
    ecg : ndarray, shape (n_samples, n_leads) or (n_samples,)
        Clinical ECG. If 2-D, lead index 0 is assumed to be Lead I (standard
        PTB-XL order: I, II, III, aVR, aVL, aVF, V1..V6). Override with
        lead_names / a lead index.
    fs_in : float
        Input sampling rate.
    cfg : WatchSimConfig
    lead_names : optional list of lead names; if present, 'I' is selected.
    rng : optional np.random.Generator.

    Returns
    -------
    dict with keys:
        'leadI_clean'    : Lead-I before any watch transform (lead-count axis only)
        'bandlimited'    : after Apple bandpass + resample (bandwidth axis)
        'electrode'      : after dry-contact model
        'noisy'          : after noise injection (noise axis)
        'watch'          : final quantized watch signal (all axes)
        'stages_applied' : ordered list of stage names actually applied
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    ecg = np.asarray(ecg, dtype=np.float64)
    if ecg.ndim == 1:
        ecg = ecg[:, None]
    n_samp, n_leads = ecg.shape

    # --- Stage 1: Lead reduction (lead-count axis) ---
    if cfg.apply_lead_reduction:
        if lead_names is not None and "I" in lead_names:
            idx = lead_names.index("I")
        else:
            idx = 0  # PTB-XL default: first lead is Lead I
        sig = ecg[:, idx].copy()
    else:
        # keep all leads (caller must handle multilead downstream)
        sig = ecg.copy()
    out = {"leadI_clean": sig.copy(), "stages_applied": []}

    # work at the input rate through the analog-domain stages, then resample
    fs = fs_in

    # --- Stage 2: Bandwidth (sampling/bandwidth axis) ---
    if cfg.apply_bandwidth:
        # apply Apple bandpass at input rate first (analog equivalent)
        sos = _design_bandpass(cfg.highpass, cfg.lowpass, fs, cfg.filter_order)
        if sig.ndim == 1:
            sig = sosfiltfilt(sos, sig)
        else:
            sig = sosfiltfilt(sos, sig, axis=0)
        out["bandlimited_pre"] = sig.copy()
        # resample to watch rate
        # use polyphase for rational rate conversion
        from math import gcd
        g = gcd(int(fs), int(cfg.fs_watch))
        up = int(cfg.fs_watch) // g
        down = int(fs) // g
        if sig.ndim == 1:
            sig = resample_poly(sig, up, down)
        else:
            sig = resample_poly(sig, up, down, axis=0)
        fs = cfg.fs_watch
        out["bandlimited"] = sig.copy()
        out["stages_applied"].append("bandwidth")
    else:
        out["bandlimited"] = sig.copy()

    # --- Stage 3: Electrode physics (dry contact) ---
    if cfg.apply_electrode:
        # variable contact gain (log-normal -> multiplicative)
        gain = np.exp(rng.normal(0, cfg.contact_gain_sigma))
        sig = sig * gain
        # coupling high-pass (mild)
        if cfg.contact_hp > 0:
            sos_hp = _butter_highpass(cfg.contact_hp, fs, order=2)
            if sig.ndim == 1:
                sig = sosfiltfilt(sos_hp, sig)
            else:
                sig = sosfiltfilt(sos_hp, sig, axis=0)
        out["electrode"] = sig.copy()
        out["stages_applied"].append("electrode")
    else:
        out["electrode"] = sig.copy()

    # normalize by current std for noise scaling (per-record norm, A3)
    ref_std = np.std(sig) if np.std(sig) > 1e-9 else 1.0

    # --- Stage 4: Noise (noise axis) ---
    if cfg.apply_noise:
        n = sig.shape[0]
        bw = _baseline_wander(n, fs, rng, cfg.baseline_wander_sigma * ref_std)
        motion = _colored_noise(n, fs, cfg.motion_cutoff / 2, cfg.motion_cutoff * 2,
                                rng, sigma=cfg.motion_sigma * ref_std)
        emg = _colored_noise(n, fs, cfg.emg_band[0], cfg.emg_band[1],
                             rng, sigma=cfg.emg_sigma * ref_std)
        if sig.ndim == 1:
            sig = sig + bw + motion + emg
        else:
            sig = sig + bw[:, None] + motion[:, None] + emg[:, None]
        out["noisy"] = sig.copy()
        out["stages_applied"].append("noise")
    else:
        out["noisy"] = sig.copy()

    # --- Stage 5: Quantization (ADC) ---
    if cfg.apply_quantization:
        # scale to ADC full-scale (in microvolts), quantize, scale back
        # assume input is in mV -> *1000 to uV
        scale = 1000.0
        levels = (1 << cfg.adc_bits)
        q = cfg.adc_vref_uv
        x_uv = sig * scale
        x_q = np.clip(np.round(x_uv / q * levels), -levels / 2, levels / 2 - 1)
        sig = (x_q * q / levels) / scale
        out["watch"] = sig.copy()
        out["stages_applied"].append("quantization")
    else:
        out["watch"] = sig.copy()

    return out


# Convenience: minimal config presets for ablation experiments
def cfg_leadcount_only(fs_in=100.0, **kw):
    """Isolate the lead-count axis: Lead-I only, no watch distortion."""
    return WatchSimConfig(fs_in=fs_in,
                          apply_bandwidth=False, apply_electrode=False,
                          apply_noise=False, apply_quantization=False, **kw)


def cfg_bandwidth_only(fs_in=100.0, **kw):
    """Lead-I + Apple bandpass/resample, no electrode/noise/quant."""
    return WatchSimConfig(fs_in=fs_in,
                          apply_electrode=False, apply_noise=False,
                          apply_quantization=False, **kw)


def cfg_full_watch(fs_in=100.0, **kw):
    """Full watch transfer function (default)."""
    return WatchSimConfig(fs_in=fs_in, **kw)
