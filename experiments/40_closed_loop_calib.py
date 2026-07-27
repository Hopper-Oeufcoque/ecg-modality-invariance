#!/usr/bin/env python3
"""
E40 — Closed-loop calibrated augmenter: hit the real Apple profile precisely.

E39 showed the open-loop CalibratedAWAugmenter overshoots baseline wander
(injects bw~0.53 when real Apple is 0.23) and collapses kurtosis. Fix: a
CLOSED-LOOP calibrator that tunes the injection amplitude until the MEASURED
output profile matches the target on the dominant axis (bw_energy), using a
1/f-shaped wander (not pure multi-tone sinusoids) to avoid crushing kurtosis.

Design:
  - wander model: low-freq (<1 Hz) coloured noise, amplitude `a`.
  - binary-search `a` so measured bw_energy(clinical + a*wander) == target bw.
  - keep QRS band intact -> morphology preserved (validate QRS-corr + R-peak).
  - compare closed-loop vs E39 variants on distance-to-real-Apple + morphology.

HONEST CAVEATS (unchanged from E39):
  - coverage is necessary-not-sufficient (fidelity != utility, E25b).
  - population distance, not beat-aligned; no labels for downstream AUROC.
"""
import os, sys, glob, json
import numpy as np
from scipy.signal import resample, butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/projects/ecg-modality-invariance/src"))
from aw_generator import (signal_modality_stats, measure_distribution,  # noqa
                          qrs_morphology_preserved, StochasticAWAugmenter,
                          CalibratedAWAugmenter)

ROOT = os.path.expanduser("~/projects/ecg-modality-invariance")
SJ = f"{ROOT}/data/sjlife"
OUT = f"{ROOT}/results/40_closed_loop_calib"
os.makedirs(OUT, exist_ok=True)
FS, SIGLEN = 100.0, 1000
AXES = ["kurtosis", "bw_energy", "qrs_energy", "hf_energy", "mid_energy"]

def to_common(x, fs_in, n=SIGLEN):
    y = resample(np.asarray(x, float), int(round(len(x) * FS / fs_in)))
    return (y[:n] if len(y) >= n else np.concatenate([y, np.zeros(n - len(y))]))

def clinical_leadI(path):
    c = np.squeeze(np.load(path).astype(np.float64))
    if c.ndim == 2 and c.shape[0] in (12, 15): return c[0]
    if c.ndim == 2 and c.shape[1] in (12, 15): return c[:, 0]
    return c.ravel()

def _low_wander(n, fs, rng):
    """Unit-power-ish <1 Hz coloured wander (1/f in-band)."""
    w = rng.standard_normal(n)
    b, a = butter(2, 0.9 / (fs / 2), "low")
    w = filtfilt(b, a, w)
    return w / (w.std() + 1e-9)

class ClosedLoopCalibrator:
    """Tune wander amplitude so MEASURED bw_energy hits target; 1/f wander keeps
    kurtosis intact. Morphology-preserving (QRS band untouched)."""
    def __init__(self, tgt_bw, fs=FS, siglen=SIGLEN, seed=0, n_probe=40):
        self.tgt_bw = tgt_bw; self.fs = fs; self.siglen = siglen
        self.rng = np.random.default_rng(seed)
        self.amp = self._calibrate(n_probe)

    def _add(self, x, a, rng):
        x = np.asarray(x, float)[:self.siglen]
        x = (x - x.mean()) / (x.std() + 1e-9)
        return x + a * _low_wander(self.siglen, self.fs, rng)

    def _measured_bw(self, amp, probe_sigs):
        vals = []
        for s in probe_sigs:
            y = self._add(s, amp, np.random.default_rng(int(amp*1e6) % 99991 + len(vals)))
            vals.append(signal_modality_stats(y, self.fs)["bw_energy"])
        return float(np.mean(vals))

    def _calibrate(self, n_probe):
        # need probe signals; injected lazily via set_probe before calibrate
        probe = getattr(self, "_probe", None)
        if probe is None:  # fallback: cannot calibrate without probe
            return 0.5
        lo, hi = 0.0, 3.0
        for _ in range(18):
            mid = (lo + hi) / 2
            if self._measured_bw(mid, probe) < self.tgt_bw: lo = mid
            else: hi = mid
        return (lo + hi) / 2

    @classmethod
    def fit(cls, tgt_bw, probe_sigs, **kw):
        obj = cls.__new__(cls)
        obj.tgt_bw = tgt_bw; obj.fs = kw.get("fs", FS); obj.siglen = kw.get("siglen", SIGLEN)
        obj.rng = np.random.default_rng(kw.get("seed", 0))
        obj._probe = probe_sigs[:kw.get("n_probe", 40)]
        obj.amp = obj._calibrate(kw.get("n_probe", 40))
        return obj

    def generate(self, x, rng=None):
        rng = rng or self.rng
        y = self._add(x, self.amp, rng)
        # mild global gain wander (keeps it wearable-like, morphology safe)
        t = np.arange(self.siglen) / self.fs
        g = 1.0 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 6.28))
        y = y * g
        return ((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)

def profile_dist(pm, tgt_mean, tgt_std):
    return float(np.sqrt(np.mean([((pm[k]-tgt_mean[k])/max(tgt_std[k],1e-6))**2 for k in AXES])))

def main():
    ap = sorted(glob.glob(f"{SJ}/apple/apple_ecg_*.npy"),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    apple_c, clin_c = [], []
    for a in ap:
        n = a.split("_")[-1].split(".")[0]
        cp = f"{SJ}/clinical/clinical_ecg_{n}.npy"
        if not os.path.exists(cp): continue
        apple_c.append(to_common(np.load(a).astype(np.float64).ravel(), 512.0))
        clin_c.append(to_common(clinical_leadI(cp), 500.0))
    print(f"paired: {len(apple_c)}")

    apple_prof = measure_distribution(apple_c, FS, n=len(apple_c))
    clin_prof  = measure_distribution(clin_c,  FS, n=len(clin_c))
    tgt_mean = {k: apple_prof[k][0] for k in AXES}
    tgt_std  = {k: apple_prof[k][1] for k in AXES}
    print("real Apple target bw:", round(tgt_mean["bw_energy"], 3), "kurt:", round(tgt_mean["kurtosis"],2))

    # closed-loop fit on a probe subset, apply to all
    clc = ClosedLoopCalibrator.fit(tgt_mean["bw_energy"], clin_c, fs=FS, siglen=SIGLEN, seed=7, n_probe=40)
    print(f"closed-loop calibrated wander amp = {clc.amp:.3f}")

    variants = {
        "clean": clin_c,
        "light_DR": [StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=0.5, seed=1).generate(x) for x in clin_c],
        "open_loop_calDR": [CalibratedAWAugmenter(clin_prof, apple_prof, fs=FS, siglen=SIGLEN, seed=3, cover=1.3).generate(x) for x in clin_c],
        "closed_loop": [clc.generate(x) for x in clin_c],
    }

    print("\n===== distance to REAL Apple + morphology =====")
    rows = {}
    for name, sigs in variants.items():
        prof = measure_distribution(sigs, FS, n=len(sigs))
        pm = {k: prof[k][0] for k in AXES}
        d = profile_dist(pm, tgt_mean, tgt_std)
        if name == "clean":
            qrs, rpk = 1.0, 1.0
        else:
            qs, rs = [], []
            for i in range(0, len(sigs), 5):
                q, r = qrs_morphology_preserved(clin_c[i], sigs[i], FS)
                if not np.isnan(q): qs.append(q)
                if not np.isnan(r): rs.append(r)
            qrs, rpk = float(np.mean(qs)), float(np.mean(rs))
        rows[name] = dict(dist=d, qrs=qrs, rpeak=rpk, profile=pm)
        print(f"  {name:16s} dist={d:6.3f}  QRS={qrs:.3f} Rpk={rpk:.3f}  "
              f"bw={pm['bw_energy']:.3f} kurt={pm['kurtosis']:.2f} qrs_e={pm['qrs_energy']:.3f}")

    valid = {k: v for k, v in rows.items() if k != "clean" and v["qrs"] >= 0.85 and v["rpeak"] >= 0.90}
    best = min(valid, key=lambda k: valid[k]["dist"]) if valid else None
    improved = best and rows[best]["dist"] < rows["clean"]["dist"]
    print(f"\n  clean floor dist : {rows['clean']['dist']:.3f}")
    print(f"  BEST label-valid : {best} dist={rows[best]['dist']:.3f}" if best else "  none valid")
    print(f"  >>> closes gap vs clean? {'YES' if improved else 'NO'} <<<")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = list(rows.keys()); dists = [rows[n]["dist"] for n in names]
    ax[0].bar(names, dists, color=["#999","#6a6","#c74","#d4a017"])
    ax[0].axhline(rows["clean"]["dist"], color="k", ls=":", label="clean floor")
    for i,d in enumerate(dists): ax[0].text(i,d+0.01,f"{d:.2f}",ha="center",fontsize=9)
    ax[0].set_ylabel("distance to REAL Apple"); ax[0].set_title("Closed-loop vs open-loop coverage"); ax[0].tick_params(axis="x",rotation=15)
    idx=np.arange(len(AXES)); w=0.19
    ax[1].bar(idx-1.5*w,[tgt_mean[k] for k in AXES],w,label="REAL Apple",color="#d4a017",edgecolor="k")
    ax[1].bar(idx-0.5*w,[rows["clean"]["profile"][k] for k in AXES],w,label="clean",color="#999")
    ax[1].bar(idx+0.5*w,[rows["open_loop_calDR"]["profile"][k] for k in AXES],w,label="open-loop",color="#c74")
    ax[1].bar(idx+1.5*w,[rows["closed_loop"]["profile"][k] for k in AXES],w,label="closed-loop",color="#6a6")
    ax[1].set_xticks(idx); ax[1].set_xticklabels(AXES,rotation=30,ha="right"); ax[1].legend(fontsize=8); ax[1].set_title("Profile match")
    fig.tight_layout(); fig.savefig(f"{OUT}/closed_loop.png", dpi=110)
    print(f"\nSaved -> {OUT}/closed_loop.png")

    with open(f"{OUT}/metrics.json","w") as f:
        json.dump({"n":len(apple_c),"calibrated_wander_amp":clc.amp,
                   "real_apple_target":tgt_mean,
                   "variants":{k:{"dist":v["dist"],"qrs":v["qrs"],"rpeak":v["rpeak"],"profile":v["profile"]} for k,v in rows.items()},
                   "best_label_valid":best,"closes_gap_vs_clean":bool(improved),
                   "caveats":["coverage necessary-not-sufficient (E25b)","population distance not beat-aligned","no labels for AUROC"]},f,indent=2)
    print(f"Saved -> {OUT}/metrics.json\nDONE.")

if __name__ == "__main__":
    main()
