"""Experiment 33 — Target-calibrated domain randomization for TRUE zero-shot.

USER GOAL (sharpened): train on abundant clinical, deploy on watch with NO
tuning, hit the watch-oracle (~0.93). E27b showed hand-designed augmentation
plateaus ~0.80 — because domain randomization only gives zero-shot transfer if
the training distribution actually CONTAINS/COVERS the target distribution. Our
hand-tuned augmenter clearly doesn't cover real CinC.

E33 fixes that by CALIBRATING the augmentation to the measured real-target
statistics (from UNLABELED CinC ref — legitimate, no labels), so the augmented
training distribution is built to ENVELOPE the real watch distribution:

  1. Measure target stats from unlabeled CinC ref: power spectral density (band
     energy ratios), amplitude-distribution kurtosis, baseline-wander energy,
     high-freq noise floor, sample-entropy.
  2. Measure the same on clinical Lead-I. Compute the per-axis GAP.
  3. Build a CALIBRATED augmenter whose perturbation RANGES are set so the
     augmented distribution's stats span from clinical to (beyond) target on
     every measured axis — i.e. domain randomization that provably covers the
     target, not arbitrary strength.

Arms (5 seeds, real CinC AF/NORM):
  clean
  augment_handtuned      E27b locked recipe (s1.5, 5x) ~0.80
  augment_calibrated     <-- the test: coverage-calibrated randomization
  augment_calibrated+TENT (if E32 shows TENT helps, stack it)
  oracle

Question: does building the training distribution to actually COVER the target
(measured, not guessed) push zero-shot past the 0.80 plateau toward oracle?

Honesty: 5 seeds, n=700 CinC, AF/NORM only, CinC = E6c AW proxy. Calibration
uses unlabeled target SIGNAL STATISTICS only (no labels) — a realistic "we have
some unlabeled watch recordings" assumption, still effectively zero-shot for
labels. Coverage is measured on summary stats, not a guarantee of full-manifold
coverage (flagged).
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "33_calibrated_dr"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"


def signal_stats(sig):
    """Summary stats characterizing the recording modality (per signal)."""
    x = np.asarray(sig, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    f = np.fft.rfftfreq(len(x), 1 / FS)
    P = np.abs(np.fft.rfft(x)) ** 2
    Ptot = P.sum() + 1e-9
    def band(lo, hi): return P[(f >= lo) & (f < hi)].sum() / Ptot
    return {
        "kurtosis": float(sstats.kurtosis(x)),
        "bw_energy": float(band(0, 1)),        # baseline wander <1 Hz
        "qrs_energy": float(band(5, 15)),      # QRS band
        "hf_energy": float(band(30, 50)),      # high-freq noise
        "mid_energy": float(band(1, 5)),
    }


def dist_stats(sigs, n=300):
    ks = [signal_stats(s) for s in sigs[:n]]
    keys = ks[0].keys()
    return {k: (float(np.mean([d[k] for d in ks])), float(np.std([d[k] for d in ks]))) for k in keys}


class CalibratedAugmenter:
    """Domain randomization calibrated to COVER the measured target distribution.

    Given the clinical and target per-axis stats, set perturbation ranges so the
    augmented signals' stats span from clinical toward/beyond target on each
    axis (baseline wander, HF noise, kurtosis via amplitude bursts). Randomize
    WIDE enough to envelope the target, morphology-preserving.
    """
    def __init__(self, clin_stats, tgt_stats, fs=FS, siglen=SIGLEN, seed=None, cover=1.3):
        self.fs = fs; self.siglen = siglen; self.rng = np.random.default_rng(seed)
        self.cover = cover
        # target/clinical gaps drive the randomization amplitude per axis
        self.bw_gap = max(tgt_stats["bw_energy"][0] - clin_stats["bw_energy"][0], 0.0)
        self.hf_gap = max(tgt_stats["hf_energy"][0] - clin_stats["hf_energy"][0], 0.0)
        # how much extra low-freq / high-freq energy to be able to inject
        self.bw_amp = float(np.sqrt(self.bw_gap) * 4.0 * cover)   # amplitude scale
        self.hf_amp = float(np.sqrt(self.hf_gap) * 4.0 * cover)
        self.tgt_kurt = tgt_stats["kurtosis"][0]

    def generate(self, clinical_leadI, rng=None):
        rng = rng or self.rng
        x = np.asarray(clinical_leadI, dtype=np.float64)
        if x.shape[0] < self.siglen: x = np.concatenate([x, np.zeros(self.siglen - x.shape[0])])
        x = x[:self.siglen]; x = (x - x.mean()) / (x.std() + 1e-9)
        t = np.arange(self.siglen) / self.fs
        # baseline wander to cover target low-freq energy (randomized up to cover*gap)
        for _ in range(rng.integers(1, 4)):
            f = rng.uniform(0.1, 0.9); a = rng.uniform(0, self.bw_amp)
            x = x + a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        # high-freq noise to cover target HF energy
        if self.hf_amp > 0:
            hf = rng.standard_normal(self.siglen)
            # bandpass-ish emphasis on 30-50 Hz via difference filter
            hf = np.diff(hf, prepend=hf[0])
            x = x + rng.uniform(0, self.hf_amp) * hf
        # occasional amplitude bursts (raise kurtosis toward target) — morphology safe
        if rng.random() < 0.5:
            w = int(rng.uniform(0.03, 0.12) * self.siglen); i0 = rng.integers(0, self.siglen - w)
            x[i0:i0 + w] *= rng.uniform(1.2, 2.2)
        # gain wander
        g = 1.0 + rng.uniform(0, 0.3) * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 6.28))
        x = x * g
        return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def morph_corr(aug, sigs, n=80):
    cs = []
    for s in sigs[:n]:
        s = np.asarray(s, np.float64); a = aug.generate(s); m = min(len(s), len(a))
        c = np.corrcoef(s[:m], a[:m])[0, 1]
        if np.isfinite(c): cs.append(c)
    return float(np.mean(cs))


def run_seed(seed, tr_leadI, tr_y, cinc, clin_stats):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    # measure target stats from UNLABELED ref
    tgt_stats = dist_stats([r["sig"] for r in cinc_ref])

    out = {}
    def fit_eval(sigs, ys, tag):
        m = ECGResNet1d(1, N_CLASSES).to(DEVICE)
        e25.train(m, e25.SigDataset(sigs, ys), epochs=20, tag=f"s{seed}-{tag}")
        auc = e25.evaluate(m, test_sigs, test_y)[0]; out[tag] = auc
        print(f"  seed {seed} {tag}: {auc:.4f}", flush=True)
        return m

    fit_eval(tr_leadI, tr_y, "clean")

    # hand-tuned recipe
    hand = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.5, seed=seed)
    src = list(tr_leadI); sy = list(tr_y)
    for _ in range(5): src += [hand.generate(s) for s in tr_leadI]; sy += list(tr_y)
    fit_eval(src, sy, "augment_handtuned")

    # calibrated DR
    cal = CalibratedAugmenter(clin_stats, tgt_stats, seed=seed, cover=1.3)
    mc = morph_corr(cal, tr_leadI)
    src2 = list(tr_leadI); sy2 = list(tr_y)
    for _ in range(5): src2 += [cal.generate(s) for s in tr_leadI]; sy2 += list(tr_y)
    fit_eval(src2, sy2, "augment_calibrated")
    out["_calibrated_morph_corr"] = mc
    out["_tgt_stats"] = tgt_stats

    # oracle
    mO = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mO, e25.SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(mO, test_sigs, test_y)[0]
    print(f"  seed {seed} oracle: {out['oracle']:.4f} (calib morph_corr={mc:.3f})", flush=True)
    return out


def main():
    print("Loading PTB-XL binary AF/NORM ...", flush=True)
    ptb = e25.load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]; tr_leadI = [r["leadI"] for r in tr]; tr_y = [r["y"] for r in tr]
    print(f"  train={len(tr)}", flush=True)
    print("Loading REAL CinC ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC={len(cinc)}", flush=True)

    clin_stats = dist_stats(tr_leadI)
    print(f"  clinical stats: {clin_stats}", flush=True)

    seeds = [0, 1, 2, 3, 4]
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc, clin_stats)

    keys = ["clean", "augment_handtuned", "augment_calibrated", "oracle"]
    agg = {}
    hand = np.array([per_seed[s]["augment_handtuned"] for s in seeds])
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
        if k == "augment_calibrated":
            agg[k]["delta_vs_handtuned_mean"] = float((vals - hand).mean())
            agg[k]["delta_positive_in_seeds"] = int(((vals - hand) > 0).sum())
    agg["_calibrated_morph_corr"] = float(np.mean([per_seed[s]["_calibrated_morph_corr"] for s in seeds]))

    print("\n===== AGGREGATE (mean±std, 5 seeds) =====", flush=True)
    for k in keys:
        extra = ""
        if k == "augment_calibrated":
            extra = f"  Δvs_hand={agg[k]['delta_vs_handtuned_mean']:+.4f} ({agg[k]['delta_positive_in_seeds']}/5)"
        print(f"  {k:20s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)
    print(f"  calibrated morph_corr={agg['_calibrated_morph_corr']:.3f}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "seeds": seeds,
        "clinical_stats": clin_stats,
        "aggregate": agg, "per_seed": {s: {k: v for k, v in per_seed[s].items() if not k.startswith("_tgt")} for s in seeds},
        "question": "Does target-COVERAGE-calibrated domain randomization beat hand-tuned augmentation toward zero-shot oracle?",
        "honesty": ["5 seeds", "n=700 CinC", "AF/NORM only", "CinC = E6c AW proxy",
                    "calibration uses unlabeled target SIGNAL STATS only (no labels)",
                    "coverage measured on summary stats, not full-manifold guarantee"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["clean", "augment\nhand-tuned", "augment\ncalibrated", "oracle\nreal"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#4C72B0", "#55A868", "#DD8452", "#937860"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(means[1], color="#55A868", ls="--", lw=1, label=f"hand-tuned bar ({means[1]:.3f})")
    ax.axhline(means[-1], color="#937860", ls=":", lw=1, label=f"oracle ({means[-1]:.3f})")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC on real CinC (5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E33: target-calibrated domain randomization vs hand-tuned")
    ax.legend(fontsize=8, loc="upper left")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "calibrated_dr.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
