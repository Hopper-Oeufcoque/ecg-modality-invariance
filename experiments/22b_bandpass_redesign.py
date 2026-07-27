"""Experiment 22b — Bandpass redesign to recover kurtosis (the kurtosis-killer).

DIRECT FOLLOW-UP TO E22: E22 found the simulator's kurtosis gap is FILTER-bound,
not noise-bound. Across a 20× noise reduction (m=1.0→0.05), kurtosis stayed
stuck at ~5 vs real CinC 17.71. The bandpass (0.3-40 Hz, order 4, zero-phase
Butterworth) + electrode high-pass coupling destroy QRS peakedness by removing
the high-frequency QRS components that create sharp morphology.

E22b is diagnostic: sweep bandpass configurations to find what recovers kurtosis
while keeping the bandwidth axis minor for AUROC (E1 found it IS minor — so a
gentler filter that's more realistic should be strictly better).

Configurations swept (each generates sim-watch from 256 clinical records, computes
distribution stats vs real CinC, AND trains a single-lead+sim model at best config):
  B0 default        — bandpass 0.3-40 Hz, order 4, zero-phase (current)
  B1 no-bandwidth    — skip the bandpass entirely (lead-reduction + noise only)
  B2 gentle-order    — order 2 (less steep rolloff, less ringing)
  B3 wider-passband  — 0.3-100 Hz (keep more high-freq QRS content; 100= Nyquist)
  B4 no-electrode-HP — skip the contact high-pass coupling (0.05 Hz HP)
  B5 causal-filter   — lfilter (causal) vs sosfiltfilt (zero-phase): phase distortion?
  B6 minimal-sim     — B1 + B4 combined (lead-reduction + light noise only)

For each: measure kurtosis, sample_entropy, baseline_wander, mean dist to real.
Then train single-lead+sim at the best-kurtosis config and the default, compare L4.

KEY QUESTION: can a bandpass redesign recover kurtosis toward the real 17.7 target,
and does a more realistic sim (higher kurtosis) improve the single-lead+sim model?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

_spec6 = importlib.util.spec_from_file_location(
    "e6", Path(__file__).resolve().parents[1] / "experiments" / "06_sim_validation.py")
e6 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(e6)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "22b_bandpass_redesign"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# Bandpass configurations (deterministic for distribution comparison)
# ---------------------------------------------------------------------------

CONFIGS = {
    "B0_default":        _cfg(highpass=0.3, lowpass=40.0, filter_order=4, seed=0),
    "B1_no_bandwidth":   _cfg(apply_bandwidth=False, seed=0),
    "B2_gentle_order2":  _cfg(highpass=0.3, lowpass=40.0, filter_order=2, seed=0),
    "B3_wider_passband": _cfg(highpass=0.3, lowpass=49.0, filter_order=4, seed=0),  # Nyquist=50
    "B4_no_electrodeHP": _cfg(contact_hp=0.0, seed=0),
    "B5_minimal_sim":    _cfg(apply_bandwidth=False, contact_hp=0.0, seed=0),
    # B6: very light noise too (combine with B5)
    "B6_light_noise":    _cfg(apply_bandwidth=False, contact_hp=0.0,
                              baseline_wander_sigma=0.05, motion_sigma=0.03,
                              emg_sigma=0.02, seed=0),
}


def distribution_stats(clin, cfg):
    """Generate sim-watch from clinical with given cfg, compute distribution stats."""
    stats = []
    for rec in clin:
        x = rec["ecg"][:1000]
        if x.shape[0] < 1000:
            x = np.concatenate([x, np.zeros((1000 - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
        stats.append(e6.all_stats(out["watch"], FS))
    means = {k: float(np.mean([s[k] for s in stats if not np.isnan(s.get(k, float("nan")))]))
             for k in ["baseline_wander", "sample_entropy", "kurtosis", "dfa_alpha"]}
    return stats, means


# ---------------------------------------------------------------------------
# Single-lead+sim training (reuse E17/E22 recipe)
# ---------------------------------------------------------------------------

class SimSingleLeadDataset(torch.utils.data.Dataset):
    def __init__(self, records, labels_idx, cfg):
        self.records = records; self.labels_idx = labels_idx; self.cfg = cfg
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, self.cfg, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.labels_idx[i]))


def train_singlelead_sim(records, labels_idx, cfg, epochs=20, lr=1e-3, batch_size=64, tag="sl"):
    ds = SimSingleLeadDataset(records, labels_idx, cfg)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate_singlelead(model, records, labels_idx, eval_cfg, batch_size=64):
    model.eval()
    all_logits = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, eval_cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]; leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
            batch.append(leadI[:, None].astype(np.float32))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        all_logits.append(model(xb).numpy())
        all_y.append(np.array(labels_idx[i:i+batch_size]))
    return np.concatenate(all_logits), np.concatenate(all_y)


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    from sklearn.metrics import roc_auc_score
    aucs = []
    for c in range(n_classes):
        yb = (y == c).astype(int)
        if yb.sum() == 0 or yb.sum() == len(yb): aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, logits[:, c])))
        except: aucs.append(float("nan"))
    return aucs

def macro_auroc(logits, y):
    aucs = [a for a in per_class_auroc(logits, y) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading PTB-XL clinical (max_per_class=200 for calibration) ...", flush=True)
    splits = load_all(max_per_class=200)
    clin = splits["test"]
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    for r in splits["train"]: r["label_idx"] = c2i[r["label"]]
    for r in splits["test"]: r["label_idx"] = c2i[r["label"]]
    print(f"  clinical (calib): {len(clin)}", flush=True)

    print("Loading real CinC 2017 (n=300) ...", flush=True)
    real = e6.load_cinc2017(n=300)
    real_stats = [e6.all_stats(s) for s, _ in real]
    real_means = {k: float(np.mean([s[k] for s in real_stats if not np.isnan(s.get(k, float("nan")))]))
                  for k in ["baseline_wander", "sample_entropy", "kurtosis", "dfa_alpha"]}
    print(f"  real CinC: kurt={real_means['kurtosis']:.2f} entropy={real_means['sample_entropy']:.3f}", flush=True)

    print("\n=== Part A: bandpass config sweep (distribution stats) ===", flush=True)
    sweep = {}
    for name, cfg in CONFIGS.items():
        stats, means = distribution_stats(clin, cfg)
        dist = e6.distribution_distance(stats, real_stats)
        mean_dist = float(np.mean(list(dist.values())))
        sweep[name] = {"means": means, "dist_to_real": dist, "mean_dist_to_real": mean_dist}
        print(f"  {name}: kurt={means['kurtosis']:.2f} entropy={means['sample_entropy']:.3f} "
              f"bw={means['baseline_wander']:.3f} | dist={mean_dist:.3f}", flush=True)

    # best config = highest kurtosis (closest to real 17.7), tiebreak by mean dist
    best_name = max(sweep, key=lambda n: (sweep[n]["means"]["kurtosis"], -sweep[n]["mean_dist_to_real"]))
    print(f"\n  -> best (highest kurtosis): {best_name}", flush=True)

    print("\n=== Part B: single-lead+sim rerun (B0 default vs best) ===", flush=True)
    tr, te = splits["train"], splits["test"]
    ytr = np.array([r["label_idx"] for r in tr])
    yte = np.array([r["label_idx"] for r in te])
    cfg_L1 = _cfg(apply_bandwidth=False, apply_electrode=False, apply_noise=False,
                  apply_quantization=False, seed=0)
    train_results = {}
    for name in ["B0_default", best_name]:
        # stochastic training cfg from the deterministic sweep cfg (seed=None for aug)
        cfg_train = WatchSimConfig(**{**CONFIGS[name].__dict__, "seed": None})
        cfg_L4 = WatchSimConfig(**{**CONFIGS[name].__dict__, "seed": 0})
        print(f"\n--- Training {name} ---", flush=True)
        model = train_singlelead_sim(tr, ytr, cfg_train, tag=name)
        logits, y = evaluate_singlelead(model, te, yte, cfg_L1)
        l1 = macro_auroc(logits, y)
        logits, y = evaluate_singlelead(model, te, yte, cfg_L4)
        l4 = macro_auroc(logits, y)
        train_results[name] = {"L1": l1, "L4": l4}
        print(f"  {name} @ L1={l1:.4f} L4={l4:.4f}", flush=True)

    metrics = {
        "fs": FS, "n_train": len(tr), "n_test": len(te), "classes": SUPERCLASSES,
        "real_cinc_means": real_means, "sweep": sweep, "best_config": best_name,
        "singlelead_sim_rerun": train_results,
        "e17_reference_L4": 0.742, "e22_default_L4": 0.711,
        "honesty": ["single seed", "CinC handheld (cleaner than wrist dry-electrode)",
                    "bandpass redesign trades realism for Apple-fidelity — a gentler filter is less Apple-accurate"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(sweep, train_results, real_means, best_name)
    print("\nDONE.", flush=True)


def _plot(sweep, train_results, real_means, best_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(sweep.keys())
    kurts = [sweep[n]["means"]["kurtosis"] for n in names]
    dists = [sweep[n]["mean_dist_to_real"] for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bars = ax.bar(range(len(names)), kurts, color="#4C72B0")
    ax.axhline(real_means["kurtosis"], color="#C44E52", ls="--", label="real CinC kurtosis")
    ax.axhline(15, color="green", ls=":", label="target >=15")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=25)
    ax.set_ylabel("kurtosis"); ax.set_title("Kurtosis by bandpass config")
    ax.legend(fontsize=7)
    for b, v in zip(bars, kurts): ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.1f}", ha="center", fontsize=7)
    ax = axes[1]
    bars = ax.bar(range(len(names)), dists, color="#55A868")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7, rotation=25)
    ax.set_ylabel("mean dist to real"); ax.set_title("Distribution distance to real CinC")
    for b, v in zip(bars, dists): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=7)
    plt.suptitle("E22b: bandpass redesign — recovering kurtosis", fontsize=11)
    plt.tight_layout(); plt.savefig(RESULTS / "bandpass_sweep.png", dpi=130); plt.close()
    # rerun comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tn = list(train_results.keys())
    l4 = [train_results[t]["L4"] for t in tn]
    l1 = [train_results[t]["L1"] for t in tn]
    x = np.arange(len(tn)); w = 0.35
    ax.bar(x-w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    ax.bar(x+w/2, l4, w, label="L4 full sim", color="#C44E52")
    ax.axhline(0.742, color="green", ls="--", lw=1, label="E17 ref 0.742")
    ax.set_xticks(x); ax.set_xticklabels(tn, fontsize=8); ax.set_ylabel("macro AUROC")
    ax.set_ylim(0.5, 0.82); ax.set_title("E22b: single-lead+sim under bandpass redesign")
    ax.legend(fontsize=8, loc="lower right")
    for i, v in enumerate(l4): ax.text(i+w/2, v+0.005, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(RESULTS / "rerun.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
