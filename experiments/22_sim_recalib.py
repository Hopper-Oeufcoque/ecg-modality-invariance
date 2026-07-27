"""Experiment 22 — Simulator recalibration against real single-lead + E17 rerun.

DIRECT RESPONSE TO E6: E6 found the forward-physics simulator OVER-DEGRADES
relative to real single-lead (CinC 2017). sim_vs_real distance (1.077) >
real_vs_clinical (0.717). The added noise flattens QRS peakedness: sim kurtosis
4.77 vs real 17.71 (3.7x too low), sample_entropy 0.818 vs 0.282 (2.9x too high),
baseline_wander 0.435 vs 0.190 (2.3x too high).

E22 asks two questions:
  (A) CALIBRATION: can the simulator's noise magnitudes be tuned so its output
      distribution matches real single-lead (kurtosis >= 15, entropy <= 0.4)?
      Sweep a global noise multiplier m in {1.0, 0.5, 0.25, 0.1, 0.05} applied
      to baseline_wander / motion / EMG sigma; find the m minimizing the
      distribution distance to real CinC.
  (B) EDGE ROBUSTNESS: does the E17 winner (single-lead trained on sim) survive
      recalibration? Train single-lead+sim at the best-calibrated m and at the
      default m=1.0; compare L4 AUROC. If the edge GROWS at calibration → the
      over-degradation was hurting the model. If it VANISHES → the E17 edge was
      an artifact of the favorable sim->sim train/test match (an honesty flag).

Honesty: CinC 2017 is handheld lead-I (cleaner than wrist dry-electrode), so
matching it may UNDER-model the true Apple Watch noise floor. The kurtosis-
preserving objective is the key insight regardless: the sim must keep QRS
peakiness, which additive broadband noise destroys.
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

# reuse E6 stats + CinC loader
_spec6 = importlib.util.spec_from_file_location(
    "e6", Path(__file__).resolve().parents[1] / "experiments" / "06_sim_validation.py")
e6 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(e6)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "22_sim_recalib"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(noise_mult=1.0, **kw):
    kw.setdefault("fs_watch", FS)
    kw.setdefault("baseline_wander_sigma", 0.15 * noise_mult)
    kw.setdefault("motion_sigma", 0.10 * noise_mult)
    kw.setdefault("emg_sigma", 0.05 * noise_mult)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# Part A: calibration sweep
# ---------------------------------------------------------------------------

def calibration_sweep(clin, real_stats):
    """For each noise multiplier, generate sim-watch and compute distribution stats."""
    results = {}
    for m in [1.0, 0.5, 0.25, 0.1, 0.05]:
        cfg = _cfg(noise_mult=m, seed=0)  # deterministic for fair comparison
        sim_stats = []
        for rec in clin:
            x = rec["ecg"][:1000]
            if x.shape[0] < 1000:
                x = np.concatenate([x, np.zeros((1000 - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            sim_stats.append(e6.all_stats(out["watch"], FS))
        # distance to real
        dist = e6.distribution_distance(sim_stats, real_stats)
        means = {k: float(np.mean([s[k] for s in sim_stats if not np.isnan(s.get(k, float("nan")))]))
                 for k in ["baseline_wander", "sample_entropy", "kurtosis", "dfa_alpha"]}
        mean_dist = float(np.mean(list(dist.values())))
        results[f"m={m}"] = {"noise_mult": m, "means": means, "dist_to_real": dist,
                             "mean_dist_to_real": mean_dist}
        print(f"  m={m}: kurt={means['kurtosis']:.2f} entropy={means['sample_entropy']:.3f} "
              f"bw={means['baseline_wander']:.3f} | dist_to_real={mean_dist:.3f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Part B: E17 single-lead+sim training at a given config
# ---------------------------------------------------------------------------

class SimSingleLeadDataset(torch.utils.data.Dataset):
    def __init__(self, records, labels_idx, cfg):
        self.records = records; self.labels_idx = labels_idx; self.cfg = cfg
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:1000] if T >= 1000 else np.concatenate([x, np.zeros((1000-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, self.cfg, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        m1 = leadI.mean(); s1 = leadI.std() + 1e-6
        leadI = (leadI - m1) / s1
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.labels_idx[i]))


def train_singlelead_sim(records, labels_idx, cfg, epochs=20, tag="sl-sim"):
    ds = SimSingleLeadDataset(records, labels_idx, cfg)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0)
    model = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
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
def evaluate_singlelead(model, records, labels_idx, eval_cfg, stage, batch_size=64):
    """stage L1 clean Lead-I / L4 full watch (eval_cfg sets the toggles)."""
    model.eval()
    all_logits = []; all_y = []
    for i in range(0, len(records), batch_size):
        batch = []
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:1000]
            if x.shape[0] < 1000:
                x = np.concatenate([x, np.zeros((1000 - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, eval_cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]
            leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
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
        except Exception: aucs.append(float("nan"))
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
    print(f"  real means: kurt={real_means['kurtosis']:.2f} entropy={real_means['sample_entropy']:.3f} "
          f"bw={real_means['baseline_wander']:.3f}", flush=True)

    print("\n=== Part A: noise-multiplier calibration sweep ===", flush=True)
    calib = calibration_sweep(clin, real_stats)
    # pick best (min mean dist to real)
    best_m = min(calib.values(), key=lambda r: r["mean_dist_to_real"])["noise_mult"]
    print(f"  -> best noise_mult = {best_m}", flush=True)

    print("\n=== Part B: E17 single-lead+sim rerun (default m=1.0 vs best m) ===", flush=True)
    tr, te = splits["train"], splits["test"]
    ytr = np.array([r["label_idx"] for r in tr])
    yte = np.array([r["label_idx"] for r in te])
    # Eval configs: train/test must use the SAME sim config family.
    # L4 eval = full sim at that multiplier; L1 = clean Lead-I (no sim axes).
    cfg_L1 = _cfg(noise_mult=1.0, apply_bandwidth=False, apply_electrode=False,
                  apply_noise=False, apply_quantization=False, seed=0)
    train_results = {}
    for m_train, tag in [(1.0, "default_m1.0"), (best_m, f"calib_m{best_m}")]:
        cfg_train = _cfg(noise_mult=m_train, seed=None)
        cfg_L4 = _cfg(noise_mult=m_train, seed=0)
        print(f"\n--- Training {tag} ---", flush=True)
        model = train_singlelead_sim(tr, ytr, cfg_train, epochs=20, tag=tag)
        r = {}
        for stage, ec in [("L1", cfg_L1), ("L4", cfg_L4)]:
            logits, y = evaluate_singlelead(model, te, yte, ec, stage)
            ma = macro_auroc(logits, y); pc = per_class_auroc(logits, y)
            r[stage] = {"macro_auroc": ma, "per_class": pc}
            print(f"  {tag} @ {stage}: macro={ma:.4f} per={[round(a,3) for a in pc]}", flush=True)
        train_results[tag] = {"noise_mult": m_train, **r}

    metrics = {
        "fs": FS, "n_train": len(tr), "n_test": len(te),
        "classes": SUPERCLASSES,
        "real_cinc_means": real_means,
        "calibration_sweep": calib,
        "best_noise_mult": best_m,
        "singlelead_sim_rerun": train_results,
        "e17_reference_L4": 0.742,
        "honesty": ["single seed", "CinC is handheld (cleaner than wrist dry-electrode) — matching it may under-model true Apple Watch noise",
                    "calibration matches distribution stats, not task transfer (E6b needed for task-level)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(calib, train_results, real_means, best_m)
    print("\nDONE.", flush=True)


def _plot(calib, train_results, real_means, best_m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [r["noise_mult"] for r in calib.values()]
    kurts = [r["means"]["kurtosis"] for r in calib.values()]
    ents = [r["means"]["sample_entropy"] for r in calib.values()]
    dists = [r["mean_dist_to_real"] for r in calib.values()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax = axes[0]
    ax.plot(ms, kurts, "o-", color="#4C72B0", label="sim kurtosis")
    ax.axhline(real_means["kurtosis"], color="#C44E52", ls="--", label="real CinC kurtosis")
    ax.axhline(15, color="green", ls=":", label="target >=15")
    ax.set_xlabel("noise multiplier"); ax.set_ylabel("kurtosis"); ax.set_title("Kurtosis vs noise mult")
    ax.legend(fontsize=7); ax.set_xscale("log")
    ax = axes[1]
    ax.plot(ms, ents, "s-", color="#4C72B0", label="sim sample entropy")
    ax.axhline(real_means["sample_entropy"], color="#C44E52", ls="--", label="real CinC entropy")
    ax.axhline(0.4, color="green", ls=":", label="target <=0.4")
    ax.set_xlabel("noise multiplier"); ax.set_ylabel("sample entropy"); ax.set_title("Entropy vs noise mult")
    ax.legend(fontsize=7); ax.set_xscale("log")
    ax = axes[2]
    ax.plot(ms, dists, "^-", color="#55A868", label="dist to real")
    ax.axvline(best_m, color="orange", ls=":", label=f"best m={best_m}")
    ax.set_xlabel("noise multiplier"); ax.set_ylabel("mean abs z-score dist"); ax.set_title("Distribution distance to real")
    ax.legend(fontsize=7); ax.set_xscale("log")
    plt.suptitle("E22: simulator recalibration — noise mult vs realism", fontsize=11)
    plt.tight_layout(); plt.savefig(RESULTS / "recalib_sweep.png", dpi=130); plt.close()
    # E17 rerun comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tags = list(train_results.keys())
    l4 = [train_results[t]["L4"]["macro_auroc"] for t in tags]
    l1 = [train_results[t]["L1"]["macro_auroc"] for t in tags]
    x = np.arange(len(tags)); w = 0.35
    ax.bar(x - w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    ax.bar(x + w/2, l4, w, label="L4 full sim-watch", color="#C44E52")
    ax.axhline(0.742, color="green", ls="--", lw=1, label="E17 default L4 = 0.742")
    ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=8)
    ax.set_ylabel("macro AUROC"); ax.set_ylim(0.5, 0.85)
    ax.set_title("E22: single-lead+sim edge under recalibration")
    ax.legend(fontsize=8, loc="lower right")
    for i, v in enumerate(l4): ax.text(i + w/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "edge_rerun.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
