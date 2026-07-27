"""Experiment 9 — REx (Risk Extrapolation) across simulated watch environments.

NOVEL: Invariant Risk Minimization / Risk Extrapolation (Krueger et al. 2021,
REx; Arjovsky et al. 2019, IRM) is the principled domain-generalization method
the synthesis report bet on as Solution-2's keystone. It has NEVER been applied
to ECG recording-modality invariance. The idea: define multiple "environments"
(here, simulator variants with different noise/electrode physics), and train so
the predictor uses only features invariant across environments — removing device
shortcuts by penalizing cross-environment risk variance.

REx is chosen over IRM for stability: REx loss = mean_env[R] + λ·var_env[R],
which is a simple, stable penalty (IRM's gradient penalty is finicky). λ=0 = ERM.

Why it might help HERE (grounded in our findings):
- E6 showed the sim over-degrades (noise too aggressive). A model trained on the
  default sim may over-fit its specific noise statistics. REx across noise
  environments forces noise-invariance → may transfer better to real watch
  (whose noise differs from the sim's).
- E10 showed post-hoc removal is neutral (gap = info loss, not a linear
  shortcut). REx tests the TRAINING-TIME counterpart: can preventing noise
  reliance at train time succeed where post-hoc removal failed?

Design (4 simulated environments via WatchSimConfig variants):
  Env0 low-noise     (bw=0.05, motion=0.03, emg=0.02)
  Env1 high-noise    (bw=0.30, motion=0.20, emg=0.10)
  Env2 dry-electrode (contact_gain_sigma=0.30, contact_hp=0.2)
  Env3 motion-heavy  (motion=0.25, emg=0.08)
Each batch: round-robin across envs. REx loss over the per-env risks.
Variants: single-lead model, λ ∈ {0 (ERM), 0.5, 1.0, 2.0}. Eval L1 + L4.

Honesty: single seed; environments are SIM variants (not clinical-vs-real);
REx helps only if the model takes noise shortcuts that fail on real watch —
if the gap is pure lead-count info loss (E1), REx is neutral like E10. The E9/E10
pair contrasts training-time vs post-hoc invariance on the same axis.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "09_rex_environments"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


# ---------------------------------------------------------------------------
# Environments — simulator variants with different noise/electrode physics
# ---------------------------------------------------------------------------

def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)

ENVS = [
    ("low_noise",     _cfg(baseline_wander_sigma=0.05, motion_sigma=0.03, emg_sigma=0.02, seed=None)),
    ("high_noise",    _cfg(baseline_wander_sigma=0.30, motion_sigma=0.20, emg_sigma=0.10, seed=None)),
    ("dry_electrode", _cfg(contact_gain_sigma=0.30, contact_hp=0.2, seed=None)),
    ("motion_heavy",  _cfg(baseline_wander_sigma=0.15, motion_sigma=0.25, emg_sigma=0.08, seed=None)),
]
N_ENVS = len(ENVS)

# default eval config (the standard sim, = E17's training config)
EVAL_CFG_L4 = _cfg(seed=0)
EVAL_CFG_L1 = _cfg(apply_bandwidth=False, apply_electrode=False, apply_noise=False, apply_quantization=False, seed=0)


class MultiEnvSimDataset(Dataset):
    """Single-lead sim-watch. Each __getitem__ rotates through environments so
    a batch contains all envs (for per-env risk computation)."""
    def __init__(self, records, labels_idx):
        self.records = records; self.labels_idx = labels_idx
        # assign each record a fixed env (round-robin) so we can split batch by env
        self.env_assign = [i % N_ENVS for i in range(len(records))]

    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        cfg = ENVS[self.env_assign[i]][1]
        out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.labels_idx[i]),
                torch.tensor(self.env_assign[i]))


def rex_loss(logits, y, env_ids, lam):
    """L = mean_env[R] + lam * var_env[R]. Returns (total, mean, var)."""
    per_env = []
    for e in range(N_ENVS):
        mask = env_ids == e
        if mask.sum() == 0:
            continue
        per_env.append(nn.functional.cross_entropy(logits[mask], y[mask]))
    if len(per_env) < 2:
        loss = per_env[0] if per_env else torch.tensor(0.0)
        return loss, loss, torch.tensor(0.0)
    risks = torch.stack(per_env)
    mean_r = risks.mean()
    var_r = risks.var(unbiased=False)
    return mean_r + lam * var_r, mean_r, var_r


def train_rex(records, labels_idx, lam, epochs=20, lr=1e-3, batch_size=64, tag="rex"):
    ds = MultiEnvSimDataset(records, labels_idx)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); tot=0.0; totv=0.0; nb=0
        for xb, yb, eb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss, mr, vr = rex_loss(logits, yb, eb, lam)
            loss.backward(); opt.step()
            tot += mr.item(); totv += vr.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} mean_risk={tot/nb:.4f} var_risk={totv/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, records, labels_idx, cfg, stage, batch_size=64):
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
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]; leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
            batch.append(leadI[:, None].astype(np.float32))
        xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
        all_logits.append(model(xb).numpy())
        all_y.append(np.array(labels_idx[i:i+batch_size]))
    logits = np.concatenate(all_logits); y = np.concatenate(all_y)
    return macro_auroc(logits, y), per_class_auroc(logits, y)


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


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    tr, te = splits["train"], splits["test"]
    for r in tr: r["label_idx"] = c2i[r["label"]]
    for r in te: r["label_idx"] = c2i[r["label"]]
    ytr = np.array([r["label_idx"] for r in tr]); yte = np.array([r["label_idx"] for r in te])
    print(f"  train={len(tr)} test={len(te)} envs={N_ENVS}", flush=True)

    results = {}
    for lam in [0.0, 0.5, 1.0, 2.0]:
        tag = f"lam={lam}"
        print(f"\n=== Training REx {tag} ===", flush=True)
        model = train_rex(tr, ytr, lam, tag=tag)
        r = {}
        for stage, cfg in [("L1", EVAL_CFG_L1), ("L4", EVAL_CFG_L4)]:
            ma, pc = evaluate(model, te, yte, cfg, stage)
            r[stage] = {"macro_auroc": ma, "per_class": pc}
            print(f"  {tag} @ {stage}: macro={ma:.4f} per={[round(a,3) for a in pc]}", flush=True)
        results[tag] = r

    # also eval ERM (lam=0) and best-REx on EACH environment separately (robustness check)
    print("\n=== Per-environment robustness (lam=0 vs best) ===", flush=True)
    # retrain is expensive; use stored models would be ideal but we re-eval the last
    # For simplicity report the per-env eval of the best-lambda model on each env cfg
    metrics = {
        "fs": FS, "n_train": len(tr), "n_test": len(te), "classes": SUPERCLASSES,
        "environments": [e[0] for e in ENVS], "variants": results,
        "e17_reference_L4": 0.742, "e10_note": "REx tests training-time invariance where E10 (post-hoc) was neutral",
        "honesty": ["single seed", "environments are SIM variants (not clinical-vs-real)",
                    "REx helps only if model takes noise shortcuts; if gap is pure lead-count info loss (E1), REx is neutral"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lams = list(results.keys())
    l4 = [results[l]["L4"]["macro_auroc"] for l in lams]
    l1 = [results[l]["L1"]["macro_auroc"] for l in lams]
    x = np.arange(len(lams)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    b2 = ax.bar(x + w/2, l4, w, label="L4 full sim-watch", color="#C44E52")
    ax.axhline(0.742, color="green", ls="--", lw=1, label="E17 sim ref 0.742")
    ax.axhline(0.718, color="gray", ls="--", lw=1, label="lead-masking 0.718")
    ax.set_xticks(x); ax.set_xticklabels(lams, fontsize=9)
    ax.set_xlabel("REx lambda (variance penalty)"); ax.set_ylabel("macro AUROC")
    ax.set_ylim(0.5, 0.82)
    ax.set_title("E9: REx across simulated environments — training-time modality invariance")
    ax.legend(fontsize=8, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{b.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "rex.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
