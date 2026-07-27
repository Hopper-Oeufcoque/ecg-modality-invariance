"""Experiment 15 — MixStyle: modality-style mixing for domain generalization.

NOVEL CROSS-DOMAIN METHOD: MixStyle (Zhou et al. 2021, NeurIPS, "Domain
Generalization with MixStyle") is a domain-generalization technique from image
classification that *randomly mixes the instance-level feature statistics
(per-sample, per-channel mean/std) between samples in a batch* during training,
simulating novel "style" combinations the model never sees. It has NEVER been
applied to ECG recording-modality invariance. Here "style" = the channel-level
statistical signature of the recording chain (filter response, electrode
impedance, noise color, gain) and "content" = pathology. By mixing styles across
samples, the classifier is forced to use style-invariant content.

This is the principled, by-construction sibling of E10's post-hoc INLP
projection: E10 removes a learned linear modality direction after the fact;
MixStyle prevents the model from relying on style at all during training.

Design (MixStyle layer inserted after the stem, before the residual blocks):
  x_i' = sigma_j * (x_i - mu_i) / sigma_i + mu_j,   j = random perm(i)
  where mu_i, sigma_i are per-sample, per-channel statistics of x_i over the
  time axis. Applied with probability p during training; identity at test.

Variants (all 20 ep, single seed):
  V1 baseline lead-masking (reproduce E2 V2 ~0.718)
  V2 lead-masking + MixStyle (p=0.5)           — does style-mixing close residual gap?
  V3 single-lead+sim + MixStyle (p=0.5)        — does it help the E17 winner too?
  V4 MixStyle prob sweep [0.3, 0.5, 0.7] on V2 — is more style mixing better?

Honesty: single seed, sim-watch (E6 realism caveat applies), MixStyle placement
(after stem) is one choice (deeper placement untested), only channel 1st/2nd
moments mixed (higher-order style survives).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all
from src.model import ECGResNet1d, ResBlock1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "15_mixstyle"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = len(SUPERCLASSES)
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)


# ---------------------------------------------------------------------------
# MixStyle layer
# ---------------------------------------------------------------------------

class MixStyle(nn.Module):
    """Mix instance-level (per-sample, per-channel) statistics across the batch.

    x: (B, C, T). With prob p (train only), for each sample i, replace its
    per-channel (mu, sigma) with those of a random sample j=perm(i).
    """
    def __init__(self, p=0.5, alpha=0.1):
        super().__init__()
        self.p = p
        self.alpha = alpha  # Beta distribution concentration for soft mixing

    def forward(self, x):
        if not self.training or self.p == 0.0 or x.size(0) < 2:
            return x
        # decide which samples get mixed
        mix_mask = torch.rand(x.size(0), device=x.device) < self.p
        if not mix_mask.any():
            return x
        # per-sample per-channel stats over time
        mu = x.mean(dim=2, keepdim=True)   # (B, C, 1)
        sigma = x.std(dim=2, keepdim=True) + 1e-6
        # random permutation for style source
        perm = torch.randperm(x.size(0), device=x.device)
        # Beta mixing weight (soft style interpolation)
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (x.size(0), 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sigma + (1 - lam) * sigma[perm]
        # normalize with own stats, denormalize with mixed stats
        x_norm = (x - mu) / sigma
        x_mixed = x_norm * sig_mix + mu_mix
        # apply only to selected samples
        mask = mix_mask.view(-1, 1, 1).float()
        return mask * x_mixed + (1 - mask) * x


class MixStyleResNet(nn.Module):
    """ECGResNet1d with a MixStyle layer after the stem."""
    def __init__(self, n_leads=12, n_classes=N_CLASSES, base_ch=32, n_blocks=3, mixstyle_p=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(4),
        )
        self.mixstyle = MixStyle(p=mixstyle_p)
        self.blocks = nn.Sequential(*[ResBlock1d(base_ch) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(base_ch, n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.mixstyle(x)
        x = self.blocks(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Data + training (lead-masking and single-lead+sim regimes)
# ---------------------------------------------------------------------------

class LeadMaskDataset(torch.utils.data.Dataset):
    def __init__(self, records, labels_idx, lead_mask_prob=0.5):
        self.records = records; self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"]).copy()
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.lead_mask_prob > 0.0:
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False
            x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.labels_idx[i])


class SimSingleLeadDataset(torch.utils.data.Dataset):
    def __init__(self, records, labels_idx, sim_cfg):
        self.records = records; self.labels_idx = labels_idx; self.sim_cfg = sim_cfg
    def __len__(self): return len(self.records)
    def _fixlen(self, x):
        T = x.shape[0]
        return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)
    def __getitem__(self, i):
        x = self._fixlen(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, self.sim_cfg, LEAD_NAMES, rng=rng)
        leadI = out["watch"]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.labels_idx[i]))


def train(model, records, labels_idx, epochs=20, lr=1e-3, batch_size=64, tag="m"):
    dl = DataLoader(_to_ds(model, records, labels_idx), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


def _to_ds(model, records, labels_idx):
    if getattr(model, "single_lead", False):
        return SimSingleLeadDataset(records, labels_idx, model.sim_cfg)
    return LeadMaskDataset(records, labels_idx, lead_mask_prob=0.5)


@torch.no_grad()
def evaluate(model, records, labels_idx, n_leads, batch_size=64):
    model.eval()
    cfg_L1 = _cfg(apply_bandwidth=False, apply_electrode=False, apply_noise=False, apply_quantization=False, seed=0)
    cfg_L4 = _cfg(seed=0)
    out = {}
    for stage, cfg in [("L1", cfg_L1), ("L4", cfg_L4)]:
        all_logits = []; all_y = []
        for i in range(0, len(records), batch_size):
            batch = []
            for rec in records[i:i+batch_size]:
                x = rec["ecg"][:SIGLEN]
                if x.shape[0] < SIGLEN:
                    x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
                mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
                x = (x - mu) / sd
                o = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
                leadI = o["watch"]; leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
                if n_leads == 12:
                    framed = np.zeros((SIGLEN, 12), dtype=np.float32); framed[:, 0] = leadI
                else:
                    framed = leadI[:, None].astype(np.float32)
                batch.append(framed)
            xb = torch.from_numpy(np.stack(batch)).permute(0, 2, 1)
            all_logits.append(model(xb).numpy())
            all_y.append(np.array(labels_idx[i:i+batch_size]))
        logits = np.concatenate(all_logits); y = np.concatenate(all_y)
        out[stage] = {"macro_auroc": macro_auroc(logits, y), "per_class": per_class_auroc(logits, y)}
        print(f"  {stage}: macro={out[stage]['macro_auroc']:.4f}", flush=True)
    return out


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
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i = {c: i for i, c in enumerate(SUPERCLASSES)}
    tr, te = splits["train"], splits["test"]
    for r in tr: r["label_idx"] = c2i[r["label"]]
    for r in te: r["label_idx"] = c2i[r["label"]]
    ytr = np.array([r["label_idx"] for r in tr]); yte = np.array([r["label_idx"] for r in te])
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    sim_cfg = _cfg(seed=None)
    results = {}

    # V1 baseline lead-masking (no MixStyle)
    print("\n=== V1 baseline lead-masking (p_mix=0) ===", flush=True)
    m1 = MixStyleResNet(N_LEADS, mixstyle_p=0.0).to(DEVICE)
    train(m1, tr, ytr, tag="V1"); results["V1_LeadMask"] = evaluate(m1, te, yte, 12)

    # V2 lead-masking + MixStyle p=0.5
    print("\n=== V2 lead-masking + MixStyle (p_mix=0.5) ===", flush=True)
    m2 = MixStyleResNet(N_LEADS, mixstyle_p=0.5).to(DEVICE)
    train(m2, tr, ytr, tag="V2"); results["V2_LeadMask+MixStyle"] = evaluate(m2, te, yte, 12)

    # V3 single-lead+sim + MixStyle
    print("\n=== V3 single-lead+sim + MixStyle (p_mix=0.5) ===", flush=True)
    m3 = MixStyleResNet(1, mixstyle_p=0.5).to(DEVICE); m3.single_lead = True; m3.sim_cfg = sim_cfg
    train(m3, tr, ytr, tag="V3"); results["V3_SimLead+MixStyle"] = evaluate(m3, te, yte, 1)

    # V4 MixStyle prob sweep on lead-masking
    for p in [0.3, 0.7]:
        print(f"\n=== V4 lead-masking + MixStyle (p_mix={p}) ===", flush=True)
        m = MixStyleResNet(N_LEADS, mixstyle_p=p).to(DEVICE)
        train(m, tr, ytr, tag=f"V4_p{p}"); results[f"V4_LeadMask+MixStyle_p{p}"] = evaluate(m, te, yte, 12)

    metrics = {
        "fs": FS, "n_train": len(tr), "n_test": len(te), "classes": SUPERCLASSES,
        "variants": results,
        "e2_reference_L4": 0.718, "e17_reference_L4": 0.742,
        "honesty": ["single seed", "sim-watch (E6 realism caveat)", "MixStyle after stem only",
                    "only 1st/2nd channel moments mixed (higher-order style survives)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    l4 = [results[n]["L4"]["macro_auroc"] for n in names]
    l1 = [results[n]["L1"]["macro_auroc"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, l1, w, label="L1 clean Lead-I", color="#4C72B0")
    b2 = ax.bar(x + w/2, l4, w, label="L4 full sim-watch", color="#C44E52")
    ax.axhline(0.718, color="gray", ls="--", lw=1, label="lead-masking ref 0.718")
    ax.axhline(0.742, color="green", ls="--", lw=1, label="E17 sim ref 0.742")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("macro AUROC"); ax.set_ylim(0.5, 0.82)
    ax.set_title("E15: MixStyle modality-style mixing")
    ax.legend(fontsize=8, loc="lower right")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{b.get_height():.3f}", ha="center", fontsize=7)
    plt.tight_layout(); plt.savefig(RESULTS / "mixstyle.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
