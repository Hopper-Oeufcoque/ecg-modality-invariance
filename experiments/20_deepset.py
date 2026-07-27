"""Experiment 20 — Set-invariant (DeepSet) lead networks (E3 in taxonomy, novel).

Treat leads as an UNORDERED SET rather than fixed channels. A permutation-
invariant architecture (DeepSets) is naturally lead-count-agnostic: it pools
over whatever leads are present, so going from 12 to 1 lead is just a smaller
set, not a domain shift. Borrowed from point-cloud/set literature — never
applied to ECG modality invariance.

Architecture: per-lead encoder φ(lead) -> feature, then permutation-invariant
sum/mean pool -> classifier. By construction:
  - 12 leads and 1 lead use the SAME model (no channel-count mismatch).
  - Adding/dropping leads doesn't change the input interface.
  - The model can't rely on lead identity (position), only lead content.

Compare to lead-masking (which still uses fixed 12-channel input with zeros)
and the single-lead model (which can't use multi-lead). The DeepSet should
gracefully handle both, ideally matching lead-masking on 12-lead training and
single-lead+sim on single-lead inference.
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
from src.model import ResBlock1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "20_deepset"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0
SIGLEN = 1000
N_LEADS = 12
N_CLASSES = len(SUPERCLASSES)
FEAT_DIM = 32
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def _cfg(**kw):
    kw.setdefault("fs_watch", FS)
    return WatchSimConfig(FS, **kw)
WATCH_CFG = _cfg(seed=None)


# ---------------------------------------------------------------------------
# DeepSet architecture
# ---------------------------------------------------------------------------

class LeadEncoder(nn.Module):
    """φ(lead) -> feature vector. Shared across leads (permutation-invariant)."""
    def __init__(self, feat_dim=FEAT_DIM, base_ch=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch), nn.ReLU(), nn.MaxPool1d(4))
        self.blocks = nn.Sequential(*[ResBlock1d(base_ch) for _ in range(2)])
        self.head = nn.Sequential(nn.BatchNorm1d(base_ch), nn.ReLU(),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                   nn.Linear(base_ch, feat_dim))

    def forward(self, x):  # x: (B*n_leads, 1, T)
        return self.head(self.blocks(self.stem(x)))


class DeepSetClassifier(nn.Module):
    """Permutation-invariant over leads: pool φ(lead) features."""
    def __init__(self, feat_dim=FEAT_DIM, n_classes=N_CLASSES, pool="mean"):
        super().__init__()
        self.encoder = LeadEncoder(feat_dim)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(feat_dim, n_classes))
        self.pool = pool

    def forward(self, x, lead_mask=None):
        # x: (B, n_leads, T) -> encode each lead, pool
        B, L, T = x.shape
        x = x.reshape(B*L, 1, T)
        feats = self.encoder(x).reshape(B, L, -1)  # (B, L, feat_dim)
        if lead_mask is not None:
            # mask out absent leads (lead_mask: (B,L), 1=present)
            m = lead_mask.unsqueeze(-1).float()  # (B,L,1)
            feats = feats * m
            if self.pool == "mean":
                pooled = feats.sum(1) / (m.sum(1) + 1e-6)
            else:
                pooled = feats.sum(1)
        else:
            pooled = feats.mean(1) if self.pool == "mean" else feats.sum(1)
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DeepSetDataset(Dataset):
    """12-lead input + lead-masking via mask tensor (not zeroing)."""
    def __init__(self, records, labels_idx, lead_mask_prob=0.5, watch_sim=False):
        self.records = records; self.labels_idx = labels_idx
        self.lead_mask_prob = lead_mask_prob; self.watch_sim = watch_sim

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        x = self.records[i]["ecg"][:SIGLEN]
        if x.shape[0] < SIGLEN:
            x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.watch_sim:
            out = simulate_watch(x, FS, WATCH_CFG, LEAD_NAMES, rng=rng)
            x = x.copy(); x[:, 0] = out["watch"]
        # lead mask: 1 = present, 0 = dropped. Always keep Lead I (idx 0).
        mask = (rng.random(N_LEADS) >= self.lead_mask_prob).astype(np.float32)
        mask[0] = 1.0
        x = x * mask[None, :]  # zero dropped leads (broadcast over time)
        return (torch.from_numpy(x.astype(np.float32)).permute(1, 0),  # (n_leads, T)
                torch.from_numpy(mask),
                torch.tensor(self.labels_idx[i]))


def train_model(records, labels_idx, model, lead_mask_prob=0.5, watch_sim=False,
                epochs=20, lr=1e-3, batch_size=64, tag="m"):
    ds = DeepSetDataset(records, labels_idx, lead_mask_prob, watch_sim)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, mb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb, mb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def evaluate(model, records, labels_idx, n_leads_present=1, stage="L4", batch_size=64):
    """Eval: only n_leads_present leads active (Lead-I first)."""
    model.eval()
    cfg = (_cfg(apply_bandwidth=False, apply_electrode=False,
                apply_noise=False, apply_quantization=False, seed=0)
           if stage == "L1" else _cfg(seed=0))
    all_l=[]; all_y=[]
    for i in range(0, len(records), batch_size):
        batch=[]; masks=[]; ys=[]
        for rec in records[i:i+batch_size]:
            x = rec["ecg"][:SIGLEN]
            if x.shape[0] < SIGLEN:
                x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0], 12))], 0)
            mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
            x = (x - mu) / sd
            out = simulate_watch(x, FS, cfg, LEAD_NAMES, rng=np.random.default_rng(0))
            leadI = out["watch"]
            leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-9)
            x = np.zeros((SIGLEN, 12), dtype=np.float32)
            x[:, 0] = leadI
            mask = np.zeros(N_LEADS, dtype=np.float32); mask[0] = 1.0
            batch.append(x.T.copy()); masks.append(mask)  # (n_leads, T)
        xb = torch.from_numpy(np.stack(batch))
        mb = torch.from_numpy(np.stack(masks))
        all_l.append(model(xb, mb).numpy())
        all_y.append(labels_idx[i:i+len(batch)])
    return np.concatenate(all_l), np.concatenate(all_y)


def per_class_auroc(logits, y, n_classes=N_CLASSES):
    from sklearn.metrics import roc_auc_score
    aucs=[]
    for c in range(n_classes):
        yb=(y==c).astype(int)
        if yb.sum()==0 or yb.sum()==len(yb): aucs.append(float("nan")); continue
        try: aucs.append(float(roc_auc_score(yb, logits[:,c])))
        except Exception: aucs.append(float("nan"))
    return aucs
def macro_auroc(logits, y):
    aucs=[a for a in per_class_auroc(logits,y) if not np.isnan(a)]
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    print("Loading PTB-XL (max_per_class=400) ...", flush=True)
    splits = load_all(max_per_class=400)
    c2i={c:i for i,c in enumerate(SUPERCLASSES)}
    def lab(rs): return np.array([c2i[r['label']] for r in rs])
    tr,te=splits['train'],splits['test']; ytr,yte=lab(tr),lab(te)
    print(f"  train={len(tr)} test={len(te)}", flush=True)

    results = {}
    # V1: DeepSet + lead-masking (12-lead training)
    print("\n=== V1: DeepSet + lead-masking (12-lead train) ===", flush=True)
    m1 = DeepSetClassifier(FEAT_DIM, N_CLASSES).to(DEVICE)
    m1 = train_model(tr, ytr, m1, lead_mask_prob=0.5, epochs=20, tag="V1")
    for st in ("L1","L4"):
        lg,_ = evaluate(m1, te, yte, stage=st)
        results[f"V1_deepset_{st}"]={"macro_auroc":macro_auroc(lg,yte)}
        print(f"  V1 deepset @ {st}: {results[f'V1_deepset_{st}']['macro_auroc']:.4f}", flush=True)

    # V2: DeepSet + lead-masking + watch-sim (single-lead targeted)
    print("\n=== V2: DeepSet + lead-masking + watch-sim ===", flush=True)
    m2 = DeepSetClassifier(FEAT_DIM, N_CLASSES).to(DEVICE)
    m2 = train_model(tr, ytr, m2, lead_mask_prob=0.5, watch_sim=True, epochs=20, tag="V2")
    for st in ("L1","L4"):
        lg,_ = evaluate(m2, te, yte, stage=st)
        results[f"V2_deepset_sim_{st}"]={"macro_auroc":macro_auroc(lg,yte)}
        print(f"  V2 deepset+sim @ {st}: {results[f'V2_deepset_sim_{st}']['macro_auroc']:.4f}", flush=True)

    metrics={"n_train":len(tr),"n_test":len(te),"classes":SUPERCLASSES,"variants":results}
    (RESULTS/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    print("\nDONE.", flush=True)


if __name__=="__main__":
    main()
