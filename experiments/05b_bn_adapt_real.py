"""Experiment 5b — Test-time BN adaptation on REAL CinC (salvage the 12-lead model).

DIRECT FOLLOW-UP TO E23: E23 found the 12-lead lead-masking model catastrophically
fails on real CinC (0.557 vs sim 0.718). Hypothesized mechanism: BN normalization
mismatch (11 zero-padded channels + real-device Lead-I violate PTB-XL-computed
BN running stats). E5b tests whether recomputing BN running stats from the REAL
target batch before predicting recovers the gap.

If BN-adapt recovers 0.557 → ~0.72 (single-lead level) → lead-masking IS viable
for real deployment with a test-time BN recompute step (a label-free, parameter-
free intervention). If it doesn't → the 12-lead prior is fundamentally broken on
real shift, and single-lead is the only robust path.

E5 (the sim version) was marginal (+0.007 L4) because lead-masking already closed
the dominant axis on sim. But E23's real failure looks BN-driven specifically, so
BN-adapt may matter MUCH more on real than it did on sim.

Variants (binary NORM-vs-AF, tested on REAL CinC 2017):
  V1 12-lead lead-masking, no adapt (E23 V1 = 0.557, reference)
  V2 12-lead lead-masking + BN-adapt (recompute BN stats from real CinC batch)
  V3 12-lead lead-masking + BN-adapt (2 passes for stability)
  Also: single-lead clean (E23 V2 = 0.721) and sim (0.731) for reference.

Honesty: single seed; binary AF/NORM; CinC handheld; BN-adapt is a form of TTA
(transmissible to deployment since it's label-free — uses the target batch stats).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "05b_bn_adapt_real"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = 2
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)


def load_ptbxl_binary(data_dir, max_per_class=700):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    df["ecg_id"] = df["ecg_id"].astype(int)
    def has(code, s):
        try: c = ast.literal_eval(s)
        except: return False
        return code in c
    afib = df["scp_codes"].apply(lambda s: has("AFIB", s))
    norm = df["scp_codes"].apply(lambda s: has("NORM", s))
    aflt = df["scp_codes"].apply(lambda s: has("AFLT", s))
    pos = df[afib & ~norm & ~aflt].copy(); pos["y"] = 1
    neg = df[norm & ~afib & ~aflt].copy(); neg["y"] = 0
    pos = pos.sample(frac=1.0, random_state=0).head(max_per_class)
    neg = neg.sample(frac=1.0, random_state=0).head(max_per_class)
    import wfdb
    out = {"train": [], "test": []}
    for sub in (pos, neg):
        for _, row in sub.iterrows():
            fold = int(row["strat_fold"])
            split = "train" if fold <= 8 else "test"
            try: sig, _ = wfdb.rdsamp(str(data_dir / row["filename_lr"]))
            except: continue
            out[split].append({"ecg": sig.astype(np.float32), "y": int(row["y"])})
    return out


def load_cinc_binary(n_per_class=700, fs=300):
    data_dir = Path.home() / "data" / "cinc2017" / "training2017"
    ref = {}
    for line in (data_dir / "REFERENCE.csv").read_text().splitlines():
        p = line.strip().split(",")
        if len(p) == 2: ref[p[0]] = p[1]
    import scipy.io as sio
    A = []; N = []
    for mf in sorted(data_dir.glob("A*.mat")):
        lab = ref.get(mf.stem, "O")
        if lab not in ("A", "N"): continue
        try: sig = sio.loadmat(mf)["val"][0].astype(np.float64)
        except: continue
        if sig.size < SIGLEN: continue
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        from math import gcd
        g = gcd(int(fs), int(FS)); up = int(FS)//g; down = int(fs)//g
        sig = resample_poly(sig, up, down)
        rec = {"ecg": sig[:SIGLEN].astype(np.float32), "y": 1 if lab == "A" else 0}
        (A if lab == "A" else N).append(rec)
    return A[:n_per_class], N[:n_per_class]


class LeadMaskDataset(Dataset):
    def __init__(self, records, lead_mask_prob=0.5):
        self.records = records; self.lead_mask_prob = lead_mask_prob
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = self.records[i]["ecg"][:SIGLEN].copy()
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros((SIGLEN-x.shape[0], x.shape[1]))], 0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.lead_mask_prob > 0.0:
            mask = rng.random(N_LEADS) < self.lead_mask_prob; mask[0] = False
            x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.records[i]["y"])


class CinCReal12Lead(Dataset):
    """Real CinC single-lead placed in ch0, zero-pad to 12 (the E23 eval protocol)."""
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = self.records[i]["ecg"]
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0],))], 0)
        x = x[:SIGLEN]; x = (x - x.mean()) / (x.std() + 1e-6)
        framed = np.zeros((SIGLEN, N_LEADS), dtype=np.float32); framed[:, 0] = x
        return torch.from_numpy(framed.T), torch.tensor(self.records[i]["y"])


def train(model, ds, epochs=20, lr=1e-3, batch_size=64, tag="m"):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, ds, batch_size=64):
    model.eval()
    dl = DataLoader(ds, batch_size=batch_size)
    logits = []; ys = []
    for xb, yb in dl:
        logits.append(model(xb).numpy()); ys.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def bn_adapt(model, ds, passes=1, batch_size=64):
    """Recompute BN running stats from target batch. Train mode (BN uses batch stats)
    but no gradient/parameter update. Update running stats via momentum."""
    model.train()  # BN uses batch stats in train mode
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for _ in range(passes):
        for xb, _ in dl:
            _ = model(xb)  # forward updates BN running stats (momentum default 0.1)
    model.eval()
    return model


def auroc(logits, y):
    from sklearn.metrics import roc_auc_score, accuracy_score
    proba = torch.softmax(torch.from_numpy(logits), 1).numpy()[:, 1]
    return float(roc_auc_score(y, proba)), float(accuracy_score(y, logits.argmax(1)))


def main():
    print("Loading PTB-XL binary AF/NORM ...", flush=True)
    ptb = load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]
    print(f"  train={len(tr)} (AF={sum(r['y'] for r in tr)})", flush=True)
    print("Loading REAL CinC (N vs A, 700 each) ...", flush=True)
    cA, cN = load_cinc_binary(n_per_class=700)
    cinc = cA + cN
    print(f"  CinC real: {len(cinc)}", flush=True)

    cinc_ds = CinCReal12Lead(cinc)

    print("\n=== Training 12-lead lead-masking model ===", flush=True)
    model = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    train(model, LeadMaskDataset(tr, lead_mask_prob=0.5), tag="leadmask")

    results = {}
    # V1 no adapt (E23 reference, re-run in-experiment seed)
    logits, y = predict(model, cinc_ds)
    auc, acc = auroc(logits, y)
    results["V1_no_adapt"] = {"auroc": auc, "acc": acc}
    print(f"  V1 no-adapt: AUROC={auc:.4f} ACC={acc:.4f}", flush=True)

    # V2 BN-adapt 1 pass
    model2 = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    model2.load_state_dict(model.state_dict())
    bn_adapt(model2, cinc_ds, passes=1)
    logits, y = predict(model2, cinc_ds)
    auc, acc = auroc(logits, y)
    results["V2_bn_adapt_1pass"] = {"auroc": auc, "acc": acc}
    print(f"  V2 BN-adapt 1pass: AUROC={auc:.4f} ACC={acc:.4f}", flush=True)

    # V3 BN-adapt 3 passes
    model3 = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    model3.load_state_dict(model.state_dict())
    bn_adapt(model3, cinc_ds, passes=3)
    logits, y = predict(model3, cinc_ds)
    auc, acc = auroc(logits, y)
    results["V3_bn_adapt_3pass"] = {"auroc": auc, "acc": acc}
    print(f"  V3 BN-adapt 3pass: AUROC={auc:.4f} ACC={acc:.4f}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "n_train": len(tr), "n_cinc": len(cinc),
        "variants": results,
        "e23_reference": {"V1_no_adapt": 0.557, "single_lead_clean": 0.721, "single_lead_sim": 0.731},
        "headline": "V2/V3 vs V1: does BN-adapt recover the 12-lead model's catastrophic real failure?",
        "honesty": ["single seed", "binary AF/NORM", "CinC handheld (cleaner than wrist)",
                    "BN-adapt is label-free TTA (transmissible: uses target batch stats only)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["V1_no_adapt", "V2_bn_adapt_1pass", "V3_bn_adapt_3pass"]
    labels = ["V1 no adapt\n(E23 ref)", "V2 BN-adapt\n1 pass", "V3 BN-adapt\n3 passes"]
    aucs = [results[n]["auroc"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(3), aucs, color=["#C44E52", "#4C72B0", "#55A868"])
    ax.axhline(0.721, color="gray", ls="--", lw=1, label="single-lead clean ref 0.721")
    ax.axhline(0.557, color="red", ls=":", lw=1, label="E23 no-adapt 0.557")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC on REAL CinC (binary NORM vs AF)"); ax.set_ylim(0.4, 0.85)
    ax.set_title("E5b: test-time BN adaptation — salvaging the 12-lead model on real")
    ax.legend(fontsize=8, loc="lower right")
    for b, v in zip(bars, aucs): ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(RESULTS / "bn_adapt_real.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
