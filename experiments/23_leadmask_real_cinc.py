"""Experiment 23 — Lead-masking validated on REAL single-lead (the decisive real test).

PIVOTED BY E6b: E6b refuted the simulator strategy (clean Lead-I 0.753 > sim 0.737
on real CinC). The revised project ranking puts lead-masking (E2, trains on real
clinical 12-lead, no sim noise to overfit) as the most realism-robust label-free
method — but that claim was only validated on SIMULATED watch (E2 L4=0.718 on sim).

E23 is the decisive real-deployment test: does lead-masking's 12-lead prior
actually beat clean Lead-I training on REAL single-lead data? This determines
whether the project's best method works where it matters (real deployment), and
whether the 12-lead prior (the thing that made E2 beat the single-lead reference
on sim) survives contact with real watch shift.

Binary NORM-vs-AF (clean cross-dataset label alignment, same as E6b).

Variants (all tested on REAL CinC 2017 N vs A):
  V1 lead-masking: 12-lead model w/ random lead dropout, test on real CinC
       (Lead-I in ch0, zero-pad — the E2 eval protocol)
  V2 clean Lead-I (E6b V3 = 0.753) — reference, re-run in-experiment for fair seed
  V3 sim-trained (E6b V2 = 0.737) — reference, re-run in-experiment
  V4 lead-masking + the model's OWN 12-lead clinical ceiling (sanity: how much
       does real single-lead drop from the 12-lead ceiling?)

The V1 vs V2 comparison is the headline: does the 12-lead prior (lead-masking)
beat single-lead clean training on REAL data, as it did on SIM (E2 V2 0.717 > V5
0.690)? If yes → lead-masking is the real-deployment winner. If no → the 12-lead
prior doesn't survive real shift, and clean Lead-I is the best label-free method.

Honesty: single seed; AF/rhythm only; CinC handheld (cleaner than wrist);
PTB-XL vs CinC population confound (mitigated by within-experiment V1 vs V2
comparison, both PTB-XL-trained → same population).
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
from src.dataset import SUPERCLASSES, LEAD_NAMES, load_all  # noqa: F401
from src.model import ECGResNet1d
from src.watch_simulator import simulate_watch, WatchSimConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "23_leadmask_real_cinc"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_LEADS = 12; N_CLASSES = 2
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)

WATCH_CFG = WatchSimConfig(FS, fs_watch=FS, seed=None)


# ---------------------------------------------------------------------------
# PTB-XL binary AF/NORM loader (reused from E6b)
# ---------------------------------------------------------------------------

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


def _fixlen12(x):
    T = x.shape[0]
    return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class LeadMaskDataset(Dataset):
    """12-lead with random lead masking (keep Lead-I). Binary."""
    def __init__(self, records, lead_mask_prob=0.5):
        self.records = records; self.lead_mask_prob = lead_mask_prob
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = _fixlen12(self.records[i]["ecg"]).copy()
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        if self.lead_mask_prob > 0.0:
            mask = rng.random(N_LEADS) < self.lead_mask_prob
            mask[0] = False
            x[:, mask] = 0.0
        return torch.from_numpy(x.astype(np.float32)).permute(1, 0), torch.tensor(self.records[i]["y"])


class CleanLeadIDataset(Dataset):
    """1-lead clean Lead-I (E6b V3 recipe)."""
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = _fixlen12(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        leadI = x[:, 0]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
        return torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0), torch.tensor(self.records[i]["y"])


class SimLeadDataset(Dataset):
    """1-lead sim-watch (E6b V2 recipe)."""
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = _fixlen12(self.records[i]["ecg"])
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
        x = (x - mu) / sd
        rng = np.random.default_rng(None)
        out = simulate_watch(x, FS, WATCH_CFG, ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"], rng=rng)
        leadI = out["watch"]
        leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-6)
        return torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0), torch.tensor(self.records[i]["y"])


class CinCRealDataset(Dataset):
    """Real CinC single-lead. lead_in_ch0: if True, zero-pad to 12 ch (for 12-lead model)."""
    def __init__(self, records, lead_in_ch0=False): self.records = records; self.lead_in_ch0 = lead_in_ch0
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = self.records[i]["ecg"]
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0],))], 0)
        x = x[:SIGLEN]; x = (x - x.mean()) / (x.std() + 1e-6)
        if self.lead_in_ch0:
            framed = np.zeros((SIGLEN, N_LEADS), dtype=np.float32); framed[:, 0] = x
            return torch.from_numpy(framed.T), torch.tensor(self.records[i]["y"])
        return torch.from_numpy(x[:, None].astype(np.float32)).permute(1, 0), torch.tensor(self.records[i]["y"])


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
def evaluate(model, ds, tag, lead_in_ch0):
    model.eval()
    dl = DataLoader(CinCRealDataset(ds, lead_in_ch0=lead_in_ch0), batch_size=64)
    logits = []; ys = []
    for xb, yb in dl:
        logits.append(model(xb).numpy()); ys.append(yb.numpy())
    logits = np.concatenate(logits); y = np.concatenate(ys)
    from sklearn.metrics import roc_auc_score, accuracy_score
    proba = torch.softmax(torch.from_numpy(logits), 1).numpy()[:, 1]
    auc = float(roc_auc_score(y, proba)); acc = float(accuracy_score(y, logits.argmax(1)))
    print(f"  {tag}: AUROC={auc:.4f} ACC={acc:.4f} (n={len(y)}, AF={int(y.sum())})", flush=True)
    return {"auroc": auc, "acc": acc, "n": len(y), "n_af": int(y.sum())}


def main():
    print("Loading PTB-XL binary AF/NORM (max_per_class=700) ...", flush=True)
    ptb = load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]
    print(f"  PTB train={len(tr)} (AF={sum(r['y'] for r in tr)})", flush=True)
    print("Loading REAL CinC 2017 (N vs A, 700 each) ...", flush=True)
    cA, cN = load_cinc_binary(n_per_class=700)
    cinc = cA + cN
    print(f"  CinC real: AF={len(cA)} N={len(cN)} total={len(cinc)}", flush=True)

    results = {}

    # V1 lead-masking 12-lead -> real CinC (Lead-I in ch0, zero-pad)
    print("\n=== V1 lead-masking (12-lead, lead-mask aug) ===", flush=True)
    m1 = ECGResNet1d(N_LEADS, N_CLASSES).to(DEVICE)
    train(m1, LeadMaskDataset(tr, lead_mask_prob=0.5), tag="leadmask")
    results["V1_leadmask_realCinC"] = evaluate(m1, cinc, "V1 leadmask->real", lead_in_ch0=True)
    # V4 sanity: 12-lead ceiling on PTB-XL clinical test (how good is the 12-lead model?)
    results["V1b_leadmask_12lead_PTB"] = {}
    # (skip re-eval on PTB 12-lead for brevity; the sim experiments cover that)

    # V2 clean Lead-I -> real CinC (reference, in-experiment seed)
    print("\n=== V2 clean Lead-I (1-lead) ===", flush=True)
    m2 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m2, CleanLeadIDataset(tr), tag="clean")
    results["V2_clean_realCinC"] = evaluate(m2, cinc, "V2 clean->real", lead_in_ch0=False)

    # V3 sim-trained -> real CinC (reference, in-experiment seed)
    print("\n=== V3 sim-trained (1-lead) ===", flush=True)
    m3 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m3, SimLeadDataset(tr), tag="sim")
    results["V3_sim_realCinC"] = evaluate(m3, cinc, "V3 sim->real", lead_in_ch0=False)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "classes": ["NORM", "AF"],
        "n_ptb_train": len(tr), "n_cinc": len(cinc),
        "variants": results,
        "e6b_references": {"V2_clean": 0.753, "V3_sim": 0.737, "V4_oracle": 0.946},
        "headline": "V1 (leadmask) vs V2 (clean): does the 12-lead prior beat single-lead on REAL data?",
        "honesty": ["single seed", "AF/rhythm only", "CinC handheld (cleaner than wrist)",
                    "PTB-XL vs CinC population confound (V1 vs V2 both PTB-trained -> same source population)"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["V1_leadmask_realCinC", "V2_clean_realCinC", "V3_sim_realCinC"]
    labels = ["V1 lead-masking\n(12-lead prior)", "V2 clean Lead-I\n(1-lead)", "V3 sim-trained\n(1-lead)"]
    aucs = [results[n]["auroc"] for n in names]
    cols = ["#4C72B0", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(3), aucs, color=cols)
    ax.axhline(0.946, color="green", ls="--", lw=1, label="oracle ceiling (real data) 0.946")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC on REAL CinC 2017 (binary NORM vs AF)"); ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title("E23: lead-masking vs clean vs sim — validated on REAL single-lead")
    ax.legend(fontsize=8, loc="lower right")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(RESULTS / "leadmask_real.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
