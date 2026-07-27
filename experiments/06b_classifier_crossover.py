"""Experiment 6b — Classifier cross-over: train on sim, test on REAL single-lead.

THE DEFINITIVE sim/real transfer test (spawned by E6, which was distribution-level
only). E6 found the simulator over-degrades; E6b asks whether that mismatch
matters at the *task* level — does a model trained on simulated watch actually
detect disease on REAL single-lead ECG?

Binary task chosen for clean cross-dataset label alignment: NORM vs Atrial
Fibrillation. AF maps cleanly in both PTB-XL (scp code AFIB) and CinC 2017
(REFERENCE label 'A' = AF, 'N' = normal) — no ambiguous spatial-class mapping.

Variants (all 1-lead ECGResNet1d, binary, 20 ep, single seed):
  V1 sim-trained  -> sim PTB-XL test     (sim->sim baseline; the favorable match)
  V2 sim-trained  -> REAL CinC N vs A     (cross-over: does the edge transfer?)
  V3 clinical-LeadI-trained -> REAL CinC  (control: does sim help or HURT real transfer vs clean clinical training?)
  V4 REAL CinC-trained -> REAL CinC test  (oracle: upper bound with real target data)

The V1→V2 gap = the sim/real transfer debt. V2 vs V3 = does simulation help or
hurt (relative to just training on clean Lead-I). V4 = how far sim-trained is
from the real-data ceiling.

Honesty: single seed; AF only (spatial classes MI/STTC/HYP have no clean CinC
mapping so this tests rhythm-pathology transfer specifically); CinC is handheld
lead-I (cleaner than wrist dry-electrode); class imbalance handled by AUROC
(imbalance-robust) + balanced training sampler.
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
from src.watch_simulator import simulate_watch, WatchSimConfig
import importlib.util

_spec6 = importlib.util.spec_from_file_location(
    "e6", Path(__file__).resolve().parents[1] / "experiments" / "06_sim_validation.py")
e6 = importlib.util.module_from_spec(_spec6); _spec6.loader.exec_module(e6)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "06b_classifier_crossover"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"
torch.manual_seed(0); np.random.seed(0)

WATCH_CFG = WatchSimConfig(FS, fs_watch=FS, seed=None)
CLEAN_CFG = WatchSimConfig(FS, fs_watch=FS, apply_bandwidth=False, apply_electrode=False,
                           apply_noise=False, apply_quantization=False, seed=0)


# ---------------------------------------------------------------------------
# PTB-XL binary AF/NORM loader
# ---------------------------------------------------------------------------

def load_ptbxl_binary(data_dir, max_per_class=700):
    """Return train/test lists of (ecg[T,12], label 0=NORM/1=AF, ecg_id).
    Uses strat_fold 1-8 train, 9-10 test."""
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
            rec_path = str(data_dir / row["filename_lr"])
            try: sig, _ = wfdb.rdsamp(rec_path)
            except: continue
            out[split].append({"ecg": sig.astype(np.float32), "y": int(row["y"]),
                               "ecg_id": int(row["ecg_id"])})
    return out


# ---------------------------------------------------------------------------
# CinC 2017 binary loader (real single-lead)
# ---------------------------------------------------------------------------

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
        # downsample 300->100
        from math import gcd
        g = gcd(int(fs), int(FS)); up = int(FS)//g; down = int(fs)//g
        sig = resample_poly(sig, up, down)
        rec = {"ecg": sig[:SIGLEN].astype(np.float32), "y": 1 if lab == "A" else 0}
        (A if lab == "A" else N).append(rec)
    rng = np.random.default_rng(0)
    rng.shuffle(A); rng.shuffle(N)
    return A[:n_per_class], N[:n_per_class]


# ---------------------------------------------------------------------------
# Datasets: PTB-XL (sim or clean) and CinC (real, as-is)
# ---------------------------------------------------------------------------

def _fixlen12(x):
    T = x.shape[0]
    return x[:SIGLEN] if T >= SIGLEN else np.concatenate([x, np.zeros((SIGLEN-T, x.shape[1]))], 0)

def _leadI_watch(x12, cfg, rng):
    x = _fixlen12(x12)
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    out = simulate_watch(x, FS, cfg, ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"], rng=rng)
    leadI = out["watch"]
    return (leadI - leadI.mean()) / (leadI.std() + 1e-6)

def _leadI_clean(x12):
    x = _fixlen12(x12)
    mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    leadI = x[:, 0]
    return (leadI - leadI.mean()) / (leadI.std() + 1e-6)


class PTBXLSim(Dataset):
    """1-lead sim-watch from PTB-XL. mode='sim' or 'clean'."""
    def __init__(self, records, mode="sim"):
        self.records = records; self.mode = mode
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        rng = np.random.default_rng(None)
        if self.mode == "sim":
            leadI = _leadI_watch(self.records[i]["ecg"], WATCH_CFG, rng)
        else:
            leadI = _leadI_clean(self.records[i]["ecg"])
        return (torch.from_numpy(leadI[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.records[i]["y"]))


class CinCReal(Dataset):
    def __init__(self, records): self.records = records
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = self.records[i]["ecg"]
        if x.shape[0] < SIGLEN:
            x = np.concatenate([x, np.zeros((SIGLEN - x.shape[0],))], 0)
        x = x[:SIGLEN]
        x = (x - x.mean()) / (x.std() + 1e-6)
        return (torch.from_numpy(x[:, None].astype(np.float32)).permute(1, 0),
                torch.tensor(self.records[i]["y"]))


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
def evaluate(model, ds, tag):
    model.eval()
    dl = DataLoader(ds, batch_size=64)
    logits = []; ys = []
    for xb, yb in dl:
        logits.append(model(xb).numpy()); ys.append(yb.numpy())
    logits = np.concatenate(logits); y = np.concatenate(ys)
    from sklearn.metrics import roc_auc_score, accuracy_score
    proba = torch.softmax(torch.from_numpy(logits), 1).numpy()[:, 1]
    auc = float(roc_auc_score(y, proba))
    acc = float(accuracy_score(y, logits.argmax(1)))
    n = len(y); npos = int(y.sum())
    print(f"  {tag}: AUROC={auc:.4f} ACC={acc:.4f} (n={n}, AF={npos})", flush=True)
    return {"auroc": auc, "acc": acc, "n": n, "n_af": npos}


def main():
    print("Loading PTB-XL binary AF/NORM (max_per_class=700) ...", flush=True)
    ptb = load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr, te = ptb["train"], ptb["test"]
    print(f"  PTB train={len(tr)} (AF={sum(r['y'] for r in tr)}) test={len(te)} (AF={sum(r['y'] for r in te)})", flush=True)

    print("Loading REAL CinC 2017 binary (N vs A, n_per_class=700) ...", flush=True)
    cA, cN = load_cinc_binary(n_per_class=700)
    cinc_all = cA + cN
    print(f"  CinC: AF={len(cA)} N={len(cN)} total={len(cinc_all)}", flush=True)
    # CinC train/test split (for V4 oracle): 80/20
    rng = np.random.default_rng(0); idx = np.arange(len(cinc_all)); rng.shuffle(idx)
    cut = int(0.8 * len(cinc_all))
    cinc_train = [cinc_all[i] for i in idx[:cut]]; cinc_test = [cinc_all[i] for i in idx[cut:]]

    results = {}

    # V1+V2: sim-trained model, tested on sim + real
    print("\n=== V1/V2: train on PTB-XL sim-watch ===", flush=True)
    m = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m, PTBXLSim(tr, "sim"), tag="sim-train")
    results["V1_sim_to_sim"] = evaluate(m, PTBXLSim(te, "sim"), "V1 sim->sim")
    results["V2_sim_to_realCinC"] = evaluate(m, CinCReal(cinc_all), "V2 sim->real CinC")

    # V3: clean-LeadI-trained, tested on real CinC
    print("\n=== V3: train on PTB-XL clean Lead-I ===", flush=True)
    m3 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m3, PTBXLSim(tr, "clean"), tag="clean-train")
    results["V3_clean_to_realCinC"] = evaluate(m3, CinCReal(cinc_all), "V3 clean->real CinC")

    # V4 oracle: real-CinC-trained
    print("\n=== V4 oracle: train on REAL CinC ===", flush=True)
    m4 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m4, CinCReal(cinc_train), tag="real-train")
    results["V4_realCinC_to_realCinC"] = evaluate(m4, CinCReal(cinc_test), "V4 real->real (oracle)")

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "classes": ["NORM", "AF"],
        "n_ptb_train": len(tr), "n_ptb_test": len(te),
        "n_cinc_total": len(cinc_all), "n_cinc_train": len(cinc_train), "n_cinc_test": len(cinc_test),
        "variants": results,
        "interpretation": {
            "sim_real_debt": "V1 - V2 (sim->sim vs sim->real gap)",
            "sim_vs_clean_for_real": "V2 vs V3 (does simulation help or hurt real transfer?)",
            "real_ceiling": "V4 (oracle upper bound with real target data)",
        },
        "honesty": ["single seed", "AF/rhythm only — spatial classes (MI/STTC/HYP) have no clean CinC mapping",
                    "CinC is handheld lead-I, cleaner than wrist dry-electrode",
                    "different populations (PTB-XL clinical vs CinC challenge cohort) confound the sim/real axis"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results)
    print("\nDONE.", flush=True)


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["V1_sim_to_sim", "V2_sim_to_realCinC", "V3_clean_to_realCinC", "V4_realCinC_to_realCinC"]
    labels = ["V1 sim->sim", "V2 sim->real", "V3 clean->real", "V4 real->real (oracle)"]
    aucs = [results[n]["auroc"] for n in names]
    cols = ["#4C72B0", "#C44E52", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(4), aucs, color=cols)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC (binary NORM vs AF)"); ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_title("E6b: classifier cross-over — does sim-training transfer to REAL single-lead?")
    for b, v in zip(bars, aucs):
        ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
    # annotate the gap
    gap = results["V1_sim_to_sim"]["auroc"] - results["V2_sim_to_realCinC"]["auroc"]
    ax.annotate(f"sim/real debt = {gap:.3f}", xy=(1, results["V2_sim_to_realCinC"]["auroc"]),
                xytext=(1.5, results["V1_sim_to_sim"]["auroc"]+0.02), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"))
    plt.tight_layout(); plt.savefig(RESULTS / "crossover.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
