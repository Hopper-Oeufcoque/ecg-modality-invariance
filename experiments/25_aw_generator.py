"""Experiment 25 — Validate the Apple-Watch generator (THE north-star tool).

GOAL: build a tool that turns abundant clinical Lead-I ECGs into Apple-Watch-style
training data, so models trained on it transfer BETTER to real Apple Watch than
models trained on raw clinical Lead-I.

THE ACCEPTANCE TEST (honest bar): train-on-generated must BEAT train-on-clean-
clinical-Lead-I on REAL data. E6b established the baseline: clean Lead-I -> real
CinC = 0.753, and the old heavy-noise simulator -> real = 0.737 (worse). E6c showed
real Apple Watch ~ real CinC (dist 0.247), so real CinC is the validated real-AW
proxy test set.

The generator (src/aw_generator.py) learns an empirical spectral transfer function
clinical->CinC and applies it as a zero-phase magnitude filter (preserves QRS
morphology = preserves label) + light calibrated noise (real AW is clean, E6c).

Binary NORM-vs-AF (clean cross-dataset labels, same as E6b/E23). All tested on
REAL CinC 2017 (validated AW proxy).

Variants (1-lead ECGResNet1d, 20 ep, single seed):
  V1 clean Lead-I           (E6b baseline = 0.753, the bar to beat)
  V2 old heavy-noise sim    (E6b = 0.737, the failed approach, for contrast)
  V3 GENERATED AW-style     (clinical -> learned-transfer -> AW-style)  <-- the test
  V4 clean + GENERATED aug  (50/50 mix — does generated data help as augmentation?)
  V5 real CinC -> real CinC (oracle upper bound, E6b V4 = 0.946)

Transfer function is FIT on a held-out CinC split (the "unlabeled real reference"
a practitioner would have), NOT on the CinC test split — no leakage of test signals.
Labels are never used from CinC for fitting the generator (unsupervised transfer).

Honesty: single seed; AF/NORM only; the transfer function is fit toward CinC (the
validated proxy), not real AW directly (HOME is eval-only); generator preserves
morphology by construction so labels stay valid.
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
from src.aw_generator import build_transfer_function, AWGenerator

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "25_aw_generator"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
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
            # store clean Lead-I (ch0), z-normed
            leadI = sig[:SIGLEN, 0].astype(np.float64)
            if leadI.size < SIGLEN:
                leadI = np.concatenate([leadI, np.zeros(SIGLEN - leadI.size)])
            leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-9)
            out[split].append({"leadI": leadI.astype(np.float32), "y": int(row["y"]),
                               "ecg12": sig.astype(np.float32)})
    return out


def load_cinc(n_per_class=700, fs=300):
    data_dir = Path.home() / "data" / "cinc2017" / "training2017"
    ref = {}
    for line in (data_dir / "REFERENCE.csv").read_text().splitlines():
        p = line.strip().split(",")
        if len(p) == 2: ref[p[0]] = p[1]
    import scipy.io as sio
    A = []; N = []
    from math import gcd
    g = gcd(int(fs), int(FS)); up = int(FS)//g; down = int(fs)//g
    for mf in sorted(data_dir.glob("A*.mat")):
        lab = ref.get(mf.stem, "O")
        if lab not in ("A", "N"): continue
        try: sig = sio.loadmat(mf)["val"][0].astype(np.float64)
        except: continue
        if sig.size < SIGLEN: continue
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        sig = resample_poly(sig, up, down)
        rec = {"sig": sig[:SIGLEN].astype(np.float32), "y": 1 if lab == "A" else 0}
        (A if lab == "A" else N).append(rec)
    return A[:n_per_class], N[:n_per_class]


class SigDataset(Dataset):
    """Generic 1-lead dataset from a list of (signal, y)."""
    def __init__(self, sigs, ys):
        self.sigs = sigs; self.ys = ys
    def __len__(self): return len(self.sigs)
    def __getitem__(self, i):
        x = self.sigs[i]
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.shape[0])])
        x = x[:SIGLEN]; x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x[:, None].astype(np.float32)).permute(1, 0), torch.tensor(self.ys[i])


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
def evaluate(model, sigs, ys):
    model.eval()
    dl = DataLoader(SigDataset(sigs, ys), batch_size=64)
    logits = []; yy = []
    for xb, yb in dl:
        logits.append(model(xb).numpy()); yy.append(yb.numpy())
    logits = np.concatenate(logits); y = np.concatenate(yy)
    from sklearn.metrics import roc_auc_score, accuracy_score
    proba = torch.softmax(torch.from_numpy(logits), 1).numpy()[:, 1]
    return float(roc_auc_score(y, proba)), float(accuracy_score(y, logits.argmax(1)))


def main():
    print("Loading PTB-XL binary AF/NORM ...", flush=True)
    ptb = load_ptbxl_binary(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr = ptb["train"]
    tr_leadI = [r["leadI"] for r in tr]; tr_y = [r["y"] for r in tr]
    print(f"  PTB train={len(tr)} (AF={sum(tr_y)})", flush=True)

    print("Loading REAL CinC (N vs A) ...", flush=True)
    cA, cN = load_cinc(n_per_class=700)
    cinc = cA + cN
    # split CinC: half as UNLABELED reference for fitting the transfer function,
    # half as the test set (no signal leakage between them)
    rng = np.random.default_rng(0); idx = np.arange(len(cinc)); rng.shuffle(idx)
    cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]      # "unlabeled real reference"
    cinc_test = [cinc[i] for i in idx[cut:]]     # held-out real test
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]
    print(f"  CinC ref (unlabeled)={len(cinc_ref)} test={len(cinc_test)} (AF={sum(test_y)})", flush=True)

    print("\n=== Fitting empirical transfer function (clinical Lead-I -> CinC) ===", flush=True)
    ref_sigs = [r["sig"] for r in cinc_ref]  # signals only; labels NOT used
    H = build_transfer_function(tr_leadI, ref_sigs, fs=FS, siglen=SIGLEN)
    print(f"  H(f): {H.size} bins, range [{H.min():.2f}, {H.max():.2f}], mean {H.mean():.2f}", flush=True)
    gen = AWGenerator(H, fs=FS, siglen=SIGLEN, noise_level=0.03, baseline_boost=0.04, seed=0)

    # Build training sets
    # V3 generated: apply generator to each clinical Lead-I
    gen_sigs = [gen.generate(s) for s in tr_leadI]
    # V2 old heavy-noise sim: apply the forward-physics sim to the 12-lead
    sim_cfg = WatchSimConfig(FS, fs_watch=FS, seed=None)
    LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    sim_sigs = []
    for r in tr:
        x = r["ecg12"][:SIGLEN]
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros((SIGLEN-x.shape[0],12))],0)
        mu = x.mean(0, keepdims=True); sd = x.std(0, keepdims=True)+1e-6; x=(x-mu)/sd
        o = simulate_watch(x, FS, sim_cfg, LEADS, rng=np.random.default_rng(None))
        s = o["watch"]; sim_sigs.append(((s-s.mean())/(s.std()+1e-6)).astype(np.float32))

    results = {}
    def run(tag, sigs, ys):
        print(f"\n=== {tag} ===", flush=True)
        m = ECGResNet1d(1, N_CLASSES).to(DEVICE)
        train(m, SigDataset(sigs, ys), tag=tag)
        auc, acc = evaluate(m, test_sigs, test_y)
        print(f"  {tag} -> real CinC: AUROC={auc:.4f} ACC={acc:.4f}", flush=True)
        results[tag] = {"auroc": auc, "acc": acc}

    run("V1_clean_leadI", tr_leadI, tr_y)
    run("V2_old_heavy_sim", sim_sigs, tr_y)
    run("V3_generated_AW", gen_sigs, tr_y)
    run("V4_clean+generated", tr_leadI + gen_sigs, tr_y + tr_y)

    # V5 oracle: real CinC -> real CinC (train on ref split, test on test split)
    print("\n=== V5_oracle_real ===", flush=True)
    m5 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    train(m5, SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), tag="V5")
    auc, acc = evaluate(m5, test_sigs, test_y)
    results["V5_oracle_real"] = {"auroc": auc, "acc": acc}
    print(f"  V5 oracle real->real: AUROC={auc:.4f} ACC={acc:.4f}", flush=True)

    metrics = {
        "fs": FS, "task": "binary NORM vs AF", "n_ptb_train": len(tr),
        "n_cinc_ref": len(cinc_ref), "n_cinc_test": len(cinc_test),
        "transfer_fn": {"bins": int(H.size), "min": float(H.min()), "max": float(H.max()), "mean": float(H.mean())},
        "variants": results,
        "acceptance_bar": "V3/V4 must beat V1 (clean Lead-I) on real CinC to be worth using",
        "e6b_reference": {"clean": 0.753, "old_sim": 0.737, "oracle": 0.946},
        "honesty": ["single seed", "AF/NORM only", "transfer fn fit toward CinC (validated real-AW proxy), not real AW directly",
                    "generator preserves morphology (zero-phase mag filter) so labels stay valid",
                    "CinC ref/test split prevents signal leakage; labels never used to fit generator"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(results, H)
    print("\nDONE.", flush=True)


def _plot(results, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["V1_clean_leadI", "V2_old_heavy_sim", "V3_generated_AW", "V4_clean+generated", "V5_oracle_real"]
    labels = ["V1 clean\nLead-I", "V2 old heavy\nsim", "V3 GENERATED\nAW-style", "V4 clean+\ngenerated", "V5 oracle\nreal→real"]
    aucs = [results[n]["auroc"] for n in names]
    cols = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#937860"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})
    bars = ax1.bar(range(5), aucs, color=cols)
    ax1.axhline(0.753, color="#4C72B0", ls="--", lw=1, label="clean Lead-I bar (E6b 0.753)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1)
    ax1.set_xticks(range(5)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("AUROC on REAL CinC (validated AW proxy)"); ax1.set_ylim(0.5, 1.0)
    ax1.set_title("E25: does the AW-generator beat clean Lead-I on real data?")
    ax1.legend(fontsize=8, loc="upper left")
    for b, v in zip(bars, aucs): ax1.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=9)
    # transfer function
    freqs = np.fft.rfftfreq(SIGLEN, 1/FS)
    ax2.plot(freqs, H, color="#55A868")
    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_xlabel("frequency (Hz)"); ax2.set_ylabel("gain H(f)")
    ax2.set_title("Learned clinical->CinC transfer function")
    ax2.set_xlim(0, 50)
    plt.tight_layout(); plt.savefig(RESULTS / "generator_validation.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
