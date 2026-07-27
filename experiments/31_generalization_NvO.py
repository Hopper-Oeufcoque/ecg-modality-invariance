"""Experiment 31 — Does the gap ladder GENERALIZE beyond AF/NORM?

E26/E27b/E30 all used binary AF-vs-NORM (the cleanest cross-dataset label). The
biggest honesty risk: maybe the whole ladder (augmentation helps ~0.80 ceiling;
few-shot knee at k~50) is specific to AF's very distinct signature. E31 re-runs
the key rungs on a DIFFERENT, harder task: Normal vs OTHER-abnormal-rhythm.

Task: CinC 2017 "N" (normal) vs "O" (other rhythm — a heterogeneous abnormal
class, NOT AF, NOT noise). Source (clinical): PTB-XL NORM vs non-NORM/non-AFIB
abnormal records (analogous "normal vs generic-abnormal" grouping). This is a
fuzzier, messier label than AF/NORM by design — a stress test of generality.

Arms (5 seeds, 1-lead ResNet, tested on real CinC N-vs-O):
  clean Lead-I
  augment (strength 1.5, 5x = E27b locked recipe)
  augment-pretrain + finetune k=50 real
  oracle (real N-vs-O)

Question: does augmentation still beat clean, and does k=50 still close a big
chunk of the gap, on a task with a much less distinctive signature?

Honesty: 5 seeds, n varies by CinC class availability, CinC = E6c AW proxy,
label alignment clinical<->CinC is approximate for the "Other/abnormal" class
(flagged) — this is a generality stress test, not a clean benchmark.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import ECGResNet1d
from src.aw_generator import StochasticAWAugmenter
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "e25", Path(__file__).resolve().parents[1] / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "31_generalization_NvO"
RESULTS.mkdir(parents=True, exist_ok=True)

FS = 100.0; SIGLEN = 1000; N_CLASSES = 2
DEVICE = "cpu"


def load_ptbxl_normal_vs_abnormal(data_dir, max_per_class=700):
    """PTB-XL NORM (y=0) vs non-NORM & non-AFIB abnormal (y=1) — generic abnormal."""
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    def codes(s):
        try: return ast.literal_eval(s)
        except: return {}
    scp = df["scp_codes"].apply(codes)
    is_norm = scp.apply(lambda c: "NORM" in c)
    is_afib = scp.apply(lambda c: "AFIB" in c or "AFLT" in c)
    has_any = scp.apply(lambda c: len(c) > 0)
    norm = df[is_norm & ~is_afib].copy(); norm["y"] = 0
    abn = df[~is_norm & ~is_afib & has_any].copy(); abn["y"] = 1  # abnormal, not AF
    norm = norm.sample(frac=1.0, random_state=0).head(max_per_class)
    abn = abn.sample(frac=1.0, random_state=0).head(max_per_class)
    import wfdb
    out = []
    for sub in (norm, abn):
        for _, row in sub.iterrows():
            try: sig, _ = wfdb.rdsamp(str(data_dir / row["filename_lr"]))
            except: continue
            leadI = sig[:SIGLEN, 0].astype(np.float64)
            if leadI.size < SIGLEN: leadI = np.concatenate([leadI, np.zeros(SIGLEN - leadI.size)])
            leadI = (leadI - leadI.mean()) / (leadI.std() + 1e-9)
            out.append({"leadI": leadI.astype(np.float32), "y": int(row["y"])})
    return out


def load_cinc_normal_vs_other(n_per_class=700, fs=300):
    data_dir = Path.home() / "data" / "cinc2017" / "training2017"
    ref = {}
    for line in (data_dir / "REFERENCE.csv").read_text().splitlines():
        p = line.strip().split(",")
        if len(p) == 2: ref[p[0]] = p[1]
    import scipy.io as sio
    from math import gcd
    g = gcd(int(fs), int(FS)); up = int(FS)//g; down = int(fs)//g
    N = []; O = []
    for mf in sorted(data_dir.glob("A*.mat")):
        lab = ref.get(mf.stem, "")
        if lab not in ("N", "O"): continue
        try: sig = sio.loadmat(mf)["val"][0].astype(np.float64)
        except: continue
        if sig.size < SIGLEN: continue
        sig = (sig - sig.mean()) / (sig.std() + 1e-9)
        sig = resample_poly(sig, up, down)
        rec = {"sig": sig[:SIGLEN].astype(np.float32), "y": 0 if lab == "N" else 1}
        (N if lab == "N" else O).append(rec)
    return N[:n_per_class], O[:n_per_class]


def run_seed(seed, tr_leadI, tr_y, cinc):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc) // 2
    cinc_ref = [cinc[i] for i in idx[:cut]]; cinc_test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in cinc_test]; test_y = [r["y"] for r in cinc_test]

    out = {}

    # clean
    m0 = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(m0, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"] = e25.evaluate(m0, test_sigs, test_y)[0]

    # augment (locked recipe s1.5, 5x)
    aug = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=1.5, seed=seed)
    src = list(tr_leadI); sy = list(tr_y)
    for _ in range(5):
        src += [aug.generate(s) for s in tr_leadI]; sy += list(tr_y)
    mA = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mA, e25.SigDataset(src, sy), epochs=20, tag=f"s{seed}-aug")
    out["augment"] = e25.evaluate(mA, test_sigs, test_y)[0]

    # augment-pretrain + finetune k=50
    import random
    rr = list(cinc_ref); random.Random(seed).shuffle(rr)
    pos = [r for r in rr if r["y"] == 1][:25]; neg = [r for r in rr if r["y"] == 0][:25]
    ks_sigs = [r["sig"] for r in pos + neg]; ks_y = [r["y"] for r in pos + neg]
    mF = copy.deepcopy(mA)
    e25.train(mF, e25.SigDataset(ks_sigs, ks_y), epochs=15, lr=5e-4, batch_size=32, tag=f"s{seed}-ft50")
    out["finetune_k50"] = e25.evaluate(mF, test_sigs, test_y)[0]

    # oracle
    mO = ECGResNet1d(1, N_CLASSES).to(DEVICE)
    e25.train(mO, e25.SigDataset([r["sig"] for r in cinc_ref], [r["y"] for r in cinc_ref]), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(mO, test_sigs, test_y)[0]

    for k, v in out.items(): print(f"  seed {seed} {k}: {v:.4f}", flush=True)
    return out


def main():
    print("Loading PTB-XL NORM vs abnormal(non-AF) ...", flush=True)
    tr = load_ptbxl_normal_vs_abnormal(Path.home() / "data" / "ptbxl", max_per_class=700)
    tr_leadI = [r["leadI"] for r in tr]; tr_y = [r["y"] for r in tr]
    print(f"  train={len(tr)} (abn={sum(tr_y)})", flush=True)
    print("Loading REAL CinC N vs O ...", flush=True)
    cN, cO = load_cinc_normal_vs_other(n_per_class=700); cinc = cN + cO
    print(f"  CinC N={len(cN)} O={len(cO)}", flush=True)

    seeds = [0, 1, 2, 3, 4]
    per_seed = {}
    for s in seeds:
        print(f"\n===== SEED {s} =====", flush=True)
        per_seed[s] = run_seed(s, tr_leadI, tr_y, cinc)

    keys = ["clean", "augment", "finetune_k50", "oracle"]
    agg = {}
    clean = np.array([per_seed[s]["clean"] for s in seeds])
    for k in keys:
        vals = np.array([per_seed[s][k] for s in seeds])
        agg[k] = {"mean": float(vals.mean()), "std": float(vals.std()), "vals": vals.tolist()}
        if k not in ("clean",):
            agg[k]["delta_vs_clean_mean"] = float((vals - clean).mean())
            agg[k]["delta_positive_in_seeds"] = int(((vals - clean) > 0).sum())

    print("\n===== AGGREGATE (mean±std, 5 seeds) — Normal vs Other =====", flush=True)
    for k in keys:
        extra = ""
        if "delta_vs_clean_mean" in agg[k]:
            extra = f"  Δvs_clean={agg[k]['delta_vs_clean_mean']:+.4f} ({agg[k]['delta_positive_in_seeds']}/5)"
        print(f"  {k:14s} {agg[k]['mean']:.4f}±{agg[k]['std']:.4f}{extra}", flush=True)

    metrics = {
        "fs": FS, "task": "binary Normal vs Other (generalization stress test)", "seeds": seeds,
        "aggregate": agg, "per_seed": per_seed,
        "question": "does the augmentation-helps + few-shot-knee ladder generalize beyond AF/NORM?",
        "honesty": ["5 seeds", "CinC N-vs-O; O is heterogeneous abnormal", "AF-vs-NORM was cleaner",
                    "clinical<->CinC label alignment approximate for abnormal class", "CinC = E6c AW proxy"],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved -> {RESULTS/'metrics.json'}", flush=True)
    _plot(agg, keys)
    print("\nDONE.", flush=True)


def _plot(agg, keys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["clean\nLead-I", "augment\n(s1.5,5x)", "aug+finetune\nk=50", "oracle\nreal"]
    means = [agg[k]["mean"] for k in keys]; stds = [agg[k]["std"] for k in keys]
    cols = ["#4C72B0", "#55A868", "#DD8452", "#937860"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(range(len(keys)), means, yerr=stds, capsize=5, color=cols)
    ax.axhline(means[0], color="#4C72B0", ls="--", lw=1, label=f"clean bar ({means[0]:.3f})")
    ax.axhline(means[-1], color="#937860", ls=":", lw=1, label=f"oracle ({means[-1]:.3f})")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC on real CinC N-vs-O (5 seeds)"); ax.set_ylim(0.5, 1.0)
    ax.set_title("E31: does the ladder generalize? (Normal vs Other rhythm)")
    ax.legend(fontsize=8, loc="upper left")
    for b, m, s in zip(bars, means, stds):
        ax.text(b.get_x()+b.get_width()/2, m+s+0.01, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(RESULTS / "generalization.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
