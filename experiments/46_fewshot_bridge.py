#!/usr/bin/env python3
"""
E46 — The "minimal tuning" bridge: closed-loop calibration + k real labels.

North-star question: "train on clinical, use on Apple Watch with NO or MINIMAL
tuning, reach ~0.93." E42 answered the zero-label case (closed-loop 0.742 vs
clean 0.701 vs oracle 0.931 on real CinC). E46 answers the MINIMAL-TUNING case:
as we add k real labeled target examples (fine-tune), how fast does each arm
climb toward oracle — and does starting from the closed-loop-calibrated model
need FEWER real labels to hit a target than starting from clean?

Arms, for k in {0, 10, 25, 50, 100} real CinC labels:
  clean+k   : train clinical clean, then fine-tune on k real CinC
  closed+k  : train clinical closed-loop-calibrated, then fine-tune on k real CinC
Plus oracle (train on all ref labels). Test on held-out real CinC. 10 seeds
(k-sampling + split vary). Calibration uses unlabeled CinC bw (as before); the k
labels are the ONLY labeled target data used, mimicking "minimal tuning".

Key metric: labels-to-target — how many real labels each arm needs to reach,
e.g., 0.85 AUROC. If closed+k reaches it at smaller k, calibration SAVES labels.

HONEST FLAGS: CinC finger != AW wrist; AF/NORM easy; fine-tune on tiny k is high
variance (E30 showed <25 labels can HURT); single clinical train set.
"""
from __future__ import annotations
import json, sys, copy
from pathlib import Path
import numpy as np
import torch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "46_fewshot_bridge"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000
KS = [0, 10, 25, 50, 100]

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def finetune(model, sigs, ys, epochs=15, lr=5e-4):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    if len(sigs) == 0: return model
    dl = DataLoader(e25.SigDataset(sigs, ys), batch_size=min(32, len(sigs)), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr); lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
    return model

def sample_k(rng, ref, k):
    if k == 0: return [], []
    af = [r for r in ref if r["y"] == 1]; nn_ = [r for r in ref if r["y"] == 0]
    rng.shuffle(af); rng.shuffle(nn_)
    half = k // 2
    picks = af[:half] + nn_[:k - half]
    return [p["sig"] for p in picks], [p["y"] for p in picks]

def run_seed(seed, tr_leadI, tr_y, cinc):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc)//2
    ref = [cinc[i] for i in idx[:cut]]; test = [cinc[i] for i in idx[cut:]]
    test_sigs=[r["sig"] for r in test]; test_y=[r["y"] for r in test]
    tgt_bw = float(np.mean([signal_modality_stats(r["sig"], FS)["bw_energy"] for r in ref[:200]]))

    # base models (train once, then fine-tune copies per k)
    m_clean = make_model(); e25.train(m_clean, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    clc = ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug = [clc.generate(x) for x in tr_leadI]
    m_closed = make_model(); e25.train(m_closed, e25.SigDataset(aug, tr_y), epochs=20, tag=f"s{seed}-closed")

    res = {"clean": {}, "closed": {}}
    rk = np.random.default_rng(1000+seed)
    for k in KS:
        sk, yk = sample_k(rk, ref, k)
        mc = finetune(copy.deepcopy(m_clean), sk, yk)
        res["clean"][k] = e25.evaluate(mc, test_sigs, test_y)[0]
        mo = finetune(copy.deepcopy(m_closed), sk, yk)
        res["closed"][k] = e25.evaluate(mo, test_sigs, test_y)[0]
    # oracle
    mo = make_model(); e25.train(mo, e25.SigDataset([r["sig"] for r in ref],[r["y"] for r in ref]), epochs=20, tag=f"s{seed}-oracle")
    res["oracle"] = e25.evaluate(mo, test_sigs, test_y)[0]
    return res

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL n={len(tr_leadI)}  CinC n={len(cinc)}", flush=True)

    seeds=list(range(10)); per=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per.append(run_seed(s, tr_leadI, tr_y, cinc))

    def curve(arm): return {k: np.array([p[arm][k] for p in per]) for k in KS}
    cc = curve("clean"); cl = curve("closed")
    orc = np.array([p["oracle"] for p in per])

    print("\n===== AUROC vs k real labels (10 seeds) =====")
    print(f"  {'k':>4} {'clean+k':>14} {'closed+k':>14} {'Δ':>8}")
    for k in KS:
        d_ = cl[k].mean()-cc[k].mean()
        print(f"  {k:>4} {cc[k].mean():.3f}±{cc[k].std():.3f}  {cl[k].mean():.3f}±{cl[k].std():.3f}  {d_:+.3f}")
    print(f"  oracle = {orc.mean():.3f}")

    # labels-to-target
    def labels_to(curve_d, target):
        for k in KS:
            if curve_d[k].mean() >= target: return k
        return None
    for tgt in (0.80, 0.85, 0.90):
        kc = labels_to(cc, tgt); kl = labels_to(cl, tgt)
        print(f"  labels to reach {tgt:.2f}: clean={kc}  closed={kl}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.errorbar(KS, [cc[k].mean() for k in KS], yerr=[cc[k].std() for k in KS], marker="o", color="#999", label="clean + k labels", capsize=3)
    ax.errorbar(KS, [cl[k].mean() for k in KS], yerr=[cl[k].std() for k in KS], marker="s", color="#d4a017", label="closed-loop + k labels", capsize=3)
    ax.axhline(orc.mean(), color="r", ls="--", lw=1, label=f"oracle {orc.mean():.3f}")
    ax.set_xlabel("k = real labeled CinC examples (fine-tune)"); ax.set_ylabel("AUROC on real CinC")
    ax.set_title("E46 minimal-tuning bridge: labels vs calibration"); ax.legend(); ax.set_ylim(0.6,1.0)
    fig.tight_layout(); fig.savefig(RESULTS/"fewshot_bridge.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'fewshot_bridge.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"ks":KS,
                   "clean":{str(k):[float(cc[k].mean()),float(cc[k].std())] for k in KS},
                   "closed":{str(k):[float(cl[k].mean()),float(cl[k].std())] for k in KS},
                   "oracle":float(orc.mean()),
                   "labels_to_target":{f"{t:.2f}":{"clean":labels_to(cc,t),"closed":labels_to(cl,t)} for t in (0.80,0.85,0.90)},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy task","tiny-k fine-tune high variance (E30)","single clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
