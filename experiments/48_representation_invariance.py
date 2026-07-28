#!/usr/bin/env python3
"""
E48 — Representation-level modality invariance: can a LEARNED invariant feature
space beat the +0.041 augmentation ceiling (E42/E43)?

E43 argued pure input-augmentation tops out ~+0.041 because the residual gap is
in-band (QRS/mid), untouchable without breaking morphology. A different lever:
don't just augment — enforce that the ENCODER produces the SAME features for a
clinical sample and its modality-shifted version. Uses the closed-loop calibrator
to define two "views" (clean + wander-shifted) and adds a feature-consistency
(invariance) loss on top of classification.

Arms (train PTB-XL Lead-I AFIB/NORM, test real held-out CinC AF/N, 20 seeds):
  clean         : clinical clean (floor)
  closed_aug    : E42 winner — train on closed-loop-calibrated data (implicit)
  invariance    : two-view CE + lambda * ||feat(clean) - feat(shifted)||^2 (explicit)
Calibration/shift uses UNLABELED CinC bw only. Compare vs the +0.041 ceiling.

HONEST FLAGS: CinC finger != AW wrist; AF/NORM easy task; single clinical train
set; invariance lambda picked a priori (0.5), not tuned on test.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "48_representation_invariance"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.5

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def feats(model, x):
    """Everything before the final Linear: stem -> blocks -> BN/ReLU/pool/flatten."""
    x = model.stem(x); x = model.blocks(x)
    h = model.head
    for layer in list(h)[:-1]:   # all but final Linear (incl. dropout — fine, .eval used at test)
        x = layer(x)
    return x  # (B, base_ch)

def logits_from_feat(model, f):
    return list(model.head)[-1](f)

class PairDataset(Dataset):
    """Yields (clean, shifted, y) — shifted via closed-loop calibrator each epoch."""
    def __init__(self, sigs, ys, calib):
        self.sigs = sigs; self.ys = ys; self.calib = calib
    def __len__(self): return len(self.sigs)
    def _norm(self, x):
        x = np.asarray(x, np.float64)
        if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN-x.shape[0])])
        x = x[:SIGLEN]; return (x-x.mean())/(x.std()+1e-6)
    def __getitem__(self, i):
        clean = self._norm(self.sigs[i])
        shifted = self.calib.generate(self.sigs[i])
        return (torch.from_numpy(clean[None].astype(np.float32)),
                torch.from_numpy(np.asarray(shifted)[None].astype(np.float32)),
                torch.tensor(self.ys[i]))

def train_invariance(model, sigs, ys, calib, epochs=20, lr=1e-3, lam=LAMBDA, tag="inv"):
    dl = DataLoader(PairDataset(sigs, ys, calib), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); tot=0.0; nb=0
        for xc, xs, yb in dl:
            opt.zero_grad()
            fc = feats(model, xc); fs = feats(model, xs)
            lc = logits_from_feat(model, fc); ls = logits_from_feat(model, fs)
            loss = ce(lc, yb) + ce(ls, yb) + lam * ((fc - fs) ** 2).mean()
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return model

def run_seed(seed, tr, tr_y, cinc):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw, tr, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    aug=[clc.generate(x) for x in tr]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); train_invariance(m, tr, tr_y, clc, epochs=20, tag=f"s{seed}-inv"); out["invariance"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL n={len(tr)}  CinC n={len(cinc)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc)

    arms=["clean","closed_aug","invariance"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b): 
        t,p=sst.ttest_rel(x,b); return (x-b).mean(),int((x>b).sum()),float(p)
    da=pair(A["closed_aug"],A["clean"]); di=pair(A["invariance"],A["clean"]); dv=pair(A["invariance"],A["closed_aug"])
    print(f"\n  closed_aug − clean : Δ={da[0]:+.3f} wins {da[1]}/20 p={da[2]:.4f}")
    print(f"  invariance − clean : Δ={di[0]:+.3f} wins {di[1]}/20 p={di[2]:.4f}")
    print(f"  invariance − closed_aug (does explicit beat implicit?): Δ={dv[0]:+.3f} wins {dv[1]}/20 p={dv[2]:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=["#999","#d4a017","#4a7"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E48 learned invariance vs augmentation: inv−aug Δ={dv[0]:+.3f} (p={dv[2]:.2f})")
    fig.tight_layout(); fig.savefig(RESULTS/"representation_invariance.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'representation_invariance.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"lambda":LAMBDA,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "closed_aug_vs_clean":{"delta":da[0],"wins":da[1],"p":da[2]},
                   "invariance_vs_clean":{"delta":di[0],"wins":di[1],"p":di[2]},
                   "invariance_vs_aug":{"delta":dv[0],"wins":dv[1],"p":dv[2]},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy task","single clinical train set","lambda=0.5 a priori not tuned"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
