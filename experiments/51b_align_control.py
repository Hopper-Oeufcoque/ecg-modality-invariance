#!/usr/bin/env python3
"""
E51b — CONTROL for E51's large positive (joint 0.807, +0.106 vs clean, 20/20).

E51 added a label-anchored SJLIFE modality-alignment term to CE and jumped +0.106
with a variance collapse (std 0.048->0.023). That signature (big mean gain + low
variance) is a classic REGULARIZER footprint, which raises a confound: is the gain
from SAME-PATIENT modality alignment (true invariance), or just from bolting ANY
auxiliary InfoNCE loss on extra real ECG data (generic SSL regularization)?

This decomposes the mechanism with matched controls (identical joint loop, only the
alignment POSITIVE-PAIR definition changes):
  joint          : correct SJLIFE pairs (clin_i <-> apple_i)  [= E51, reproduce]
  joint_shuffled : MISMATCHED pairs (clin_i <-> apple_{perm(i)}, different patient)
                   -> keeps aux real-ECG InfoNCE, DESTROYS same-patient correspondence
  joint_selfclin : positive = (clinical Lead-I, its OWN closed-loop-calibrated view)
                   -> NO SJLIFE watch data at all; pure within-modality consistency

Reads:
  joint ~= joint_shuffled  -> effect is generic aux-ECG regularization (watch corresp. irrelevant)
  joint  >  joint_shuffled -> same-patient modality correspondence matters (TRUE invariance)
  joint_selfclin captures it -> you don't even need SJLIFE (augmentation-consistency suffices)

Test real CinC AF/N, 20 seeds, lambda=0.1, temp=0.1. Same clean/closed_aug refs.
HONEST FLAGS: same as E51 + shuffle uses a fixed derangement per seed.
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
_s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)
_s51 = importlib.util.spec_from_file_location("e51", ROOT / "experiments" / "51_label_anchored_align.py")
e51 = importlib.util.module_from_spec(_s51); _s51.loader.exec_module(e51)

RESULTS = ROOT / "results" / "51b_align_control"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.1; TEMP = 0.1

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def _norm(x):
    x=np.asarray(x,np.float64)
    if x.shape[0]<SIGLEN: x=np.concatenate([x,np.zeros(SIGLEN-x.shape[0])])
    return ((x[:SIGLEN]-x[:SIGLEN].mean())/(x[:SIGLEN].std()+1e-6)).astype(np.float32)

class ShuffledPairDS(Dataset):
    """clin_i paired with apple windows from a DIFFERENT patient (fixed derangement)."""
    def __init__(self, pairs, seed=0):
        self.pairs=pairs; self.rng=np.random.default_rng(seed)
        n=len(pairs); perm=self.rng.permutation(n)
        # ensure derangement (no i maps to itself)
        for i in range(n):
            if perm[i]==i:
                j=(i+1)%n; perm[i],perm[j]=perm[j],perm[i]
        self.perm=perm
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        clin,_=self.pairs[i]
        _,aws=self.pairs[self.perm[i]]
        aw=aws[self.rng.integers(len(aws))]
        return torch.from_numpy(clin[None]), torch.from_numpy(aw[None])

class SelfClinDS(Dataset):
    """positive = (clinical Lead-I, its own closed-loop-calibrated view). No watch data."""
    def __init__(self, sigs, calib, seed=0):
        self.sigs=sigs; self.calib=calib
    def __len__(self): return len(self.sigs)
    def __getitem__(self,i):
        base=_norm(self.sigs[i])
        shifted=self.calib.generate(self.sigs[i])
        return torch.from_numpy(base[None]), torch.from_numpy(_norm(shifted)[None])

def train_joint_generic(sigs, ys, pair_ds, lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="j"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=make_model(); proj=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,32))
    lab_dl=DataLoader(e51.LabeledDS(sigs,ys),batch_size=64,shuffle=True,drop_last=True)
    pair_dl=DataLoader(pair_ds,batch_size=64,shuffle=True,drop_last=True)
    opt=torch.optim.Adam(list(m.parameters())+list(proj.parameters()),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); proj.train(); pit=iter(pair_dl); tot=0.0; nb=0
        for xb,yb in lab_dl:
            try: xc,xa=next(pit)
            except StopIteration: pit=iter(pair_dl); xc,xa=next(pit)
            opt.zero_grad()
            loss_ce=ce(m(xb),yb)
            zc=proj(e50.encode(m,xc)); za=proj(e50.encode(m,xa))
            loss=loss_ce+lam*e50.info_nce(zc,za,temp=TEMP)
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f}",flush=True)
    return m

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw, tr, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint_generic(tr,tr_y,e50.PairContrastDS(pairs,seed=seed),seed=seed,tag=f"s{seed}-joint"); out["joint"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint_generic(tr,tr_y,ShuffledPairDS(pairs,seed=seed),seed=seed,tag=f"s{seed}-shuf"); out["joint_shuffled"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint_generic(tr,tr_y,SelfClinDS(tr,clc,seed=seed),seed=seed,tag=f"s{seed}-self"); out["joint_selfclin"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading SJLIFE pairs + PTB-XL + CinC ...",flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl",max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  SJLIFE={len(pairs)} PTB-XL={len(tr)} CinC={len(cinc)} lambda={LAMBDA}",flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==",flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","joint","joint_shuffled","joint_selfclin"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:16s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "joint_vs_clean":            pair(A["joint"],A["clean"]),
      "shuffled_vs_clean":         pair(A["joint_shuffled"],A["clean"]),
      "selfclin_vs_clean":         pair(A["joint_selfclin"],A["clean"]),
      "joint_vs_shuffled":         pair(A["joint"],A["joint_shuffled"]),
      "joint_vs_selfclin":         pair(A["joint"],A["joint_selfclin"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
           color=["#999","#3577c2","#c23577","#4a7"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E51b control: joint−shuffled Δ={cmp['joint_vs_shuffled'][0]:+.3f} p={cmp['joint_vs_shuffled'][2]:.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"align_control.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'align_control.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"lambda":LAMBDA,"temp":TEMP,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["shuffled=fixed derangement per seed","selfclin uses no SJLIFE watch",
                              "3 devices","lambda=0.1 a priori","AF/NORM easy","single clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
