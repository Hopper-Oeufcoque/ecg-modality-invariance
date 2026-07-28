#!/usr/bin/env python3
"""
E54 — λ / temperature sensitivity of the E51 headline (label-anchored alignment).

E51's win (joint 0.807, +0.106 vs clean) used λ=0.1, temp=0.1 chosen A-PRIORI.
Before building further on it, confirm it is not a lucky untuned point. Sweep the
two hyperparameters of the alignment term and check the win is BROAD (a plateau),
not a fragile spike.

Grid: λ ∈ {0.03, 0.1, 0.3, 1.0} × temp ∈ {0.05, 0.1, 0.2}, 10 seeds/cell (AF task,
the headline task). clean + closed_aug refs (seed-matched). Report AUROC surface +
whether every cell still beats calibration (0.742) and clean (0.701).

Read:
  most/all cells > calibration -> win is ROBUST to hyperparams (plateau)
  only λ=0.1,temp=0.1 wins       -> fragile / overfit hyperparams (weakens headline)

HONEST FLAGS: 10 seeds/cell (vs 20 in E51 — CPU budget for a 12-cell grid); CinC
finger != AW wrist; AF/NORM easy task; single clinical train set; same 3-device caveat.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

RESULTS = ROOT / "results" / "54_lambda_temp_sweep"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000
LAMBDAS = [0.03, 0.1, 0.3, 1.0]
TEMPS   = [0.05, 0.1, 0.2]
N_SEEDS = 10

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def train_joint_ht(sigs, ys, pairs, lam, temp, epochs=20, lr=1e-3, seed=0, tag="j"):
    """E51's joint loop, but λ AND temp parameterized."""
    torch.manual_seed(seed); np.random.seed(seed)
    m=make_model(); proj=nn.Sequential(nn.Linear(32,32),nn.ReLU(),nn.Linear(32,32))
    lab_dl=DataLoader(e51.LabeledDS(sigs,ys),batch_size=64,shuffle=True,drop_last=True)
    pair_dl=DataLoader(e50.PairContrastDS(pairs,seed=seed),batch_size=64,shuffle=True,drop_last=True)
    opt=torch.optim.Adam(list(m.parameters())+list(proj.parameters()),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); proj.train(); pit=iter(pair_dl); tot=0.0; nb=0
        for xb,yb in lab_dl:
            try: xc,xa=next(pit)
            except StopIteration: pit=iter(pair_dl); xc,xa=next(pit)
            opt.zero_grad()
            loss_ce=ce(m(xb),yb)
            zc=proj(e50.encode(m,xc)); za=proj(e50.encode(m,xa))
            loss=loss_ce+lam*e50.info_nce(zc,za,temp=temp)
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
    return m

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  SJLIFE={len(pairs)} PTB-XL={len(tr)} CinC={len(cinc)}", flush=True)

    seeds=list(range(N_SEEDS))
    # per-seed fixed test split + calibrator (shared across all cells for fairness)
    refs={}; tests={}; cals={}
    for s in seeds:
        rng=np.random.default_rng(s)
        idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
        ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
        refs[s]=ref; tests[s]=(([r["sig"] for r in test]),([r["y"] for r in test]))
        tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
        cals[s]=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=s,n_probe=40)

    # references
    clean=[]; aug=[]
    for s in seeds:
        ts,ty=tests[s]
        torch.manual_seed(s); np.random.seed(s)
        m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{s}-clean"); clean.append(e25.evaluate(m,ts,ty)[0])
        a=[cals[s].generate(x) for x in tr]
        m=make_model(); e25.train(m,e25.SigDataset(a,tr_y),epochs=20,tag=f"s{s}-aug"); aug.append(e25.evaluate(m,ts,ty)[0])
    clean=np.array(clean); aug=np.array(aug)
    print(f"\n  clean={clean.mean():.3f}  closed_aug={aug.mean():.3f}", flush=True)

    grid={}
    for lam in LAMBDAS:
        for temp in TEMPS:
            vals=[]
            for s in seeds:
                ts,ty=tests[s]
                m=train_joint_ht(tr,tr_y,pairs,lam,temp,seed=s,tag=f"L{lam}T{temp}s{s}")
                vals.append(e25.evaluate(m,ts,ty)[0])
            vals=np.array(vals); grid[(lam,temp)]=vals
            print(f"  λ={lam:<4} temp={temp:<4} joint={vals.mean():.3f}  Δvs_aug={vals.mean()-aug.mean():+.3f}  Δvs_clean={vals.mean()-clean.mean():+.3f}", flush=True)

    # summary
    beats_aug=sum(1 for k in grid if grid[k].mean()>aug.mean())
    beats_clean=sum(1 for k in grid if grid[k].mean()>clean.mean())
    best=max(grid, key=lambda k: grid[k].mean())
    print(f"\n  cells beating calibration: {beats_aug}/{len(grid)}   beating clean: {beats_clean}/{len(grid)}")
    print(f"  best cell: λ={best[0]} temp={best[1]} → {grid[best].mean():.3f}   (E51 default λ=0.1,temp=0.1 → {grid[(0.1,0.1)].mean():.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M=np.array([[grid[(l,t)].mean() for t in TEMPS] for l in LAMBDAS])
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    im=ax.imshow(M,cmap="viridis",aspect="auto",vmin=min(clean.mean(),M.min()),vmax=M.max())
    ax.set_xticks(range(len(TEMPS))); ax.set_xticklabels([f"temp={t}" for t in TEMPS])
    ax.set_yticks(range(len(LAMBDAS))); ax.set_yticklabels([f"λ={l}" for l in LAMBDAS])
    for i in range(len(LAMBDAS)):
        for j in range(len(TEMPS)):
            ax.text(j,i,f"{M[i,j]:.3f}",ha="center",va="center",color="w",fontsize=10)
    ax.set_title(f"E54 joint AUROC surface ({N_SEEDS} seeds)  clean={clean.mean():.3f} calib={aug.mean():.3f}\nall {beats_aug}/{len(grid)} cells beat calibration")
    fig.colorbar(im,label="AUROC real CinC"); fig.tight_layout()
    fig.savefig(RESULTS/"lambda_temp_surface.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'lambda_temp_surface.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":N_SEEDS,"lambdas":LAMBDAS,"temps":TEMPS,
                   "clean":float(clean.mean()),"closed_aug":float(aug.mean()),
                   "grid":{f"lam{l}_temp{t}":float(grid[(l,t)].mean()) for l in LAMBDAS for t in TEMPS},
                   "grid_std":{f"lam{l}_temp{t}":float(grid[(l,t)].std()) for l in LAMBDAS for t in TEMPS},
                   "cells_beating_calibration":beats_aug,"cells_beating_clean":beats_clean,"n_cells":len(grid),
                   "best_cell":{"lambda":best[0],"temp":best[1],"auroc":float(grid[best].mean())},
                   "default_cell_auroc":float(grid[(0.1,0.1)].mean()),
                   "honesty":["10 seeds/cell (vs 20 in E51)","CinC finger != AW wrist","AF/NORM easy",
                              "single clinical train set","3 devices"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
