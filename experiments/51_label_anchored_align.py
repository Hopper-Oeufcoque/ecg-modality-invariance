#!/usr/bin/env python3
"""
E51 — LABEL-ANCHORED modality alignment: the fix E48/E49/E50 pointed to.

E48 (explicit invariance loss), E49 (clinical distillation), E50 (real-paired
contrastive pretraining) ALL lost to plain calibration. Diagnosis: any
representation trick NOT anchored to the label trades away discriminative
morphology (E50 literally destroyed pathology content to satisfy alignment).

Fix: align modalities and classify JOINTLY in the same step, so the CE term
protects the label signal while a modality-alignment term pulls clinical & watch
together. The classification loss forbids the encoder from collapsing onto
label-destroying invariance.

Joint objective (per step): CE(PTB-XL Lead-I labels) + lambda * InfoNCE(real
SJLIFE clinical<->watch pairs). Shared single-lead encoder. We sweep lambda small
(light alignment regularizer) since E50 showed heavy label-free alignment destroys.

Arms (test real CinC AF/N, 20 seeds):
  clean        : CE only (floor)
  closed_aug   : E42 winner — calibrated Lead-I, CE only
  joint        : CE(clean Lead-I) + lambda*align(SJLIFE)         [lambda=0.1]
  joint_aug    : CE(calibrated Lead-I) + lambda*align(SJLIFE)    [does align add over aug?]

HONEST FLAGS: SJLIFE align term is label-free (no disease labels exist) but now
GATED by the CE anchor; 3 devices (CinC finger != Apple wrist != SJLIFE wrist);
lambda=0.1 a-priori; AF/NORM easy; single clinical train set; 20 seeds.
"""
from __future__ import annotations
import os, glob, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)
# reuse E50's SJLIFE loader + InfoNCE + encode
_s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)

RESULTS = ROOT / "results" / "51_label_anchored_align"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.1; TEMP = 0.1

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

class LabeledDS(Dataset):
    def __init__(self, sigs, ys): self.sigs=sigs; self.ys=ys
    def __len__(self): return len(self.sigs)
    def _n(self,x):
        x=np.asarray(x,np.float64)
        if x.shape[0]<SIGLEN: x=np.concatenate([x,np.zeros(SIGLEN-x.shape[0])])
        return ((x[:SIGLEN]-x[:SIGLEN].mean())/(x[:SIGLEN].std()+1e-6)).astype(np.float32)
    def __getitem__(self,i):
        return torch.from_numpy(self._n(self.sigs[i])[None]), torch.tensor(self.ys[i])

def train_joint(sigs, ys, pairs, lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="joint"):
    torch.manual_seed(seed); np.random.seed(seed)
    m = make_model()
    proj = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
    lab_dl = DataLoader(LabeledDS(sigs, ys), batch_size=64, shuffle=True, drop_last=True)
    pair_ds = e50.PairContrastDS(pairs, seed=seed)
    pair_dl = DataLoader(pair_ds, batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(list(m.parameters())+list(proj.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); proj.train()
        pit = iter(pair_dl); tot=0.0; nb=0
        for xb, yb in lab_dl:
            try: xc, xa = next(pit)
            except StopIteration:
                pit = iter(pair_dl); xc, xa = next(pit)
            opt.zero_grad()
            # classification (label anchor) — full forward
            logits = m(xb)
            loss_ce = ce(logits, yb)
            # modality alignment on real pairs (gated by lambda)
            zc = proj(e50.encode(m, xc)); za = proj(e50.encode(m, xa))
            loss_al = e50.info_nce(zc, za, temp=TEMP)
            loss = loss_ce + lam*loss_al
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    return m

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw, tr, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint(tr, tr_y, pairs, epochs=20, seed=seed, tag=f"s{seed}-joint"); out["joint"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint(aug, tr_y, pairs, epochs=20, seed=seed, tag=f"s{seed}-joint_aug"); out["joint_aug"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading SJLIFE pairs + PTB-XL + CinC ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  SJLIFE pairs={len(pairs)}  PTB-XL n={len(tr)}  CinC n={len(cinc)}  lambda={LAMBDA}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","closed_aug","joint","joint_aug"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean": pair(A["closed_aug"],A["clean"]),
      "joint_vs_clean":      pair(A["joint"],A["clean"]),
      "joint_vs_aug":        pair(A["joint"],A["closed_aug"]),
      "joint_aug_vs_aug":    pair(A["joint_aug"],A["closed_aug"]),
      "joint_aug_vs_clean":  pair(A["joint_aug"],A["clean"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
           color=["#999","#d4a017","#4a7","#3577c2"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["closed_aug"].mean(),color="#d4a017",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E51 label-anchored align (λ={LAMBDA})  joint−aug Δ={cmp['joint_vs_aug'][0]:+.3f} p={cmp['joint_vs_aug'][2]:.2f}")
    fig.tight_layout(); fig.savefig(RESULTS/"label_anchored_align.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'label_anchored_align.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"lambda":LAMBDA,"temp":TEMP,"sjlife_patients":len(pairs),
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["SJLIFE align term label-free but gated by CE anchor",
                              "3 devices: CinC finger != AW wrist != SJLIFE wrist",
                              "lambda=0.1 a priori not tuned","AF/NORM easy task",
                              "single clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
