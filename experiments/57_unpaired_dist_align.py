#!/usr/bin/env python3
"""
E57 (N1) — Can we drop the PAIRED-hardware dependency? Unpaired distribution alignment.

The headline (E51) needs an unlabeled PAIRED set (SJLIFE, same-patient clinical↔watch) —
the rarest ingredient. If a meaningful fraction of the gain survives with only UNPAIRED
watch feature clouds (no same-patient correspondence), the method becomes far more
deployable: any pile of unlabeled watch traces would do.

E51b showed shuffled INSTANCE-pairing → null. But DISTRIBUTION matching is a different
mechanism (aligns marginals, not instances), so it's a genuine open question. Retain the
CE anchor throughout (E48–E51 lesson: unanchored invariance destroys pathology).

Objective: CE(clinical Lead-I labels) + λ · D(clinical_feats, watch_feats), where D is:
  coral    : 2nd-moment (covariance) matching — Sun & Saenko 2016 (vision DA)
  sinkhorn : entropic optimal transport between the two feature minibatches
Watch feats = SJLIFE apple windows treated as an UNPAIRED pool (pairing discarded).

Arms (train PTB-XL AF/N, test real CinC, 20 seeds):
  clean        : CE only (floor)
  closed_aug   : calibration (E42)
  coral        : CE + λ·CORAL(clinical feats, unpaired watch feats)
  sinkhorn     : CE + λ·Sinkhorn-OT(clinical feats, unpaired watch feats)
  joint_paired : E51 paired InfoNCE (reference CEILING — uses correspondence)

Read:
  coral/sinkhorn >> clean, ≈ joint_paired -> correspondence NOT needed (huge deployability win)
  coral/sinkhorn ~ calibration            -> some gain, cheaper than pairs but not the full win
  coral/sinkhorn ~ clean                  -> correspondence IS essential (confirms E51b at dist level)

HONEST FLAGS: watch pool is still SJLIFE (real AW), just unpaired; CinC finger != AW wrist;
AF/N easy; λ=0.1 a-priori; single clinical train set; 20 seeds.
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

RESULTS = ROOT / "results" / "57_unpaired_dist_align"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.1

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def _norm(x):
    x=np.asarray(x,np.float64)
    if x.shape[0]<SIGLEN: x=np.concatenate([x,np.zeros(SIGLEN-x.shape[0])])
    return ((x[:SIGLEN]-x[:SIGLEN].mean())/(x[:SIGLEN].std()+1e-6)).astype(np.float32)

class WatchPool(Dataset):
    """UNPAIRED pool of all SJLIFE apple windows (pairing discarded)."""
    def __init__(self, pairs, seed=0):
        self.ws=[w for _,aws in pairs for w in aws]; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.ws)
    def __getitem__(self,i): return torch.from_numpy(self.ws[i][None])

def coral_loss(fs, ft):
    """CORAL: squared Frobenius distance between feature covariances (Sun&Saenko 2016)."""
    d = fs.size(1)
    fs = fs - fs.mean(0, keepdim=True); ft = ft - ft.mean(0, keepdim=True)
    cs = (fs.t() @ fs) / (fs.size(0)-1); ct = (ft.t() @ ft) / (ft.size(0)-1)
    return ((cs-ct)**2).sum() / (4*d*d)

def sinkhorn_loss(fs, ft, eps=0.1, iters=50):
    """Entropic OT (Sinkhorn) between two feature minibatches (squared-euclidean cost)."""
    x = torch.nn.functional.normalize(fs, dim=1); y = torch.nn.functional.normalize(ft, dim=1)
    C = torch.cdist(x, y, p=2)**2                      # (n,m)
    n, m = C.shape
    a = torch.full((n,), 1.0/n); b = torch.full((m,), 1.0/m)
    K = torch.exp(-C/eps) + 1e-9
    u = torch.ones(n); v = torch.ones(m)
    for _ in range(iters):
        u = a / (K @ v + 1e-9); v = b / (K.t() @ u + 1e-9)
    P = torch.diag(u) @ K @ torch.diag(v)
    return (P * C).sum()

def train_dist(sigs, ys, watch, mode, lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="d"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=make_model()
    lab_dl=DataLoader(e51.LabeledDS(sigs,ys),batch_size=64,shuffle=True,drop_last=True)
    w_dl=DataLoader(WatchPool(watch,seed=seed),batch_size=64,shuffle=True,drop_last=True)
    opt=torch.optim.Adam(m.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); wit=iter(w_dl); tot=0.0; nb=0
        for xb,yb in lab_dl:
            try: xw=next(wit)
            except StopIteration: wit=iter(w_dl); xw=next(wit)
            opt.zero_grad()
            loss_ce=ce(m(xb),yb)
            fc=e50.encode(m,xb); fw=e50.encode(m,xw)
            D = coral_loss(fc,fw) if mode=="coral" else sinkhorn_loss(fc,fw)
            loss=loss_ce+lam*D
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    return m

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=train_dist(tr,tr_y,pairs,"coral",seed=seed,tag=f"s{seed}-coral"); out["coral"]=e25.evaluate(m,ts,ty)[0]
    m=train_dist(tr,tr_y,pairs,"sinkhorn",seed=seed,tag=f"s{seed}-sink"); out["sinkhorn"]=e25.evaluate(m,ts,ty)[0]
    m=e51.train_joint(tr,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}-paired"); out["joint_paired"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    nw=sum(len(a) for _,a in pairs)
    print(f"  SJLIFE pairs={len(pairs)} (watch pool={nw} unpaired windows)  PTB-XL={len(tr)}  CinC={len(cinc)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","closed_aug","coral","sinkhorn","joint_paired"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean": pair(A["closed_aug"],A["clean"]),
      "coral_vs_clean":      pair(A["coral"],A["clean"]),
      "sinkhorn_vs_clean":   pair(A["sinkhorn"],A["clean"]),
      "coral_vs_aug":        pair(A["coral"],A["closed_aug"]),
      "sinkhorn_vs_aug":     pair(A["sinkhorn"],A["closed_aug"]),
      "coral_vs_paired":     pair(A["coral"],A["joint_paired"]),
      "sinkhorn_vs_paired":  pair(A["sinkhorn"],A["joint_paired"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")
    print(f"\n  REFERENCE: clean {A['clean'].mean():.3f} | calibration {A['closed_aug'].mean():.3f} | paired-InfoNCE(E51) {A['joint_paired'].mean():.3f}")
    print(f"  UNPAIRED: coral {A['coral'].mean():.3f} | sinkhorn {A['sinkhorn'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5))
    cols=["#999","#d4a017","#7b5cd6","#3aa6b9","#4a7"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["joint_paired"].mean(),color="#4a7",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E57 unpaired distribution alignment vs paired ceiling\ncoral {A['coral'].mean():.3f} · sinkhorn {A['sinkhorn'].mean():.3f} · paired(E51) {A['joint_paired'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"unpaired_dist_align.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'unpaired_dist_align.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"lambda":LAMBDA,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["watch pool is SJLIFE real AW but UNPAIRED (pairing discarded)",
                              "CinC finger != AW wrist","AF/NORM easy","lambda=0.1 a priori",
                              "single clinical train set","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
