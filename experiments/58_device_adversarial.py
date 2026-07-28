#!/usr/bin/env python3
"""
E58 (N2) — Device-adversarial content/style disentanglement (DANN + label anchor).

Attacks the SAME target as E57 (recover alignment gain WITHOUT same-patient pairs) but
through a different mechanism: instead of matching feature MOMENTS (CORAL) or transporting
MASS (Sinkhorn), train a **device discriminator** that tries to tell clinical-Lead-I from
Apple-Watch features, and use a **gradient-reversal layer (GRL)** so the encoder learns
features the discriminator CANNOT classify by modality — i.e. device-invariant *content*.
Retain the CE anchor (E48-E51 lesson) so invariance can't destroy pathology.

This is DANN (Ganin&Lempitsky 2015, vision DA) ported to ECG modality with a label anchor.
Adversarial invariance is a fundamentally different force from E57's distribution distance:
DANN pushes the encoder along the discriminator's decision-normal, an active min-max game,
not a passive statistic. Genuine open question whether it beats E57's ~⅓ recovery.

Arms (train PTB-XL AF/N, test real CinC, 20 seeds):
  clean        : CE only (floor)
  closed_aug   : calibration (E42)
  dann         : CE + λ·adversarial device-invariance (GRL device discriminator)  [UNPAIRED]
  dann_style   : content/style SPLIT — device head reads only a 'style' sub-vector,
                 classifier reads only the 'content' sub-vector, GRL scrubs device from content
  joint_paired : E51 paired InfoNCE (reference CEILING — uses correspondence)

Read:
  dann/dann_style >> clean, -> paired  : adversarial invariance recovers the gain unpaired (win)
  dann/dann_style ~ calibration        : some gain, no pairs needed but not the full win
  dann/dann_style ~ clean              : adversarial force also can't replace correspondence (confirms E57)

HONEST FLAGS: watch pool is SJLIFE (real AW) but UNPAIRED; CinC finger != AW wrist; AF/N easy;
λ=0.1 a-priori; GRL schedule standard 2/(1+e^-10p)-1; single clinical train set; 20 seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Function
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

RESULTS = ROOT / "results" / "58_device_adversarial"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; LAMBDA = 0.1; FEAT_DIM = 32; STYLE_DIM = 8  # content = 32-8 = 24

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

class WatchPool(Dataset):
    """UNPAIRED pool of all SJLIFE apple windows (pairing discarded)."""
    def __init__(self, pairs, seed=0):
        self.ws=[w for _,aws in pairs for w in aws]; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.ws)
    def __getitem__(self,i): return torch.from_numpy(self.ws[i][None])

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return g.neg() * ctx.alpha, None

def grl(x, alpha): return GradReverse.apply(x, alpha)

class DeviceHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(self, x): return self.net(x)

def train_dann(sigs, ys, watch, style_split=False, lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="dann"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=make_model()
    dev_in = STYLE_DIM if style_split else FEAT_DIM
    dhead=DeviceHead(dev_in)
    lab_dl=DataLoader(e51.LabeledDS(sigs,ys),batch_size=64,shuffle=True,drop_last=True)
    w_dl=DataLoader(WatchPool(watch,seed=seed),batch_size=64,shuffle=True,drop_last=True)
    params=list(m.parameters())+list(dhead.parameters())
    opt=torch.optim.Adam(params,lr=lr); ce=nn.CrossEntropyLoss()
    nsteps=epochs*max(len(lab_dl),1); step=0
    for ep in range(epochs):
        m.train(); dhead.train(); wit=iter(w_dl); tot=0.0; nb=0
        for xb,yb in lab_dl:
            try: xw=next(wit)
            except StopIteration: wit=iter(w_dl); xw=next(wit)
            p=step/max(nsteps,1); alpha=2.0/(1.0+np.exp(-10*p))-1.0   # DANN schedule
            opt.zero_grad()
            # ----- content classification (label anchor) -----
            fc=e50.encode(m,xb)                                       # (B,32) clinical feats
            if style_split:
                content_c = fc[:, :FEAT_DIM-STYLE_DIM]
                # classify from content only -> replace head[-1] linear manually
                logits = m.head[-1](torch.cat([content_c, torch.zeros_like(fc[:, FEAT_DIM-STYLE_DIM:])],1))
            else:
                logits = m.head[-1](fc)
            loss_ce=ce(logits,yb)
            # ----- device discrimination through GRL (both modalities) -----
            fw=e50.encode(m,xw)
            def dev_feat(f): return f[:, FEAT_DIM-STYLE_DIM:] if style_split else f
            dc=dhead(grl(dev_feat(fc),alpha)); dw=dhead(grl(dev_feat(fw),alpha))
            dev_y=torch.cat([torch.zeros(dc.size(0),dtype=torch.long),
                             torch.ones(dw.size(0),dtype=torch.long)])
            loss_dev=ce(torch.cat([dc,dw],0),dev_y)
            loss=loss_ce+lam*loss_dev
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1; step+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f} alpha={alpha:.2f}", flush=True)
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
    m=train_dann(tr,tr_y,pairs,style_split=False,seed=seed,tag=f"s{seed}-dann"); out["dann"]=e25.evaluate(m,ts,ty)[0]
    m=train_dann(tr,tr_y,pairs,style_split=True,seed=seed,tag=f"s{seed}-dannS"); out["dann_style"]=e25.evaluate(m,ts,ty)[0]
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

    arms=["clean","closed_aug","dann","dann_style","joint_paired"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean": pair(A["closed_aug"],A["clean"]),
      "dann_vs_clean":       pair(A["dann"],A["clean"]),
      "dann_style_vs_clean": pair(A["dann_style"],A["clean"]),
      "dann_vs_aug":         pair(A["dann"],A["closed_aug"]),
      "dann_style_vs_aug":   pair(A["dann_style"],A["closed_aug"]),
      "dann_vs_paired":      pair(A["dann"],A["joint_paired"]),
      "dann_style_vs_paired":pair(A["dann_style"],A["joint_paired"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")
    print(f"\n  REFERENCE: clean {A['clean'].mean():.3f} | calibration {A['closed_aug'].mean():.3f} | paired-InfoNCE(E51) {A['joint_paired'].mean():.3f}")
    print(f"  ADVERSARIAL: dann {A['dann'].mean():.3f} | dann_style {A['dann_style'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5))
    cols=["#999","#d4a017","#c0587e","#8a5cd6","#4a7"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["joint_paired"].mean(),color="#4a7",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E58 device-adversarial invariance (unpaired) vs paired ceiling\ndann {A['dann'].mean():.3f} · dann_style {A['dann_style'].mean():.3f} · paired(E51) {A['joint_paired'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"device_adversarial.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'device_adversarial.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"lambda":LAMBDA,"style_dim":STYLE_DIM,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["watch pool is SJLIFE real AW but UNPAIRED (pairing discarded)",
                              "CinC finger != AW wrist","AF/NORM easy","lambda=0.1 a priori",
                              "GRL standard schedule","single clinical train set","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
