#!/usr/bin/env python3
"""
E59 (N4) — Frequency-band "modality dropout" augmentation.

The modality gap is dominated by the LOW-FREQUENCY out-of-band component (baseline
wander, ~<1 Hz; E38/E43). Calibration (E42) *injects* matched wander so the model
learns to tolerate it. N4 tries the OPPOSITE mechanism: during training, randomly
DROP or SCRAMBLE the low-frequency band so the model CANNOT build any dependence on
it — forcing reliance on in-band (QRS/T morphology) content by construction. A
band-stochastic regularizer, needs NO watch data at all (unlike alignment).

Mechanistically distinct from every prior lever:
  - calibration (E42): INJECT out-of-band wander (domain randomization)     [needs target profile]
  - alignment (E51):   align same-patient feats across devices              [needs paired watch]
  - band-dropout (N4): REMOVE/randomize the out-of-band band during train   [needs NOTHING]

Arms (train PTB-XL AF/N, test real CinC, 20 seeds):
  clean          : CE only (floor)
  closed_aug     : calibration (E42, reference lever)
  band_drop      : per-sample random low-freq attenuation (drop band w/ random gain 0..1)
  band_scramble  : per-sample low-freq band phase-randomized (destroy wander structure, keep power)
  band_both      : union (each sample randomly drop OR scramble OR clean)
  joint_paired   : E51 paired InfoNCE (ceiling reference)

Read:
  band_* >> clean, ~ or > calibration -> a NO-DATA regularizer competitive with calibration (nice)
  band_* ~ clean                       -> removing the band doesn't help; in-band alone insufficient
  band_* < clean                       -> low-freq carries real signal too (destroys discriminative info)

HONEST FLAGS: CinC finger != AW wrist; AF/N easy; cutoff=1.5 Hz a-priori; single train set; 20 seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import importlib.util
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)
_s51 = importlib.util.spec_from_file_location("e51", ROOT / "experiments" / "51_label_anchored_align.py")
e51 = importlib.util.module_from_spec(_s51); _s51.loader.exec_module(e51)

RESULTS = ROOT / "results" / "59_band_dropout"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; CUT = 1.5   # Hz — boundary between "modality band" and in-band content

_sos_low = butter(4, CUT/(FS/2), btype="low", output="sos")

def _split_band(x):
    """Return (low_freq_component, high_freq_remainder). x is 1-D float."""
    low = sosfiltfilt(_sos_low, x).astype(np.float64)
    return low, (x - low)

def band_drop(x, rng):
    low, high = _split_band(x)
    g = rng.uniform(0.0, 1.0)               # random attenuation of the low band
    return (g*low + high).astype(np.float32)

def band_scramble(x, rng):
    low, high = _split_band(x)
    # phase-randomize the low band: keep magnitude spectrum, randomize phase
    F = np.fft.rfft(low)
    ph = np.exp(1j*rng.uniform(0, 2*np.pi, size=F.shape))
    ph[0] = 1.0
    low_s = np.fft.irfft(np.abs(F)*ph, n=len(low))
    return (low_s + high).astype(np.float32)

class BandAugDS(Dataset):
    def __init__(self, sigs, ys, mode, seed=0):
        self.sigs=sigs; self.ys=ys; self.mode=mode; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.sigs)
    def _norm(self,x):
        if x.shape[0] < SIGLEN: x=np.concatenate([x,np.zeros(SIGLEN-x.shape[0])])
        x=x[:SIGLEN]; return (x-x.mean())/(x.std()+1e-6)
    def __getitem__(self,i):
        x=np.asarray(self.sigs[i],np.float64)
        m=self.mode
        if m=="both": m=self.rng.choice(["drop","scramble","clean"])
        if m=="drop": x=band_drop(x,self.rng)
        elif m=="scramble": x=band_scramble(x,self.rng)
        x=self._norm(x)
        return torch.from_numpy(x[:,None].astype(np.float32)).permute(1,0), torch.tensor(self.ys[i])

def train_band(sigs, ys, mode, epochs=20, lr=1e-3, seed=0, tag="band"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=ECGResNet1d(n_leads=1,n_classes=2)
    dl=DataLoader(BandAugDS(sigs,ys,mode,seed=seed),batch_size=64,shuffle=True)
    opt=torch.optim.Adam(m.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0.0; nb=0
        for xb,yb in dl:
            opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
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
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=train_band(tr,tr_y,"drop",seed=seed,tag=f"s{seed}-drop"); out["band_drop"]=e25.evaluate(m,ts,ty)[0]
    m=train_band(tr,tr_y,"scramble",seed=seed,tag=f"s{seed}-scr"); out["band_scramble"]=e25.evaluate(m,ts,ty)[0]
    m=train_band(tr,tr_y,"both",seed=seed,tag=f"s{seed}-both"); out["band_both"]=e25.evaluate(m,ts,ty)[0]
    m=e51.train_joint(tr,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}-paired"); out["joint_paired"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    _s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
    e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL={len(tr)}  CinC={len(cinc)}  SJLIFE pairs={len(pairs)}  (cutoff={CUT} Hz)", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","closed_aug","band_drop","band_scramble","band_both","joint_paired"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean":   pair(A["closed_aug"],A["clean"]),
      "band_drop_vs_clean":    pair(A["band_drop"],A["clean"]),
      "band_scramble_vs_clean":pair(A["band_scramble"],A["clean"]),
      "band_both_vs_clean":    pair(A["band_both"],A["clean"]),
      "band_both_vs_aug":      pair(A["band_both"],A["closed_aug"]),
      "band_both_vs_paired":   pair(A["band_both"],A["joint_paired"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:24s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")
    print(f"\n  REFERENCE: clean {A['clean'].mean():.3f} | calibration {A['closed_aug'].mean():.3f} | paired(E51) {A['joint_paired'].mean():.3f}")
    print(f"  BAND-DROPOUT: drop {A['band_drop'].mean():.3f} | scramble {A['band_scramble'].mean():.3f} | both {A['band_both'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10.5,5))
    cols=["#999","#d4a017","#2a9d8f","#457b9d","#1d6a5a","#4a7"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["closed_aug"].mean(),color="#d4a017",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E59 frequency-band modality dropout (no watch data)\ndrop {A['band_drop'].mean():.3f} · scramble {A['band_scramble'].mean():.3f} · both {A['band_both'].mean():.3f} vs calibration {A['closed_aug'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"band_dropout.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'band_dropout.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"cutoff_hz":CUT,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy","cutoff 1.5Hz a priori",
                              "no watch data used (pure augmentation)","single clinical train set","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
