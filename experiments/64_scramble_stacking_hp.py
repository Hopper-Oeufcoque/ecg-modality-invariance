#!/usr/bin/env python3
"""
E64 (N4 follow-ups) — Resolve BOTH open questions from E59/E63 at HIGHER POWER (40 seeds):

  Q1 (from E63): does band-scramble STACK with calibration?
     E63 @20 seeds: combo 0.764 beats scramble sig (+0.026, p=0.026) but NOT calibration
     (+0.022, p=0.10, 10/20). Ambiguous — underpowered. 40 seeds should resolve sig-or-null.

  Q2 (the other half of the E59 follow-up, NEVER run): does scramble stack UNDER paired
     alignment? Alignment (E51) is the actual champion (0.807). If scramble adds on top of
     the champion (not just the augmentation tier), that's far more interesting than Q1.

Levers recap (all act on / around the <1.5 Hz baseline-wander band):
  - calibration : inject colored wander to measured target power   [needs unlabeled watch]
  - scramble    : phase-randomize low band per-epoch (keep power)   [needs NOTHING]
  - alignment   : CE(clinical) + lambda*InfoNCE(same-patient pairs) [needs paired watch] = CHAMPION
  - combos      : scramble+calibration ; scramble+alignment

Arms (train PTB-XL AF/N, test real CinC, 40 seeds):
  clean            : CE only (floor)
  calibration      : E42 closed-loop calibration
  scramble         : E59 band-scramble (per-epoch, raw clinical)
  scramble_calib   : per-epoch scramble applied to CALIBRATED signals          [Q1 combo]
  align            : E51 label-anchored paired alignment (ceiling reference)    [champion]
  scramble_align   : E51 alignment with the labeled CE stream ALSO scrambled    [Q2 combo]

Read Q1: scramble_calib vs calibration sig>0 -> COMPLEMENTARY ; ~0 -> REDUNDANT
Read Q2: scramble_align vs align       sig>0 -> scramble STACKS ON CHAMPION (big) ;
                                       ~0    -> alignment already saturates the band ;
                                       <0    -> scramble INTERFERES with alignment

HONEST FLAGS: CinC finger != AW wrist; AF/N easy; cutoff=1.5 Hz + lambda=0.1 a-priori;
combos using calibration/alignment are NOT zero-watch-data; single train set; 40 seeds.
Prior after E63: Q1 leans weak-complementary; Q2 genuinely unknown (band may already be
handled by the CE-anchored alignment, or scramble may free capacity for morphology).
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
_s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)

RESULTS = ROOT / "results" / "64_scramble_stacking_hp"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; CUT = 1.5; LAMBDA = 0.1; TEMP = 0.1

_sos_low = butter(4, CUT/(FS/2), btype="low", output="sos")

def _split_band(x):
    low = sosfiltfilt(_sos_low, x).astype(np.float64)
    return low, (x - low)

def band_scramble(x, rng):
    """Phase-randomize the <CUT Hz band: keep magnitude spectrum, randomize phase. (== E59)"""
    low, high = _split_band(x)
    F = np.fft.rfft(low)
    ph = np.exp(1j*rng.uniform(0, 2*np.pi, size=F.shape))
    ph[0] = 1.0
    low_s = np.fft.irfft(np.abs(F)*ph, n=len(low))
    return (low_s + high).astype(np.float32)

def _norm(x):
    x = np.asarray(x, np.float64)
    if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN-x.shape[0])])
    x = x[:SIGLEN]
    return ((x - x.mean())/(x.std()+1e-6)).astype(np.float32)

# ---- plain scramble-augmented CE training (== E59 scramble arm) ----
class ScrambleDS(Dataset):
    def __init__(self, sigs, ys, scramble=True, seed=0):
        self.sigs=sigs; self.ys=ys; self.scramble=scramble; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.sigs)
    def __getitem__(self,i):
        x=np.asarray(self.sigs[i],np.float64)
        if self.scramble: x=band_scramble(x,self.rng)
        x=_norm(x)
        return torch.from_numpy(x[:,None].astype(np.float32)).permute(1,0), torch.tensor(self.ys[i])

def train_scr(sigs, ys, scramble, epochs=20, lr=1e-3, seed=0, tag="scr"):
    torch.manual_seed(seed); np.random.seed(seed)
    m=ECGResNet1d(n_leads=1,n_classes=2)
    dl=DataLoader(ScrambleDS(sigs,ys,scramble=scramble,seed=seed),batch_size=64,shuffle=True)
    opt=torch.optim.Adam(m.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0.0; nb=0
        for xb,yb in dl:
            opt.zero_grad(); loss=ce(m(xb),yb); loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    return m

# ---- label-anchored alignment, with OPTIONAL per-epoch scramble on the labeled CE stream ----
class ScrLabeledDS(Dataset):
    """Labeled classification stream for the joint trainer; optionally scrambles per-epoch."""
    def __init__(self, sigs, ys, scramble=False, seed=0):
        self.sigs=sigs; self.ys=ys; self.scramble=scramble; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.sigs)
    def __getitem__(self,i):
        x=np.asarray(self.sigs[i],np.float64)
        if self.scramble: x=band_scramble(x,self.rng)
        return torch.from_numpy(_norm(x)[None]), torch.tensor(self.ys[i])

def train_joint_scr(sigs, ys, pairs, scramble=False, lam=LAMBDA, epochs=20, lr=1e-3, seed=0, tag="joint"):
    """E51 train_joint, but the labeled CE stream is optionally band-scrambled per-epoch.
    Alignment InfoNCE runs on raw SJLIFE pairs (unchanged) — faithful composition."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = ECGResNet1d(n_leads=1, n_classes=2)
    proj = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
    lab_dl = DataLoader(ScrLabeledDS(sigs, ys, scramble=scramble, seed=seed), batch_size=64, shuffle=True, drop_last=True)
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
            loss_ce = ce(m(xb), yb)
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
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-calib"); out["calibration"]=e25.evaluate(m,ts,ty)[0]
    m=train_scr(tr,tr_y,scramble=True,seed=seed,tag=f"s{seed}-scr"); out["scramble"]=e25.evaluate(m,ts,ty)[0]
    m=train_scr(aug,tr_y,scramble=True,seed=seed,tag=f"s{seed}-scrcal"); out["scramble_calib"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint_scr(tr,tr_y,pairs,scramble=False,seed=seed,tag=f"s{seed}-align"); out["align"]=e25.evaluate(m,ts,ty)[0]
    m=train_joint_scr(tr,tr_y,pairs,scramble=True,seed=seed,tag=f"s{seed}-scral"); out["scramble_align"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL={len(tr)}  CinC={len(cinc)}  SJLIFE pairs={len(pairs)}  (cut={CUT}Hz λ={LAMBDA})", flush=True)

    seeds=list(range(40)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","calibration","scramble","scramble_calib","align","scramble_align"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (40 seeds) =====")
    for a in arms: print(f"  {a:16s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    n=len(seeds)
    cmp={
      # Q1: scramble + calibration
      "calibration_vs_clean":        pair(A["calibration"],A["clean"]),
      "scramble_vs_clean":           pair(A["scramble"],A["clean"]),
      "scramble_calib_vs_scramble":  pair(A["scramble_calib"],A["scramble"]),
      "scramble_calib_vs_calib":     pair(A["scramble_calib"],A["calibration"]),  # THE Q1 decider
      # Q2: scramble + alignment
      "align_vs_clean":              pair(A["align"],A["clean"]),
      "align_vs_calib":              pair(A["align"],A["calibration"]),
      "scramble_align_vs_align":     pair(A["scramble_align"],A["align"]),         # THE Q2 decider
      "scramble_align_vs_clean":     pair(A["scramble_align"],A["clean"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:26s} Δ={dl:+.3f} wins {w}/{n} p={p:.4f}")

    def verdict(delta,p):
        if p<0.05 and delta>0: return "COMPLEMENTARY/STACKS"
        if p<0.05 and delta<0: return "INTERFERENCE"
        return "REDUNDANT/NULL"
    q1 = verdict(*cmp["scramble_calib_vs_calib"][::2])
    q2 = verdict(*cmp["scramble_align_vs_align"][::2])
    print(f"\n  Q1 scramble+calibration : {q1}  (Δ={cmp['scramble_calib_vs_calib'][0]:+.3f} p={cmp['scramble_calib_vs_calib'][2]:.4f})")
    print(f"  Q2 scramble+alignment   : {q2}  (Δ={cmp['scramble_align_vs_align'][0]:+.3f} p={cmp['scramble_align_vs_align'][2]:.4f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5))
    cols=["#999","#d4a017","#457b9d","#1d6a5a","#4a7","#2b8a3e"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.axhline(A["align"].mean(),color="#2b8a3e",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (40 seeds)")
    ax.set_title(f"E64 scramble stacking @40 seeds\nQ1 +calib: {q1} (Δ={cmp['scramble_calib_vs_calib'][0]:+.3f} p={cmp['scramble_calib_vs_calib'][2]:.3f})   |   Q2 +align: {q2} (Δ={cmp['scramble_align_vs_align'][0]:+.3f} p={cmp['scramble_align_vs_align'][2]:.3f})")
    fig.tight_layout(); fig.savefig(RESULTS/"scramble_stacking_hp.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'scramble_stacking_hp.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":n,"cutoff_hz":CUT,"lambda":LAMBDA,"temp":TEMP,
                   "verdict_q1_scramble_calib":q1,"verdict_q2_scramble_align":q2,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy","cutoff 1.5Hz + lambda 0.1 a priori",
                              "combos w/ calibration or alignment use watch data — NOT zero-data",
                              "single clinical train set","40 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
