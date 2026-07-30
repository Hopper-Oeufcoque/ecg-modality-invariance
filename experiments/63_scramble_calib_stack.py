#!/usr/bin/env python3
"""
E63 (N4 follow-up) — Does band-scramble STACK with closed-loop calibration?

E59 found band_scramble (phase-randomize the <1.5 Hz band, keep power / destroy
structure) TIES calibration (0.739 vs 0.742) using ZERO watch data, and beats clean
+0.038. Calibration (E42) INSTEAD *injects* wander at the measured target level (needs
an unlabeled watch set to know the level). Both act on the SAME sub-1.5 Hz band, so the
open question (queued in NOVEL_IDEATION) is whether they are COMPLEMENTARY or REDUNDANT.

Mechanism recap:
  - calibration : inject colored wander to hit measured target bw_energy   -> fixes wander POWER   [needs target profile]
  - scramble    : phase-randomize the low band each epoch (keep power)      -> destroys wander STRUCTURE [needs NOTHING]
  - COMBINED    : calibrate to target power, THEN scramble structure/epoch  -> right power + no structure

Because scramble PRESERVES power, applying it on top of calibrated signals keeps
calibration's level-matching contribution intact while adding structure destruction —
the fair additivity test. (Reverse order would re-inject structure and defeat scramble.)

Arms (train PTB-XL AF/N, test real CinC, 20 seeds — identical harness to E59):
  clean            : CE only (floor)
  calibration      : E42 closed-loop calibration (static injected wander)      [reference lever]
  scramble         : E59 band-scramble, per-epoch, on RAW clinical             [reference lever]
  scramble_calib   : per-epoch scramble applied to CALIBRATED signals          [THE COMBINATION]
  joint_paired     : E51 paired InfoNCE (ceiling reference)

Read:
  scramble_calib  > max(scramble, calibration)  significantly -> COMPLEMENTARY (surprising; different sub-mechanisms)
  scramble_calib ~= max(scramble, calibration)                -> REDUNDANT (shared band saturated; prior expectation)
  scramble_calib  < max(...)                                  -> INTERFERENCE (combining hurts)

HONEST FLAGS: CinC finger != AW wrist; AF/N easy; cutoff=1.5 Hz a-priori; single train set; 20 seeds.
Prior expectation: REDUNDANT (both target the same wander band).
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

RESULTS = ROOT / "results" / "63_scramble_calib_stack"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; CUT = 1.5   # identical to E59

_sos_low = butter(4, CUT/(FS/2), btype="low", output="sos")

def _split_band(x):
    low = sosfiltfilt(_sos_low, x).astype(np.float64)
    return low, (x - low)

def band_scramble(x, rng):
    """Phase-randomize the <CUT Hz band: keep magnitude spectrum, randomize phase."""
    low, high = _split_band(x)
    F = np.fft.rfft(low)
    ph = np.exp(1j*rng.uniform(0, 2*np.pi, size=F.shape))
    ph[0] = 1.0
    low_s = np.fft.irfft(np.abs(F)*ph, n=len(low))
    return (low_s + high).astype(np.float32)

class ScrambleDS(Dataset):
    """Per-epoch scramble applied to whatever base signals are passed in
    (RAW clinical for the `scramble` arm; CALIBRATED signals for the combo)."""
    def __init__(self, sigs, ys, scramble=True, seed=0):
        self.sigs=sigs; self.ys=ys; self.scramble=scramble; self.rng=np.random.default_rng(seed)
    def __len__(self): return len(self.sigs)
    def _norm(self,x):
        if x.shape[0] < SIGLEN: x=np.concatenate([x,np.zeros(SIGLEN-x.shape[0])])
        x=x[:SIGLEN]; return (x-x.mean())/(x.std()+1e-6)
    def __getitem__(self,i):
        x=np.asarray(self.sigs[i],np.float64)
        if self.scramble: x=band_scramble(x,self.rng)
        x=self._norm(x)
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

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    # calibration: measure target bw from unlabeled ref set, fit injector, make static aug set
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]   # calibrated signals (target-matched wander power)

    out={}
    # clean floor
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    # calibration reference (static injected wander, no scramble)
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-calib"); out["calibration"]=e25.evaluate(m,ts,ty)[0]
    # scramble reference (per-epoch scramble on RAW clinical)
    m=train_scr(tr,tr_y,scramble=True,seed=seed,tag=f"s{seed}-scr"); out["scramble"]=e25.evaluate(m,ts,ty)[0]
    # THE COMBINATION: per-epoch scramble applied to CALIBRATED signals
    m=train_scr(aug,tr_y,scramble=True,seed=seed,tag=f"s{seed}-scrcal"); out["scramble_calib"]=e25.evaluate(m,ts,ty)[0]
    # ceiling reference
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

    arms=["clean","calibration","scramble","scramble_calib","joint_paired"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:16s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    # the decisive comparisons: does the combo beat EACH ingredient?
    best_single = "scramble" if A["scramble"].mean()>=A["calibration"].mean() else "calibration"
    cmp={
      "calibration_vs_clean":        pair(A["calibration"],A["clean"]),
      "scramble_vs_clean":           pair(A["scramble"],A["clean"]),
      "scramble_calib_vs_clean":     pair(A["scramble_calib"],A["clean"]),
      "scramble_calib_vs_scramble":  pair(A["scramble_calib"],A["scramble"]),
      "scramble_calib_vs_calib":     pair(A["scramble_calib"],A["calibration"]),
      "scramble_calib_vs_bestsingle":pair(A["scramble_calib"],A[best_single]),
      "scramble_calib_vs_paired":    pair(A["scramble_calib"],A["joint_paired"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:28s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    # verdict logic
    d_vs_best = cmp["scramble_calib_vs_bestsingle"][0]; p_vs_best = cmp["scramble_calib_vs_bestsingle"][2]
    if d_vs_best > 0 and p_vs_best < 0.05:   verdict = "COMPLEMENTARY"
    elif d_vs_best < 0 and p_vs_best < 0.05: verdict = "INTERFERENCE"
    else:                                     verdict = "REDUNDANT"
    print(f"\n  best single lever = {best_single} ({A[best_single].mean():.3f})")
    print(f"  VERDICT: {verdict} (combo Δ={d_vs_best:+.3f} vs best single, p={p_vs_best:.4f})")
    print(f"\n  REFERENCE: clean {A['clean'].mean():.3f} | calibration {A['calibration'].mean():.3f} | scramble {A['scramble'].mean():.3f} | combo {A['scramble_calib'].mean():.3f} | paired {A['joint_paired'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9.5,5))
    cols=["#999","#d4a017","#457b9d","#1d6a5a","#4a7"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.axhline(max(A["scramble"].mean(),A["calibration"].mean()),color="#c33",ls="--",lw=1,label="best single lever")
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E63 — does scramble stack with calibration?  VERDICT: {verdict}\ncombo {A['scramble_calib'].mean():.3f} vs best-single {max(A['scramble'].mean(),A['calibration'].mean()):.3f} (Δ={d_vs_best:+.3f}, p={p_vs_best:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS/"scramble_calib_stack.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'scramble_calib_stack.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"cutoff_hz":CUT,"verdict":verdict,"best_single":best_single,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy","cutoff 1.5Hz a priori",
                              "combo uses watch data (via calibration) — NOT zero-data like pure scramble",
                              "single clinical train set","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
