#!/usr/bin/env python3
"""
E62 (N5) — Physiologically-invariant handcrafted features (amplitude/baseline-free).

The modality gap is (a) baseline wander (out-of-band low freq) + (b) ~8x amplitude gain
(SJLIFE). A feature that is INVARIANT to amplitude scaling and baseline offset BY
CONSTRUCTION cannot be corrupted by either — so it should transfer clinical->watch with no
modality leakage. Classical signal processing, cross-domain from HRV / nonlinear dynamics.

For AF-vs-NORM specifically, RR-interval IRREGULARITY is the textbook discriminator and is
pure TIMING — zero amplitude/baseline dependence. Hypothesis: a handcrafted feature model
is more modality-robust than the deep model, and ENSEMBLING the two beats either alone
(SignalMC-MED F6: handcrafted complementary to learned).

Feature vector (all amplitude/baseline invariant by construction; computed on z-normed +
QRS-bandpassed signal so scale/offset cancel):
  RR dynamics : meanRR, SDNN, RMSSD, pNN50, CV_RR, Shannon-entropy(RR), Poincare SD1/SD2,
                n_beats, HR
  morphology  : QRS-band spectral entropy (normalized), dominant-freq, spectral centroid,
                Hilbert instantaneous-phase circular variance
Classifier   : sklearn GradientBoosting on the feature vector.

Arms (train PTB-XL AF/N Lead-I, test real CinC, 20 seeds):
  clean         : deep Lead-I model (floor)
  closed_aug    : calibration (E42)
  handcrafted   : GB on amplitude/baseline-invariant features ONLY
  ensemble      : mean prob (deep clean + handcrafted)
  ensemble_cal  : mean prob (deep calibration + handcrafted)

Read:
  handcrafted >> clean          -> invariant-by-construction features ARE more modality-robust
  ensemble > max(parts)         -> handcrafted is COMPLEMENTARY to learned (the F6 claim)
  handcrafted ~ chance          -> Lead-I timing alone insufficient for this task

HONEST FLAGS: CinC finger != AW wrist; AF/N easy & RR-irregularity is almost a giveaway for
AF (feature model may be strong for a trivial reason); R-peak detection imperfect on noisy
watch-like signal; single clinical train set; 20 seeds.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.signal import hilbert
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e25 = _load("e25", "25_aw_generator.py")

RESULTS = ROOT / "results" / "62_invariant_features"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000
_bh, _ah = butter(3, 5.0/(FS/2), btype="high")     # QRS-band highpass (kills baseline)
_bl, _al = butter(3, [3.0/(FS/2), 40.0/(FS/2)], btype="band")

def _rpeaks(x):
    h = filtfilt(_bh, _ah, x)
    h = h / (np.std(h)+1e-8)                        # amplitude-normalize -> scale-invariant threshold
    pk,_ = find_peaks(h, height=1.2, distance=int(0.3*FS))
    return pk

def features(sig):
    """Amplitude/baseline-invariant feature vector."""
    x = np.asarray(sig, np.float64)
    x = (x - x.mean())/(x.std()+1e-8)               # z-norm: kill scale+offset up front
    pk = _rpeaks(x)
    f = []
    # ---- RR dynamics (pure timing, fully invariant) ----
    if len(pk) >= 3:
        rr = np.diff(pk)/FS*1000.0                  # ms
        meanrr = np.mean(rr); sdnn = np.std(rr)
        rmssd = np.sqrt(np.mean(np.diff(rr)**2)) if len(rr)>1 else 0.0
        pnn50 = np.mean(np.abs(np.diff(rr))>50) if len(rr)>1 else 0.0
        cvrr = sdnn/(meanrr+1e-8)
        # Shannon entropy of RR histogram
        hval,_ = np.histogram(rr, bins=8); p = hval/(hval.sum()+1e-8); p=p[p>0]
        shan = -np.sum(p*np.log(p))
        # Poincare
        if len(rr)>1:
            sd1 = np.std(np.diff(rr))/np.sqrt(2); sd2 = np.sqrt(2*sdnn**2 - sd1**2) if 2*sdnn**2>sd1**2 else 0.0
        else: sd1=sd2=0.0
        hr = 60000.0/(meanrr+1e-8); nb=len(pk)
    else:
        meanrr=sdnn=rmssd=pnn50=cvrr=shan=sd1=sd2=hr=0.0; nb=len(pk)
    f += [meanrr,sdnn,rmssd,pnn50,cvrr,shan,sd1,sd2,hr,float(nb)]
    # ---- morphology / spectral shape (amplitude-invariant via normalization) ----
    xb = filtfilt(_bl, _al, x)
    ff, pxx = welch(xb, fs=FS, nperseg=min(256,len(xb)))
    pxx = pxx/(pxx.sum()+1e-8)                       # normalized PSD -> amplitude invariant
    spec_ent = -np.sum(pxx[pxx>0]*np.log(pxx[pxx>0]))
    dom = ff[np.argmax(pxx)]
    centroid = np.sum(ff*pxx)
    # Hilbert instantaneous-phase circular variance
    ana = hilbert(xb); ph = np.angle(ana)
    circ_var = 1.0 - np.abs(np.mean(np.exp(1j*ph)))
    f += [spec_ent, dom, centroid, circ_var]
    return np.array(f, np.float64)

def build_feats(sigs):
    X = np.stack([features(s) for s in sigs])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def run_seed(seed, tr, tr_y, cinc):
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    import torch
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=np.array([r["y"] for r in test])

    out={}
    # deep clean
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean")
    import torch as T
    @T.no_grad()
    def deep_prob(model, sigs):
        model.eval(); from torch.utils.data import DataLoader
        dl=DataLoader(e25.SigDataset(sigs,[0]*len(sigs)),batch_size=64)
        P=[]
        for xb,_ in dl: P.append(T.softmax(model(xb),1).numpy()[:,1])
        return np.concatenate(P)
    p_clean=deep_prob(m, ts); out["clean"]=float(roc_auc_score(ty,p_clean))

    # deep calibration
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]
    mc=ECGResNet1d(n_leads=1,n_classes=2); e25.train(mc,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-cal")
    p_cal=deep_prob(mc, ts); out["closed_aug"]=float(roc_auc_score(ty,p_cal))

    # handcrafted invariant features
    Xtr=build_feats(tr); Xte=build_feats(ts)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=seed).fit(Xtr,np.array(tr_y))
    p_hand=gb.predict_proba(Xte)[:,1]; out["handcrafted"]=float(roc_auc_score(ty,p_hand))

    # ensembles (mean prob)
    out["ensemble"]=float(roc_auc_score(ty,(p_clean+p_hand)/2))
    out["ensemble_cal"]=float(roc_auc_score(ty,(p_cal+p_hand)/2))
    print(f"  [s{seed}] clean={out['clean']:.3f} cal={out['closed_aug']:.3f} hand={out['handcrafted']:.3f} ens={out['ensemble']:.3f} ens_cal={out['ensemble_cal']:.3f}", flush=True)
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC ...", flush=True)
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL={len(tr)}  CinC={len(cinc)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc)

    import numpy as np
    arms=["clean","closed_aug","handcrafted","ensemble","ensemble_cal"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean":   pair(A["closed_aug"],A["clean"]),
      "handcrafted_vs_clean":  pair(A["handcrafted"],A["clean"]),
      "handcrafted_vs_cal":    pair(A["handcrafted"],A["closed_aug"]),
      "ensemble_vs_clean":     pair(A["ensemble"],A["clean"]),
      "ensemble_vs_handcraft": pair(A["ensemble"],A["handcrafted"]),
      "ensemble_cal_vs_cal":   pair(A["ensemble_cal"],A["closed_aug"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:24s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols=["#999","#d4a017","#b5651d","#4a7","#2a7f5f"]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.006,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["closed_aug"].mean(),color="#d4a017",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E62 amplitude/baseline-invariant handcrafted features\nhand {A['handcrafted'].mean():.3f} · ens {A['ensemble'].mean():.3f} · ens+cal {A['ensemble_cal'].mean():.3f} · clean {A['clean'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"invariant_features.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'invariant_features.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/N easy & RR-irregularity near-giveaway for AF",
                              "R-peak detection imperfect on noisy signal","handcrafted invariant by construction",
                              "single clinical train set","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
