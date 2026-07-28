#!/usr/bin/env python3
"""
E55 — Does label-anchored alignment STACK with real few-shot labels toward oracle?

The north-star deployment recipe: abundant clinical data + (unlabeled) real paired
set + a HANDFUL of real target labels. E46 mapped the labels-to-target curve for
clean vs calibration (both plateaued ~0.85 << oracle 0.923). E51 alignment is the
confirmed winner at k=0 (joint 0.807 vs calibration 0.742). THE question: as we add
k real CinC labels (fine-tune), does alignment stay ahead / reach oracle FASTER, or
do the arms converge (labels wash out the alignment advantage, as they did for
calibration in E46)?

Arms, for k in {0,10,25,50,100} real CinC labels (fine-tune the base model):
  clean+k   : clinical clean base -> fine-tune on k
  closed+k  : closed-loop-calibrated base -> fine-tune on k        (E46 winner among the two)
  joint+k   : label-anchored alignment base (E51) -> fine-tune on k (the confirmed champ)
Plus oracle (train-on-real). AF task, 10 seeds (matches E46 budget).

Read:
  joint+k dominates the whole curve      -> alignment advantage persists with labels (best recipe)
  joint+k reaches 0.90+ at small k        -> alignment + few labels approaches oracle (north-star hit)
  arms converge as k grows                -> labels substitute for alignment (like E46 calibration)

HONEST FLAGS: CinC finger != AW wrist; AF/NORM easy; tiny-k fine-tune high variance
(E30); joint base uses SJLIFE pairs (3-device caveat); single clinical train set; 10 seeds.
"""
from __future__ import annotations
import json, sys, copy
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

RESULTS = ROOT / "results" / "55_align_fewshot"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; KS = [0, 10, 25, 50, 100]; N_SEEDS = 10

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def finetune(model, sigs, ys, epochs=15, lr=5e-4):
    if len(sigs)==0: return model
    dl=DataLoader(e25.SigDataset(sigs,ys),batch_size=min(32,len(sigs)),shuffle=True)
    opt=torch.optim.Adam(model.parameters(),lr=lr); lf=nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb,yb in dl:
            opt.zero_grad(); lf(model(xb),yb).backward(); opt.step()
    return model

def sample_k(rng, ref, k):
    if k==0: return [],[]
    af=[r for r in ref if r["y"]==1]; nn_=[r for r in ref if r["y"]==0]
    rng.shuffle(af); rng.shuffle(nn_); half=k//2
    picks=af[:half]+nn_[:k-half]
    return [p["sig"] for p in picks],[p["y"] for p in picks]

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))

    # base models (train once each, fine-tune copies per k)
    m_clean=make_model(); e25.train(m_clean,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean")
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]
    m_closed=make_model(); e25.train(m_closed,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-closed")
    m_joint=e51.train_joint(tr,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}-joint")

    res={"clean":{},"closed":{},"joint":{}}
    rk=np.random.default_rng(1000+seed)
    for k in KS:
        sk,yk=sample_k(rk,ref,k)
        res["clean"][k]=e25.evaluate(finetune(copy.deepcopy(m_clean),sk,yk),ts,ty)[0]
        res["closed"][k]=e25.evaluate(finetune(copy.deepcopy(m_closed),sk,yk),ts,ty)[0]
        res["joint"][k]=e25.evaluate(finetune(copy.deepcopy(m_joint),sk,yk),ts,ty)[0]
    mo=make_model(); e25.train(mo,e25.SigDataset([r["sig"] for r in ref],[r["y"] for r in ref]),epochs=20,tag=f"s{seed}-oracle")
    res["oracle"]=e25.evaluate(mo,ts,ty)[0]
    return res

def main():
    print("Loading PTB-XL AF/NORM + CinC + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  SJLIFE={len(pairs)} PTB-XL={len(tr)} CinC={len(cinc)}", flush=True)

    seeds=list(range(N_SEEDS)); per=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per.append(run_seed(s,tr,tr_y,cinc,pairs))

    def curve(arm): return {k: np.array([p[arm][k] for p in per]) for k in KS}
    cc=curve("clean"); cl=curve("closed"); cj=curve("joint")
    oracle=np.array([p["oracle"] for p in per])
    from scipy import stats as sst

    print("\n===== labels-to-target curves (AUROC real CinC, 10 seeds) =====")
    print(f"  {'k':>4}  {'clean':>14}  {'closed':>14}  {'joint':>14}")
    for k in KS:
        print(f"  {k:>4}  {cc[k].mean():.3f}±{cc[k].std():.3f}  {cl[k].mean():.3f}±{cl[k].std():.3f}  {cj[k].mean():.3f}±{cj[k].std():.3f}")
    print(f"  oracle = {oracle.mean():.3f}±{oracle.std():.3f}")
    # key comparisons
    print("\n  joint vs closed at each k:")
    for k in KS:
        t,p=sst.ttest_rel(cj[k],cl[k]); print(f"    k={k:<4} Δ={cj[k].mean()-cl[k].mean():+.3f} wins {(cj[k]>cl[k]).sum()}/{N_SEEDS} p={p:.3f}")
    def labels_to(cd, target):
        for k in KS:
            if cd[k].mean()>=target: return k
        return None
    print(f"\n  labels to reach 0.85:  clean={labels_to(cc,0.85)} closed={labels_to(cl,0.85)} joint={labels_to(cj,0.85)}")
    print(f"  labels to reach 0.90:  clean={labels_to(cc,0.90)} closed={labels_to(cl,0.90)} joint={labels_to(cj,0.90)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    ax.errorbar(KS,[cc[k].mean() for k in KS],yerr=[cc[k].std() for k in KS],marker="o",color="#999",label="clean + k",capsize=3)
    ax.errorbar(KS,[cl[k].mean() for k in KS],yerr=[cl[k].std() for k in KS],marker="s",color="#d4a017",label="calibration + k",capsize=3)
    ax.errorbar(KS,[cj[k].mean() for k in KS],yerr=[cj[k].std() for k in KS],marker="^",color="#4a7",label="alignment (E51) + k",capsize=3)
    ax.axhline(oracle.mean(),color="k",ls="--",lw=1,label=f"oracle {oracle.mean():.3f}")
    ax.set_xlabel("k = real labeled CinC examples (fine-tune)"); ax.set_ylabel("AUROC real CinC")
    ax.set_title("E55 does alignment stack with few-shot labels toward oracle?"); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS/"align_fewshot.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'align_fewshot.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":N_SEEDS,"ks":KS,
                   "clean":{str(k):[float(cc[k].mean()),float(cc[k].std())] for k in KS},
                   "closed":{str(k):[float(cl[k].mean()),float(cl[k].std())] for k in KS},
                   "joint":{str(k):[float(cj[k].mean()),float(cj[k].std())] for k in KS},
                   "oracle":[float(oracle.mean()),float(oracle.std())],
                   "honesty":["CinC finger != AW wrist","AF/NORM easy task","tiny-k fine-tune high variance (E30)",
                              "joint base uses SJLIFE pairs (3-device)","single clinical train set","10 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
