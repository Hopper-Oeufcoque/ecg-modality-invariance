#!/usr/bin/env python3
"""
E62b (N5 control) — Invariant handcrafted features on a MORPHOLOGY task (falsification).

E62 found handcrafted amplitude/baseline-invariant features hit 0.908 on CinC AF — but 78%
of that was CV-of-RR, and AF IS rhythm irregularity, so the win may be timing-tautological,
not a general invariance victory. THE TEST: run the IDENTICAL pipeline on a MORPHOLOGY task
(NORM vs Other/abnormal, E53's task) where the signal lives in wave SHAPE, not timing. If the
handcrafted margin COLLAPSES here, the E62 win is confirmed timing-specific (honest boundary);
if it holds, invariant features are generally strong (bigger claim).

Same feature extractor + GB + ensemble as E62; only the task/data change:
  TRAIN clinical: PTB-XL Lead-I NORM vs morphological-abnormal (MI/STTC/CD/HYP; AF excluded)
  TEST real:      CinC N-vs-O (Other = non-AF abnormal; the E47/E53 morphology proxy)

Arms (20 seeds): clean, closed_aug, handcrafted, ensemble, ensemble_cal (identical to E62).

HONEST FLAGS: CinC-O is a weak catch-all label (oracle only ~0.75 in E53); morphology
intrinsically harder; same feature set as E62 (now expected to be near-useless — that's the point).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e25 = _load("e25", "25_aw_generator.py")
import glob as _glob
_e47name = Path(_glob.glob(str(ROOT/"experiments"/"47_*.py"))[0]).name
e47 = _load("e47", _e47name)
e62 = _load("e62", "62_invariant_features.py")

RESULTS = ROOT / "results" / "62b_invariant_morphology"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def run_seed(seed, tr, tr_y, cinc):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    import torch as T
    rng=np.random.default_rng(seed); T.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=np.array([r["y"] for r in test])
    out={}
    @T.no_grad()
    def deep_prob(model, sigs):
        model.eval(); dl=DataLoader(e25.SigDataset(sigs,[0]*len(sigs)),batch_size=64); P=[]
        for xb,_ in dl: P.append(T.softmax(model(xb),1).numpy()[:,1])
        return np.concatenate(P)
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean")
    p_clean=deep_prob(m,ts); out["clean"]=float(roc_auc_score(ty,p_clean))
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]
    mc=ECGResNet1d(n_leads=1,n_classes=2); e25.train(mc,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-cal")
    p_cal=deep_prob(mc,ts); out["closed_aug"]=float(roc_auc_score(ty,p_cal))
    Xtr=e62.build_feats(tr); Xte=e62.build_feats(ts)
    sc=StandardScaler().fit(Xtr)
    gb=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=seed).fit(sc.transform(Xtr),np.array(tr_y))
    p_hand=gb.predict_proba(sc.transform(Xte))[:,1]; out["handcrafted"]=float(roc_auc_score(ty,p_hand))
    out["ensemble"]=float(roc_auc_score(ty,(p_clean+p_hand)/2))
    out["ensemble_cal"]=float(roc_auc_score(ty,(p_cal+p_hand)/2))
    print(f"  [s{seed}] clean={out['clean']:.3f} cal={out['closed_aug']:.3f} hand={out['handcrafted']:.3f} ens={out['ensemble']:.3f}", flush=True)
    return out

def main():
    print("Loading PTB-XL NORM-vs-morphological-abnormal + CinC N-vs-O ...", flush=True)
    d=e47.load_ptbxl_norm_vs_abnormal(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d]; tr_y=[r["y"] for r in d]
    O,N=e47.load_cinc_n_vs_o(n_per_class=700); cinc=O+N
    print(f"  PTB-XL n={len(tr)} (abn={sum(tr_y)})  CinC N-vs-O n={len(cinc)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc)

    arms=["clean","closed_aug","handcrafted","ensemble","ensemble_cal"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC N-vs-O MORPHOLOGY (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "handcrafted_vs_clean": pair(A["handcrafted"],A["clean"]),
      "handcrafted_vs_cal":   pair(A["handcrafted"],A["closed_aug"]),
      "ensemble_vs_clean":    pair(A["ensemble"],A["clean"]),
      "ensemble_vs_handcraft":pair(A["ensemble"],A["handcrafted"]),
    }
    print(f"\n  E62 (AF/timing) handcrafted was +0.208 vs clean. HERE (morphology):")
    for k,(dl,w,p) in cmp.items(): print(f"  {k:24s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols=["#999","#d4a017","#b5651d","#4a7","#2a7f5f"]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.006,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(0.5,color="r",ls=":",lw=1)
    ax.set_ylim(0.4,0.85); ax.set_ylabel("AUROC CinC N-vs-O morphology (20 seeds)")
    ax.set_title(f"E62b CONTROL: invariant features on MORPHOLOGY (vs E62's AF +0.208)\nhand {A['handcrafted'].mean():.3f} · clean {A['clean'].mean():.3f} · cal {A['closed_aug'].mean():.3f} · ens {A['ensemble'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"invariant_morphology.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'invariant_morphology.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"task":"NORM-vs-Other morphology (E62 AF control)",
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC-O weak catch-all (oracle ~0.75)","morphology intrinsically harder",
                              "same feature set as E62 (now expected near-useless — the point)","20 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
