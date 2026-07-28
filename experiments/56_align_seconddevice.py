#!/usr/bin/env python3
"""
E56 — External validity: does the E51 alignment WIN replicate on a SECOND real device,
or is it CinC-specific?

Every alignment result (E51/E51b/E53/E54/E55) is on ONE test set: CinC 2017 (AliveCor
dry FINGER). The reviewer's first question: does +0.10 replicate on any other real
device, or did we overfit the method to CinC? Calibration got this check (E44,
Icentia); alignment never did.

Test device = Icentia11k CardioSTAT CHEST-PATCH (250→100 Hz, mined 200 AF + 200 Normal,
21 patients). Note this is a THIRD device — the SJLIFE alignment pairs are clinical +
Apple WRIST; Icentia is chest-patch, which the alignment NEVER saw. So this is a
strong out-of-distribution test of the learned invariance.

Arms (train PTB-XL AF/NORM Lead-I, test real Icentia, 20 seeds):
  clean       : clinical clean (floor)
  closed_aug  : calibration (E44 showed ~idles here — low-gap chest-patch)
  joint       : label-anchored alignment (E51) on SJLIFE pairs
  oracle      : train-on-real Icentia (leakage-inflated, E44 caveat -> treat qualitatively)

Read:
  joint > clean on Icentia too   -> alignment win REPLICATES across devices (external validity)
  joint ~= clean (like calib)     -> alignment win may be CinC-specific / gap-dependent
  joint < clean                   -> alignment HURTS on an unseen device (overfit to SJLIFE wrist)

HONEST FLAGS: Icentia within-patient leakage (21 pts, oracle ~1.0 in E44) -> absolutes
optimistic, use RELATIVE deltas; chest-patch is low modality-gap (E44 bw~0.016) so
little headroom; alignment pairs are wrist not chest-patch (3rd-device OOD); AF/NORM easy.
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np
import torch
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

RESULTS = ROOT / "results" / "56_align_seconddevice"; RESULTS.mkdir(parents=True, exist_ok=True)
ICE = ROOT / "data" / "icentia"
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def load_icentia():
    af=[np.load(f) for f in sorted(glob.glob(str(ICE/"af"/"*.npy")))]
    nn=[np.load(f) for f in sorted(glob.glob(str(ICE/"normal"/"*.npy")))]
    return [{"sig":s,"y":1} for s in af]+[{"sig":s,"y":0} for s in nn]

def run_seed(seed, tr, tr_y, ice, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(ice)); rng.shuffle(idx); cut=len(ice)//2
    ref=[ice[i] for i in idx[:cut]]; test=[ice[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=e51.train_joint(tr,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}-joint"); out["joint"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset([r["sig"] for r in ref],[r["y"] for r in ref]),epochs=20,tag=f"s{seed}-oracle"); out["oracle"]=e25.evaluate(m,ts,ty)[0]
    return out, tgt

def main():
    print("Loading PTB-XL AF/NORM + Icentia + SJLIFE ...", flush=True)
    pairs=e50.load_sjlife_pairs()
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    ice=load_icentia(); ny=sum(r["y"] for r in ice)
    print(f"  PTB-XL={len(tr)}  Icentia={len(ice)} (AF={ny},N={len(ice)-ny})  SJLIFE={len(pairs)}", flush=True)

    seeds=list(range(20)); per={}; bws=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        r,bw=run_seed(s,tr,tr_y,ice,pairs); per[s]=r; bws.append(bw)

    arms=["clean","closed_aug","joint","oracle"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print(f"\n  Icentia target bw = {np.mean(bws):.4f} (low-gap chest-patch; E44 confirmed)")
    print("\n===== AUROC on REAL Icentia / SECOND DEVICE (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean": pair(A["closed_aug"],A["clean"]),
      "joint_vs_clean":      pair(A["joint"],A["clean"]),
      "joint_vs_aug":        pair(A["joint"],A["closed_aug"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:22s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")
    print(f"  COMPARE CinC (E51): joint +0.106 vs clean.  Here (Icentia 2nd device): joint {cmp['joint_vs_clean'][0]:+.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
           color=["#999","#d4a017","#4a7","#333"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.set_ylim(0.5,1.02); ax.set_ylabel("AUROC real Icentia (20 seeds)")
    ax.set_title(f"E56 2nd-device external validity  joint−clean Δ={cmp['joint_vs_clean'][0]:+.3f} (p={cmp['joint_vs_clean'][2]:.3f}) vs CinC +0.106")
    fig.tight_layout(); fig.savefig(RESULTS/"align_seconddevice.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'align_seconddevice.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"test_device":"Icentia CardioSTAT chest-patch","target_bw":float(np.mean(bws)),
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "compare_CinC_E51_joint_vs_clean":0.106,
                   "honesty":["Icentia within-patient leakage (21 pts, oracle~1.0) -> use RELATIVE deltas",
                              "chest-patch low modality-gap (bw~0.016) -> little headroom",
                              "SJLIFE alignment pairs are WRIST not chest-patch (3rd-device OOD)",
                              "AF/NORM easy task","single clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
