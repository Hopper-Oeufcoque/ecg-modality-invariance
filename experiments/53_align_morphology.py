#!/usr/bin/env python3
"""
E53 — Does the E51 label-anchored ALIGNMENT win break the RHYTHM-only boundary?

E47 showed closed-loop CALIBRATION is rhythm-specific: null (−0.002) on a
morphological task (Normal-vs-Other), while +0.041 on AF. E51 (label-anchored
alignment) is our confirmed headline win — but only tested on AF. THE key
generality question: does alignment help MORPHOLOGY too (→ far more general than
calibration), or does it share the same rhythm-only boundary?

Task = E47's harder morphological one:
  TRAIN clinical: PTB-XL Lead-I NORM vs morphological-abnormal (MI/STTC/CD/HYP; AF excluded)
  TEST real:      CinC 2017 N vs O (catch-all non-AF)
Arms (20 seeds): clean · closed_aug (calibration; E47 null here) · joint (E51
align+classify on SJLIFE real pairs) · oracle (train-on-real).

Read:
  joint >> clean on morphology  -> alignment GENERALIZES beyond rhythm (big deal)
  joint ~= clean (like calibration) -> alignment ALSO rhythm-bound (shares E47 limit)

HONEST FLAGS: CinC O heterogeneous catch-all; PTB-XL abnormal != CinC O taxonomy;
3 devices; lambda=0.1/temp=0.1 a-priori; single clinical train set; 20 seeds.
"""
from __future__ import annotations
import json, sys
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
_s47 = importlib.util.spec_from_file_location("e47", ROOT / "experiments" / "47_harder_task.py")
e47 = importlib.util.module_from_spec(_s47); _s47.loader.exec_module(e47)
_s50 = importlib.util.spec_from_file_location("e50", ROOT / "experiments" / "50_sjlife_align.py")
e50 = importlib.util.module_from_spec(_s50); _s50.loader.exec_module(e50)
_s51 = importlib.util.spec_from_file_location("e51", ROOT / "experiments" / "51_label_anchored_align.py")
e51 = importlib.util.module_from_spec(_s51); _s51.loader.exec_module(e51)

RESULTS = ROOT / "results" / "53_align_morphology"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def run_seed(seed, tr, tr_y, cinc, pairs):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    m=e51.train_joint(tr,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}-joint"); out["joint"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset([r["sig"] for r in ref],[r["y"] for r in ref]),epochs=20,tag=f"s{seed}-oracle"); out["oracle"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL NORM vs morphological-abnormal + CinC N-vs-O + SJLIFE ...", flush=True)
    d=e47.load_ptbxl_norm_vs_abnormal(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d]; tr_y=[r["y"] for r in d]
    O,N=e47.load_cinc_n_vs_o(n_per_class=700); cinc=O+N
    pairs=e50.load_sjlife_pairs()
    print(f"  PTB-XL n={len(tr)} (abn={sum(tr_y)}, norm={len(tr_y)-sum(tr_y)})  CinC N-vs-O n={len(cinc)}  SJLIFE={len(pairs)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pairs)

    arms=["clean","closed_aug","joint","oracle"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC N-vs-O / MORPHOLOGY (20 seeds) =====")
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
    print(f"  COMPARE AF (E51): joint +0.106 vs clean. Here (morphology): joint {cmp['joint_vs_clean'][0]:+.3f}")
    print(f"  oracle={A['oracle'].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
           color=["#999","#d4a017","#4a7","#333"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.01,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.set_ylim(0.5,1.0); ax.set_ylabel("AUROC CinC N-vs-O (morphology)")
    ax.set_title(f"E53 does alignment break rhythm-bound? joint−clean Δ={cmp['joint_vs_clean'][0]:+.3f} (p={cmp['joint_vs_clean'][2]:.2f}) vs AF +0.106")
    fig.tight_layout(); fig.savefig(RESULTS/"align_morphology.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'align_morphology.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"task":"Normal-vs-Other (morphological, harder)",
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "compare_AF_E51_joint_vs_clean":0.106,
                   "honesty":["CinC O heterogeneous catch-all (weak label)","PTB-XL abnormal != CinC O taxonomy",
                              "3 devices","lambda=0.1/temp=0.1 a priori","single clinical train set","AF excluded from clinical train"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
