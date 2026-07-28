#!/usr/bin/env python3
"""
E47 — Does closed-loop calibration survive a HARDER, morphology-driven task?

Everything so far (E41-E46) is AF-vs-Normal: the EASY distinctive-rhythm task
(E31 warned harder tasks generalize worse — AF changes heart-rate variability,
obvious even single-lead). For the north star to be credible for general
downstream tasks, the mechanism must help where the signal is MORPHOLOGICAL, not
rhythm-obvious.

Task: Normal vs Other/abnormal (NOT rhythm-defined).
  TRAIN (clinical): PTB-XL Lead-I, NORM vs non-NORM-non-AF abnormal (MI/STTC/CD/
         HYP superclasses) — morphological abnormality.
  TEST  (real): CinC 2017 N vs O (O = "other rhythm/abnormal", the catch-all
         non-AF-non-noise class). Real single-lead dry-finger.
  Calibrate to unlabeled CinC-ref bw. Arms clean/closed_loop/oracle. 20 seeds.

Prediction (from E31): the calibration lift will be SMALLER here than AF's
+0.041, possibly null, because the gap-closing helps rhythm robustness more than
morphology. Logging this honestly either way.

HONEST FLAGS: CinC "O" is a heterogeneous catch-all (weak label); PTB-XL
abnormal != CinC "O" taxonomy (approximate task mapping); CinC finger != AW
wrist; single clinical train set.
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

RESULTS = ROOT / "results" / "47_harder_task"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def load_ptbxl_norm_vs_abnormal(data_dir, max_per_class=700):
    import pandas as pd, ast, wfdb
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "ptbxl_database.csv")
    agg = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    diag = agg[agg.diagnostic == 1] if "diagnostic" in agg.columns else agg
    def superclasses(s):
        try: c = ast.literal_eval(s)
        except: return set()
        out = set()
        for code in c:
            if code in diag.index:
                sc = diag.loc[code].get("diagnostic_class", None)
                if isinstance(sc, str): out.add(sc)
        return out
    scs = df["scp_codes"].apply(superclasses)
    is_norm = scs.apply(lambda s: s == {"NORM"})
    # abnormal = has a morphological superclass, explicitly NOT norm (exclude pure-AF via scp too)
    def is_abn(s_codes, s_sc):
        try: c = ast.literal_eval(s_codes)
        except: return False
        if "AFIB" in c or "AFLT" in c: return False        # exclude rhythm-AF: keep it morphological
        return len(s_sc & {"MI","STTC","CD","HYP"}) > 0 and "NORM" not in s_sc
    abn = df.apply(lambda r: is_abn(r["scp_codes"], superclasses(r["scp_codes"])), axis=1)
    pos = df[abn].copy(); pos["y"] = 1
    neg = df[is_norm].copy(); neg["y"] = 0
    pos = pos.sample(frac=1.0, random_state=0).head(max_per_class)
    neg = neg.sample(frac=1.0, random_state=0).head(max_per_class)
    out = []
    for sub in (pos, neg):
        for _, row in sub.iterrows():
            try: sig,_ = wfdb.rdsamp(str(data_dir / row["filename_lr"]))
            except: continue
            leadI = sig[:SIGLEN,0].astype(np.float64)
            if leadI.size < SIGLEN: leadI = np.concatenate([leadI, np.zeros(SIGLEN-leadI.size)])
            leadI = (leadI-leadI.mean())/(leadI.std()+1e-9)
            out.append({"leadI":leadI.astype(np.float32),"y":int(row["y"])})
    return out

def load_cinc_n_vs_o(n_per_class=700, fs=300):
    from math import gcd
    import scipy.io as sio
    from scipy.signal import resample_poly
    data_dir = Path.home()/"data"/"cinc2017"/"training2017"
    ref = {}
    for line in (data_dir/"REFERENCE.csv").read_text().splitlines():
        p=line.strip().split(",")
        if len(p)==2: ref[p[0]]=p[1]
    g=gcd(int(fs),int(FS)); up=int(FS)//g; down=int(fs)//g
    O=[]; N=[]
    for mf in sorted(data_dir.glob("A*.mat")):
        lab=ref.get(mf.stem,"?")
        if lab not in ("N","O"): continue
        try: sig=sio.loadmat(mf)["val"][0].astype(np.float64)
        except: continue
        if sig.size<SIGLEN: continue
        sig=(sig-sig.mean())/(sig.std()+1e-9); sig=resample_poly(sig,up,down)
        rec={"sig":sig[:SIGLEN].astype(np.float32),"y":1 if lab=="O" else 0}
        (O if lab=="O" else N).append(rec)
    return O[:n_per_class], N[:n_per_class]

def run_seed(seed, tr, tr_y, cinc):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    out={}
    m=make_model(); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    clc=ClosedLoopCalibrator.fit(tgt_bw,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]
    m=make_model(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-closed"); out["closed_loop"]=e25.evaluate(m,ts,ty)[0]
    m=make_model(); e25.train(m,e25.SigDataset([r["sig"] for r in ref],[r["y"] for r in ref]),epochs=20,tag=f"s{seed}-oracle"); out["oracle"]=e25.evaluate(m,ts,ty)[0]
    return out, tgt_bw

def main():
    print("Loading PTB-XL NORM vs morphological-abnormal ...", flush=True)
    d=load_ptbxl_norm_vs_abnormal(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d]; tr_y=[r["y"] for r in d]
    print(f"  n={len(tr)} (abn={sum(tr_y)}, norm={len(tr_y)-sum(tr_y)})", flush=True)
    O,N=load_cinc_n_vs_o(n_per_class=700); cinc=O+N
    print(f"  CinC N-vs-O n={len(cinc)} (O={len(O)}, N={len(N)})", flush=True)

    seeds=list(range(20)); per={}; bws=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s],bw=run_seed(s,tr,tr_y,cinc); bws.append(bw)

    arms=["clean","closed_loop","oracle"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC N-vs-O (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    t,p=sst.ttest_rel(A["closed_loop"],A["clean"]); dz=(A["closed_loop"]-A["clean"]).mean()/((A["closed_loop"]-A["clean"]).std(ddof=1)+1e-9)
    wins=int((A["closed_loop"]>A["clean"]).sum())
    print(f"\n  closed−clean = {(A['closed_loop']-A['clean']).mean():+.3f} wins {wins}/20 p={p:.4f} dz={dz:.2f}")
    print(f"  COMPARE: AF/NORM (E42) was +0.041 p=0.009. Harder task here: {(A['closed_loop']-A['clean']).mean():+.3f} p={p:.3f}")
    print(f"  oracle={A['oracle'].mean():.3f}  target bw={np.mean(bws):.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=["#999","#d4a017","#333"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.01,f"{A[a].mean():.3f}",ha="center")
    ax.set_ylim(0.5,1.0); ax.set_ylabel("AUROC CinC N-vs-O"); ax.set_title(f"E47 harder task: Δ={(A['closed_loop']-A['clean']).mean():+.3f} (p={p:.2f}) vs AF +0.041")
    fig.tight_layout(); fig.savefig(RESULTS/"harder_task.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'harder_task.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"task":"Normal-vs-Other (morphological, harder)",
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "delta":float((A["closed_loop"]-A["clean"]).mean()),"wins":wins,"p":float(p),"dz":float(dz),
                   "compare_AF_E42":{"delta":0.041,"p":0.009},"oracle":float(A["oracle"].mean()),"target_bw":float(np.mean(bws)),
                   "honesty":["CinC O is heterogeneous catch-all (weak label)","PTB-XL abnormal != CinC O taxonomy (approx mapping)",
                              "CinC finger != AW wrist","single clinical train set","AF excluded from clinical train to keep it morphological"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
