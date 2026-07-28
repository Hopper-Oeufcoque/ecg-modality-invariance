#!/usr/bin/env python3
"""
E44 — SECOND-DEVICE replication: does closed-loop calibration lift transfer on a
physically DIFFERENT real single-lead device (Icentia CardioSTAT chest-patch)?

E42 showed closed-loop calibration lifts clinical->CinC (KardiaMobile finger)
AUROC by +0.041 (p=0.009). External-validity question: does the SAME zero-label
mechanism generalize to a different single-lead device with different electrode
physics? If yes on two independent devices, the inductive case for "...therefore
Apple Watch wrist" is much stronger.

Setup (mirrors E42 exactly, only the test set changes):
  TRAIN: PTB-XL Lead-I AFIB vs NORM (real 12-lead clinical, ch0).
  TEST:  Icentia11k AF vs Normal (CardioSTAT chest-patch single-lead, mod Lead I,
         250->100 Hz), mined windows (data/icentia/{af,normal}).
  Calibrate to UNLABELED Icentia-ref bw_energy only (zero test labels).
  Arms: clean / closed_loop / oracle. 20 seeds. Paired stats vs clean.

HONEST FLAGS:
  - Icentia chest-patch is electrically CLOSER to clinical than a dry-electrode
    wrist watch; it is a SECOND real single-lead device for external validity,
    NOT a better AW proxy than CinC.
  - AF windows mined from AFIB rhythm regions (200 AF / 200 Normal), 21 patients
    -> limited patient diversity; windows within a patient are correlated.
  - AF/NORM easy task; single fixed clinical train set across seeds.
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

RESULTS = ROOT / "results" / "44_icentia_seconddevice"
RESULTS.mkdir(parents=True, exist_ok=True)
ICE = ROOT / "data" / "icentia"
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def load_icentia():
    af = [np.load(f) for f in sorted(glob.glob(str(ICE/"af"/"*.npy")))]
    nn = [np.load(f) for f in sorted(glob.glob(str(ICE/"normal"/"*.npy")))]
    recs = [{"sig": s, "y": 1} for s in af] + [{"sig": s, "y": 0} for s in nn]
    return recs

def run_seed(seed, tr_leadI, tr_y, ice):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(ice)); rng.shuffle(idx); cut = len(ice)//2
    ref = [ice[i] for i in idx[:cut]]; test = [ice[i] for i in idx[cut:]]
    test_sigs=[r["sig"] for r in test]; test_y=[r["y"] for r in test]
    tgt_bw = float(np.mean([signal_modality_stats(r["sig"], FS)["bw_energy"] for r in ref[:200]]))

    out = {}
    m = make_model(); e25.train(m, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"] = e25.evaluate(m, test_sigs, test_y)[0]

    clc = ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug = [clc.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug, tr_y), epochs=20, tag=f"s{seed}-closed")
    out["closed_loop"] = e25.evaluate(m, test_sigs, test_y)[0]

    ref_sigs=[r["sig"] for r in ref]; ref_y=[r["y"] for r in ref]
    m = make_model(); e25.train(m, e25.SigDataset(ref_sigs, ref_y), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(m, test_sigs, test_y)[0]
    return out, tgt_bw

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI = [r["leadI"] for r in d["train"]]; tr_y = [r["y"] for r in d["train"]]
    print(f"  n={len(tr_leadI)}", flush=True)
    ice = load_icentia()
    ny = sum(r["y"] for r in ice)
    print(f"  Icentia n={len(ice)} (AF={ny}, Normal={len(ice)-ny})", flush=True)

    seeds = list(range(20)); per = {}; bws=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s], bw = run_seed(s, tr_leadI, tr_y, ice); bws.append(bw)

    arms=["clean","closed_loop","oracle"]
    A = {a: np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on REAL Icentia (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    t,p = sst.ttest_rel(A["closed_loop"], A["clean"])
    try: w,pw = sst.wilcoxon(A["closed_loop"], A["clean"])
    except Exception: pw=float("nan")
    dz=(A["closed_loop"]-A["clean"]).mean()/((A["closed_loop"]-A["clean"]).std(ddof=1)+1e-9)
    wins=int((A["closed_loop"]>A["clean"]).sum())
    print(f"\n  closed_loop − clean = {(A['closed_loop']-A['clean']).mean():+.3f}  wins {wins}/20")
    print(f"  paired t={t:.2f} p={p:.4f} | Wilcoxon p={pw:.4f} | dz={dz:.2f}")
    print(f"  significant at 0.05? {'YES' if p<0.05 else 'NO'}")
    print(f"  mean unlabeled-Icentia target bw={np.mean(bws):.3f}")
    print(f"\n  CROSS-DEVICE COMPARISON:")
    print(f"    CinC (E42):    clean 0.701 -> closed 0.742  (+0.041, p=0.009)")
    print(f"    Icentia (E44): clean {A['clean'].mean():.3f} -> closed {A['closed_loop'].mean():.3f}  ({(A['closed_loop']-A['clean']).mean():+.3f}, p={p:.3f})")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    ax[0].bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
              color=["#999","#d4a017","#333"],capsize=4)
    for i,a in enumerate(arms): ax[0].text(i,A[a].mean()+0.01,f"{A[a].mean():.3f}",ha="center")
    ax[0].set_ylim(0.5,1.0); ax[0].set_ylabel("AUROC real Icentia"); ax[0].set_title(f"E44 2nd device (p={p:.3f})")
    ax[1].plot([0,1],[A["clean"],A["closed_loop"]],color="#bbb",lw=0.8)
    ax[1].scatter(np.zeros(20),A["clean"],color="#999",zorder=3,label="clean")
    ax[1].scatter(np.ones(20),A["closed_loop"],color="#d4a017",zorder=3,label="closed_loop")
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["clean","closed_loop"])
    ax[1].set_title(f"Paired (Δ={(A['closed_loop']-A['clean']).mean():+.3f}, {wins}/20)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS/"icentia_transfer.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'icentia_transfer.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"device":"Icentia11k CardioSTAT chest-patch single-lead",
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "delta":float((A["closed_loop"]-A["clean"]).mean()),"wins":wins,
                   "t":float(t),"p":float(p),"wilcoxon_p":float(pw),"dz":float(dz),"sig05":bool(p<0.05),
                   "target_bw":float(np.mean(bws)),
                   "cross_device":{"cinc_E42":{"clean":0.701,"closed":0.742,"delta":0.041,"p":0.009},
                                   "icentia_E44":{"clean":float(A["clean"].mean()),"closed":float(A["closed_loop"].mean()),
                                                  "delta":float((A["closed_loop"]-A["clean"]).mean()),"p":float(p)}},
                   "per_seed":{str(s):per[s] for s in seeds},
                   "honesty":["Icentia chest-patch electrically closer to clinical than wrist watch",
                              "200AF/200N from 21 patients -> within-patient correlation","AF/NORM easy task",
                              "single fixed clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__ == "__main__":
    main()
