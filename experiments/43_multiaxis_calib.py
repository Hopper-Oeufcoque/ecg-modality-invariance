#!/usr/bin/env python3
"""
E43 — Multi-axis closed-loop calibration: can closing hf too beat the +0.041
single-axis lift (E42)?

E40 showed single-axis (bw-only) closed-loop left residual gaps in hf/qrs/mid.
E43 adds a second closed-loop axis (high-freq noise -> target hf_energy) and
tests end-to-end AUROC on real CinC, head-to-head with the E42 single-axis
calibrator, at 20 seeds for direct comparability.

Arms (train PTB-XL Lead-I AFIB/NORM, test real held-out CinC AF/N):
  clean            floor
  closed_bw        single-axis (E42 winner, +0.041)
  closed_bw_hf     multi-axis (bw + hf), THIS experiment
  oracle           train-on-real ceiling

Calibration targets (bw AND hf) measured from UNLABELED CinC-ref (zero labels).
20 seeds. Paired stats vs clean AND vs single-axis.

HONEST FLAGS: CinC finger != AW wrist; AF/NORM easy task; single fixed clinical
train set across seeds (CI omits cohort variance).
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
from src.aw_generator import (ClosedLoopCalibrator, MultiAxisClosedLoopCalibrator,
                              signal_modality_stats)

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "43_multiaxis_calib"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def run_seed(seed, tr_leadI, tr_y, cinc):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc)//2
    ref = [cinc[i] for i in idx[:cut]]; test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in test]; test_y = [r["y"] for r in test]
    st = [signal_modality_stats(r["sig"], FS) for r in ref[:200]]
    tgt_bw = float(np.mean([s["bw_energy"] for s in st]))
    tgt_hf = float(np.mean([s["hf_energy"] for s in st]))

    out = {}
    m = make_model(); e25.train(m, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"] = e25.evaluate(m, test_sigs, test_y)[0]

    clc = ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug1 = [clc.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug1, tr_y), epochs=20, tag=f"s{seed}-bw")
    out["closed_bw"] = e25.evaluate(m, test_sigs, test_y)[0]

    mc = MultiAxisClosedLoopCalibrator.fit(tgt_bw, tgt_hf, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug2 = [mc.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug2, tr_y), epochs=20, tag=f"s{seed}-bwhf")
    out["closed_bw_hf"] = e25.evaluate(m, test_sigs, test_y)[0]

    ref_sigs=[r["sig"] for r in ref]; ref_y=[r["y"] for r in ref]
    m = make_model(); e25.train(m, e25.SigDataset(ref_sigs, ref_y), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(m, test_sigs, test_y)[0]
    return out, (tgt_bw, tgt_hf, clc.amp, mc.bw_amp, mc.hf_amp)

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI = [r["leadI"] for r in d["train"]]; tr_y = [r["y"] for r in d["train"]]
    print(f"  n={len(tr_leadI)}", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC n={len(cinc)}", flush=True)

    seeds = list(range(20)); per = {}; params=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s], pr = run_seed(s, tr_leadI, tr_y, cinc); params.append(pr)

    arms = ["clean","closed_bw","closed_bw_hf","oracle"]
    A = {a: np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms:
        print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")

    def paired(x, base):
        t,p = sst.ttest_rel(x, base); dz=(x-base).mean()/((x-base).std(ddof=1)+1e-9)
        return (x-base).mean(), int((x>base).sum()), float(p), float(dz)
    d_bw = paired(A["closed_bw"], A["clean"])
    d_hf = paired(A["closed_bw_hf"], A["clean"])
    d_vs = paired(A["closed_bw_hf"], A["closed_bw"])
    print(f"\n  closed_bw    − clean: Δ={d_bw[0]:+.3f} wins {d_bw[1]}/20 p={d_bw[2]:.4f} dz={d_bw[3]:.2f}")
    print(f"  closed_bw_hf − clean: Δ={d_hf[0]:+.3f} wins {d_hf[1]}/20 p={d_hf[2]:.4f} dz={d_hf[3]:.2f}")
    print(f"  closed_bw_hf − closed_bw (does HF axis help?): Δ={d_vs[0]:+.3f} wins {d_vs[1]}/20 p={d_vs[2]:.4f} dz={d_vs[3]:.2f}")
    pr = np.array(params)
    print(f"\n  mean targets bw={pr[:,0].mean():.3f} hf={pr[:,1].mean():.3f}; multi bw_amp={pr[:,3].mean():.3f} hf_amp={pr[:,4].mean():.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    means=[A[a].mean() for a in arms]; stds=[A[a].std() for a in arms]
    ax.bar(arms, means, yerr=stds, color=["#999","#d4a017","#4a4","#333"], capsize=4)
    for i,m in enumerate(means): ax.text(i,m+0.008,f"{m:.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["oracle"].mean(),color="r",ls="--",lw=1)
    ax.set_ylim(0.5,1.0); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E43 multi-axis: bw_hf−bw Δ={d_vs[0]:+.3f} (p={d_vs[2]:.3f})"); ax.tick_params(axis="x",rotation=12)
    fig.tight_layout(); fig.savefig(RESULTS/"multiaxis.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'multiaxis.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),
                   "means":{a:float(A[a].mean()) for a in arms},
                   "stds":{a:float(A[a].std()) for a in arms},
                   "closed_bw_vs_clean":{"delta":d_bw[0],"wins":d_bw[1],"p":d_bw[2],"dz":d_bw[3]},
                   "closed_bw_hf_vs_clean":{"delta":d_hf[0],"wins":d_hf[1],"p":d_hf[2],"dz":d_hf[3]},
                   "multiaxis_vs_singleaxis":{"delta":d_vs[0],"wins":d_vs[1],"p":d_vs[2],"dz":d_vs[3]},
                   "targets":{"bw":float(pr[:,0].mean()),"hf":float(pr[:,1].mean())},
                   "per_seed":{str(s):per[s] for s in seeds},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy task","single fixed clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__ == "__main__":
    main()
