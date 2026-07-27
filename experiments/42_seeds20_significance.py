#!/usr/bin/env python3
"""
E42 — Settle E41's significance: clean vs closed-loop, 20 seeds.

E41 found closed-loop-calibrated clinical training beats clean by +0.072 on real
CinC (5/5 seeds) but p=0.086 (5 seeds, underpowered). E42 reruns ONLY the two
decisive arms (clean, closed_loop) plus oracle for reference, at 20 seeds, to
settle whether the effect is significant at 0.05.

Same protocol as E41: train PTB-XL Lead-I AFIB/NORM, test held-out real CinC
AF/N, calibrate to UNLABELED CinC-ref bw only.

HONEST FLAGS (carried from E41): CinC finger != Apple Watch wrist; AF/NORM easy
task; single-site clinical; calibration zero-label.
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

RESULTS = ROOT / "results" / "42_seeds20_significance"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def run_seed(seed, tr_leadI, tr_y, cinc):
    rng = np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc)//2
    ref = [cinc[i] for i in idx[:cut]]; test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in test]; test_y = [r["y"] for r in test]
    tgt_bw = float(np.mean([signal_modality_stats(r["sig"], FS)["bw_energy"] for r in ref[:200]]))

    m = make_model(); e25.train(m, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    clean = e25.evaluate(m, test_sigs, test_y)[0]

    clc = ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug = [clc.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug, tr_y), epochs=20, tag=f"s{seed}-closed")
    closed = e25.evaluate(m, test_sigs, test_y)[0]

    ref_sigs=[r["sig"] for r in ref]; ref_y=[r["y"] for r in ref]
    m = make_model(); e25.train(m, e25.SigDataset(ref_sigs, ref_y), epochs=20, tag=f"s{seed}-oracle")
    oracle = e25.evaluate(m, test_sigs, test_y)[0]
    return {"clean": clean, "closed_loop": closed, "oracle": oracle}

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI = [r["leadI"] for r in d["train"]]; tr_y = [r["y"] for r in d["train"]]
    print(f"  n={len(tr_leadI)}", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC n={len(cinc)}", flush=True)

    seeds = list(range(20))
    per = {}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s] = run_seed(s, tr_leadI, tr_y, cinc)

    cl = np.array([per[s]["closed_loop"] for s in seeds])
    cn = np.array([per[s]["clean"] for s in seeds])
    orc = np.array([per[s]["oracle"] for s in seeds])
    from scipy import stats as sst
    t, p = sst.ttest_rel(cl, cn)
    try:
        w, pw = sst.wilcoxon(cl, cn)
    except Exception:
        w, pw = float("nan"), float("nan")
    d_eff = (cl-cn).mean() / ((cl-cn).std(ddof=1) + 1e-9)

    print("\n===== 20-seed significance (AUROC on real CinC) =====")
    print(f"  clean       {cn.mean():.3f} ± {cn.std():.3f}")
    print(f"  closed_loop {cl.mean():.3f} ± {cl.std():.3f}")
    print(f"  oracle      {orc.mean():.3f} ± {orc.std():.3f}")
    print(f"\n  Δ closed−clean = {(cl-cn).mean():+.3f}  wins {int((cl>cn).sum())}/20")
    print(f"  paired t={t:.2f} p={p:.4f}  |  Wilcoxon p={pw:.4f}  |  Cohen dz={d_eff:.2f}")
    sig = "YES (p<0.05)" if p < 0.05 else "NO"
    print(f"  >>> significant at 0.05? {sig} <<<")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    ax[0].bar(["clean","closed_loop","oracle"], [cn.mean(),cl.mean(),orc.mean()],
              yerr=[cn.std(),cl.std(),orc.std()], color=["#999","#d4a017","#333"], capsize=4)
    for i,m in enumerate([cn.mean(),cl.mean(),orc.mean()]): ax[0].text(i,m+0.01,f"{m:.3f}",ha="center")
    ax[0].set_ylim(0.5,1.0); ax[0].set_ylabel("AUROC real CinC"); ax[0].set_title(f"20 seeds (p={p:.3f})")
    # paired scatter
    ax[1].plot([0,1],[cn,cl],color="#bbb",lw=0.8)
    ax[1].scatter(np.zeros_like(cn),cn,color="#999",label="clean",zorder=3)
    ax[1].scatter(np.ones_like(cl),cl,color="#d4a017",label="closed_loop",zorder=3)
    ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["clean","closed_loop"])
    ax[1].set_title(f"Paired per-seed (Δ={(cl-cn).mean():+.3f}, {int((cl>cn).sum())}/20)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(RESULTS/"significance.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'significance.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),
                   "clean_mean":float(cn.mean()),"clean_std":float(cn.std()),
                   "closed_mean":float(cl.mean()),"closed_std":float(cl.std()),
                   "oracle_mean":float(orc.mean()),
                   "delta":float((cl-cn).mean()),"wins":int((cl>cn).sum()),
                   "t":float(t),"p":float(p),"wilcoxon_p":float(pw),"cohen_dz":float(d_eff),
                   "significant_05": bool(p<0.05),
                   "per_seed":{str(s):per[s] for s in seeds},
                   "honesty":["CinC finger != Apple Watch wrist","AF/NORM easy task",
                              "single-site clinical","calibration zero-label (unlabeled CinC bw)"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__ == "__main__":
    main()
