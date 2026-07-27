#!/usr/bin/env python3
"""
E41 — END-TO-END AUROC: does closed-loop-calibrated clinical training data
transfer to REAL labeled single-lead ECG?

This is the experiment the whole project has been building toward: a REAL
downstream AUROC on REAL labeled single-lead data, testing whether the E40
closed-loop calibrator (which we proved hits the real modality profile) actually
LIFTS transfer accuracy — the E25b necessary-vs-sufficient question, answered.

Setup (real labels on BOTH ends):
  TRAIN (clinical): PTB-XL Lead-I, AFIB vs NORM (12-lead clinical, ch0 only).
  TEST  (real single-lead): CinC 2017, AF vs Normal (AliveCor KardiaMobile,
         dry-electrode finger ECG, 300 Hz -> 100 Hz). Same dry-electrode
         single-lead regime as Apple Watch; the closest labeled real proxy.

Arms (train on PTB-XL Lead-I, all tested on the SAME held-out real CinC):
  V1 clean          : clinical Lead-I, no augmentation (floor)
  V2 light_DR       : StochasticAWAugmenter strength 0.5 (E37 recommendation)
  V3 closed_loop    : ClosedLoopCalibrator.fit to UNLABELED CinC-ref bw (E40)
  V4 clean+closed   : cocktail (clean ∪ closed-loop) — E26 lesson: cocktail wins
  V5 oracle         : train on real CinC itself (upper bound)

Protocol: split CinC 50/50 into ref (unlabeled, for calibration ONLY) + test
(labeled, held out). 5 seeds. AUROC + acc. Calibration uses ONLY unlabeled CinC
stats (legitimate zero-label domain knowledge, same as E33).

HONEST FLAGS:
  - CinC KardiaMobile != Apple Watch (finger vs wrist dry electrode) — it is the
    closest LABELED real single-lead proxy, not real AW. Real AW (SJLIFE/HOME)
    has no labels.
  - AF vs NORM is the "easy" distinctive-rhythm task (E31: harder tasks don't
    generalize as well). This is a transfer-mechanism test, not a universal claim.
  - PTB-XL AF n capped per class; single-site clinical source.
"""
from __future__ import annotations
import json, sys, copy
from pathlib import Path
import numpy as np
import torch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import (StochasticAWAugmenter, ClosedLoopCalibrator,
                              signal_modality_stats)

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "41_endtoend_auroc"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000

def make_model():
    return ECGResNet1d(n_leads=1, n_classes=2)

def run_seed(seed, tr_leadI, tr_y, cinc):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed); np.random.seed(seed)
    # split real CinC: ref (unlabeled -> calibration only) + test (labeled, held out)
    idx = np.arange(len(cinc)); rng.shuffle(idx); cut = len(cinc)//2
    ref = [cinc[i] for i in idx[:cut]]; test = [cinc[i] for i in idx[cut:]]
    test_sigs = [r["sig"] for r in test]; test_y = [r["y"] for r in test]

    # unlabeled target bw_energy from CinC ref (no labels used)
    tgt_bw = float(np.mean([signal_modality_stats(r["sig"], FS)["bw_energy"] for r in ref[:200]]))

    out = {}
    # V1 clean
    m = make_model(); e25.train(m, e25.SigDataset(tr_leadI, tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"] = e25.evaluate(m, test_sigs, test_y)

    # V2 light-DR
    light = StochasticAWAugmenter(fs=FS, siglen=SIGLEN, strength=0.5, seed=seed)
    aug_l = [light.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug_l, tr_y), epochs=20, tag=f"s{seed}-light")
    out["light_DR"] = e25.evaluate(m, test_sigs, test_y)

    # V3 closed-loop calibrated (fit to unlabeled CinC bw)
    clc = ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug_c = [clc.generate(x) for x in tr_leadI]
    m = make_model(); e25.train(m, e25.SigDataset(aug_c, tr_y), epochs=20, tag=f"s{seed}-closed")
    out["closed_loop"] = e25.evaluate(m, test_sigs, test_y)

    # V4 cocktail clean + closed
    m = make_model(); e25.train(m, e25.SigDataset(tr_leadI + aug_c, tr_y + tr_y), epochs=20, tag=f"s{seed}-cocktail")
    out["clean+closed"] = e25.evaluate(m, test_sigs, test_y)

    # V5 oracle (train on real CinC ref labels)
    ref_sigs = [r["sig"] for r in ref]; ref_y = [r["y"] for r in ref]
    m = make_model(); e25.train(m, e25.SigDataset(ref_sigs, ref_y), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"] = e25.evaluate(m, test_sigs, test_y)

    return {k: {"auroc": v[0], "acc": v[1]} for k, v in out.items()}, tgt_bw, clc.amp

def main():
    print("Loading PTB-XL AF/NORM (clinical Lead-I) ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI = [r["leadI"] for r in d["train"]]; tr_y = [r["y"] for r in d["train"]]
    print(f"  PTB-XL train n={len(tr_leadI)} (AF={sum(tr_y)}, NORM={len(tr_y)-sum(tr_y)})", flush=True)

    print("Loading REAL CinC 2017 (labeled single-lead) ...", flush=True)
    cA, cN = e25.load_cinc(n_per_class=700); cinc = cA + cN
    print(f"  CinC n={len(cinc)} (AF={len(cA)}, N={len(cN)})", flush=True)

    seeds = [0,1,2,3,4]
    per_seed = {}; bws=[]; amps=[]
    for s in seeds:
        print(f"\n===== seed {s} =====", flush=True)
        per_seed[s], bw, amp = run_seed(s, tr_leadI, tr_y, cinc)
        bws.append(bw); amps.append(amp)

    arms = list(per_seed[seeds[0]].keys())
    agg = {}
    print("\n===== AUROC on REAL CinC (5 seeds, mean±std) =====")
    for a in arms:
        vals = np.array([per_seed[s][a]["auroc"] for s in seeds])
        agg[a] = {"auroc_mean": float(vals.mean()), "auroc_std": float(vals.std()),
                  "auroc_seeds": vals.tolist()}
        print(f"  {a:14s} {vals.mean():.3f} ± {vals.std():.3f}   seeds={np.round(vals,3).tolist()}")

    # key deltas + paired test closed_loop vs clean
    from scipy import stats as sst
    cl = np.array([per_seed[s]["closed_loop"]["auroc"] for s in seeds])
    cn = np.array([per_seed[s]["clean"]["auroc"] for s in seeds])
    ck = np.array([per_seed[s]["clean+closed"]["auroc"] for s in seeds])
    t1,p1 = sst.ttest_rel(cl, cn); t2,p2 = sst.ttest_rel(ck, cn)
    print(f"\n  closed_loop − clean = {(cl-cn).mean():+.3f}  (paired t={t1:.2f}, p={p1:.3f}, wins {int((cl>cn).sum())}/5)")
    print(f"  clean+closed − clean = {(ck-cn).mean():+.3f}  (paired t={t2:.2f}, p={p2:.3f}, wins {int((ck>cn).sum())}/5)")
    print(f"  oracle = {agg['oracle']['auroc_mean']:.3f}  (real-data upper bound)")
    print(f"\n  mean unlabeled-CinC target bw={np.mean(bws):.3f}, calibrated amp={np.mean(amps):.3f}")

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    order = ["clean","light_DR","closed_loop","clean+closed","oracle"]
    means = [agg[a]["auroc_mean"] for a in order]; stds=[agg[a]["auroc_std"] for a in order]
    cols = ["#999","#6aa","#d4a017","#4a4","#333"]
    ax.bar(order, means, yerr=stds, color=cols, capsize=4)
    for i,m in enumerate(means): ax.text(i, m+0.01, f"{m:.3f}", ha="center", fontsize=10)
    ax.axhline(agg["clean"]["auroc_mean"], color="k", ls=":", lw=1, label="clean floor")
    ax.axhline(agg["oracle"]["auroc_mean"], color="r", ls="--", lw=1, label="oracle ceiling")
    ax.set_ylabel("AUROC on REAL CinC 2017 (AF vs N)"); ax.set_ylim(0.5,1.0)
    ax.set_title("E41: clinical-train → real single-lead transfer (5 seeds)"); ax.legend()
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(RESULTS/"endtoend_auroc.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'endtoend_auroc.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"agg":agg, "per_seed":per_seed,
                   "deltas":{"closed_minus_clean":float((cl-cn).mean()),"p_closed":float(p1),
                             "cocktail_minus_clean":float((ck-cn).mean()),"p_cocktail":float(p2)},
                   "target_bw_mean":float(np.mean(bws)),"calib_amp_mean":float(np.mean(amps)),
                   "honesty":["CinC KardiaMobile finger ECG != Apple Watch wrist (closest LABELED proxy)",
                              "AF vs NORM easy task (E31 harder tasks generalize worse)",
                              "PTB-XL single-site clinical source, n capped 700/class",
                              "calibration uses UNLABELED CinC stats only (zero test labels)"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__ == "__main__":
    main()
