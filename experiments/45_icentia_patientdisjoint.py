#!/usr/bin/env python3
"""
E45 — Icentia replication with PATIENT-DISJOINT splits (fixes E44's leakage).

*** ABANDONED / NOT RUN (2026-07-27) ***
AFIB is too sparse in Icentia's accessible patient block (26 patients scanned →
AF from only 1 patient), so a patient-disjoint AF/Normal split is not feasible
without an unbounded download. Kept for provenance. See EXPERIMENT_LOG.md E44 ⟲
UPDATE. The dose-response conclusion (E44) does not depend on this — it rests on
the leakage-free CinC gap (E42, 1 record/patient) vs Icentia's ~zero modality gap.
"""
from __future__ import annotations
import json, sys, glob, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "45_icentia_patientdisjoint"
RESULTS.mkdir(parents=True, exist_ok=True)
ICE = ROOT / "data" / "icentia_pt"
FS = 100.0; SIGLEN = 1000

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def load_icentia_pt():
    recs = []
    for lab, sub in [(1, "af"), (0, "normal")]:
        for f in sorted(glob.glob(str(ICE/sub/"*.npy"))):
            pid = re.match(r"(p\d+)_", Path(f).name).group(1)
            recs.append({"sig": np.load(f), "y": lab, "pid": pid})
    return recs

def patient_split(recs, seed, frac_ref=0.5):
    by_pt = defaultdict(list)
    for r in recs: by_pt[r["pid"]].append(r)
    pts = sorted(by_pt.keys())
    rng = np.random.default_rng(seed); rng.shuffle(pts)
    cut = int(len(pts)*frac_ref)
    ref_pts = set(pts[:cut]); test_pts = set(pts[cut:])
    ref = [r for r in recs if r["pid"] in ref_pts]
    test = [r for r in recs if r["pid"] in test_pts]
    return ref, test, len(ref_pts), len(test_pts)

def run_seed(seed, tr_leadI, tr_y, ice):
    torch.manual_seed(seed); np.random.seed(seed)
    ref, test, nrp, ntp = patient_split(ice, seed)
    # guard: both classes present in test
    if len(set(r["y"] for r in test)) < 2 or len(set(r["y"] for r in ref)) < 2:
        return None
    test_sigs=[r["sig"] for r in test]; test_y=[r["y"] for r in test]
    tgt_bw = float(np.mean([signal_modality_stats(r["sig"], FS)["bw_energy"] for r in ref[:200]]))

    out={}
    m=make_model(); e25.train(m, e25.SigDataset(tr_leadI,tr_y), epochs=20, tag=f"s{seed}-clean")
    out["clean"]=e25.evaluate(m,test_sigs,test_y)[0]

    clc=ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug=[clc.generate(x) for x in tr_leadI]
    m=make_model(); e25.train(m, e25.SigDataset(aug,tr_y), epochs=20, tag=f"s{seed}-closed")
    out["closed_loop"]=e25.evaluate(m,test_sigs,test_y)[0]

    ref_sigs=[r["sig"] for r in ref]; ref_y=[r["y"] for r in ref]
    m=make_model(); e25.train(m, e25.SigDataset(ref_sigs,ref_y), epochs=20, tag=f"s{seed}-oracle")
    out["oracle"]=e25.evaluate(m,test_sigs,test_y)[0]
    return out, tgt_bw, nrp, ntp

def main():
    print("Loading PTB-XL AF/NORM ...", flush=True)
    d = e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr_leadI=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    print(f"  n={len(tr_leadI)}", flush=True)
    ice = load_icentia_pt()
    npt = len(set(r["pid"] for r in ice)); ny=sum(r["y"] for r in ice)
    print(f"  Icentia n={len(ice)} (AF={ny}, N={len(ice)-ny}) from {npt} patients", flush=True)

    seeds=list(range(20)); per={}; bws=[]; splits=[]
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        r = run_seed(s, tr_leadI, tr_y, ice)
        if r is None:
            print("  skipped (class missing in a split)", flush=True); continue
        per[s], bw, nrp, ntp = r; bws.append(bw); splits.append((nrp,ntp))

    ok=list(per.keys())
    arms=["clean","closed_loop","oracle"]
    A={a: np.array([per[s][a] for s in ok]) for a in arms}
    from scipy import stats as sst
    print(f"\n===== AUROC on Icentia, PATIENT-DISJOINT ({len(ok)} seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    t,p=sst.ttest_rel(A["closed_loop"],A["clean"]); dz=(A["closed_loop"]-A["clean"]).mean()/((A["closed_loop"]-A["clean"]).std(ddof=1)+1e-9)
    wins=int((A["closed_loop"]>A["clean"]).sum())
    print(f"\n  closed−clean = {(A['closed_loop']-A['clean']).mean():+.3f} wins {wins}/{len(ok)} p={p:.4f} dz={dz:.2f}")
    print(f"  oracle now = {A['oracle'].mean():.3f} (E44 leaky oracle was 1.000 — did it drop?)")
    print(f"  mean target bw={np.mean(bws):.3f}; avg split ref/test patients={np.mean([s[0] for s in splits]):.0f}/{np.mean([s[1] for s in splits]):.0f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=["#999","#d4a017","#333"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.01,f"{A[a].mean():.3f}",ha="center")
    ax.set_ylim(0.5,1.02); ax.set_ylabel("AUROC Icentia (patient-disjoint)")
    ax.set_title(f"E45 honest split: Δ={(A['closed_loop']-A['clean']).mean():+.3f} (p={p:.2f}), oracle={A['oracle'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"patient_disjoint.png", dpi=110)
    print(f"\nSaved -> {RESULTS/'patient_disjoint.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(ok),"n":len(ice),"n_patients":npt,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "delta":float((A["closed_loop"]-A["clean"]).mean()),"wins":wins,"p":float(p),"dz":float(dz),
                   "oracle_leaky_E44":1.000,"oracle_disjoint_E45":float(A["oracle"].mean()),
                   "target_bw":float(np.mean(bws)),
                   "honesty":["patient-disjoint ref/test (no leakage)","Icentia chest-patch low-gap -> expected null by E44 dose-response",
                              "AF/NORM easy task","single fixed clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
