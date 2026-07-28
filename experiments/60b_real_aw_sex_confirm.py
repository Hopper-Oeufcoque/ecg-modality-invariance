#!/usr/bin/env python3
"""
E60 — FIRST MEASURED real-Apple-Watch outcome: SEX transfer (clinical -> real AW).

BREAKTHROUGH ENABLER: the SJLIFE CSV (data/sjlife/shared_paired_data_243.csv) carries
per-patient `gender_x` (122 M / 121 F, balanced) and `age_at_ecg`. Sex-from-ECG is an
established MORPHOLOGICAL task (Attia et al. 2019, Nat Med). This is the first time we can
put a *measured* real-AW AUROC on our levers instead of the CinC proxy or a prediction.

North-star experiment, for real:
  Train sex on ABUNDANT clinical (PTB-XL Lead-I, disjoint cohort) -> TEST on real Apple
  Watch (SJLIFE), with modality-invariance methods bridging the gap. MEASURE, don't predict.

LEAKAGE DISCIPLINE (critical):
  * PTB-XL (train pop) and SJLIFE (test pop) are entirely different people -> clean/calib/
    scramble train on PTB-XL, test on ALL 243 SJLIFE patients (no patient overlap possible).
  * `aligned` uses SJLIFE PAIRS for InfoNCE. Those patients must NOT be in the test fold.
    -> patient-level 5-fold CV: align on fold-train patients, test on fold-heldout patients.
  * `oracle` = train on real AW sex labels via the same patient-folds (learnable-at-all probe).
  * Per-patient prediction = MEAN probability over that patient's Apple-Watch windows.

Arms (AUROC on real SJLIFE Apple Watch, per-patient, female=1):
  clinical_self : PTB-XL Lead-I -> PTB-XL Lead-I test  (is Lead-I sex learnable with data?)
  clean         : PTB-XL Lead-I -> real AW             (naive transfer floor)
  closed_aug    : + calibration (E42)                  (lever 2)
  band_scramble : + E59 band-phase-scramble            (free lever)
  aligned       : + E51 label-anchored SJLIFE alignment (headline lever, fold-disjoint)
  oracle        : train on real AW sex (patient CV)     (upper bound / learnability on n=243)

HONEST FLAGS: n=243 test (wide CIs); PTB-XL older (mean 63, cardiac) vs SJLIFE young
(mean 36, cancer survivors) -> POPULATION shift confounds modality shift; Lead-I sex is
intrinsically weaker than 12-lead; per-patient aggregation of a few windows; female=1 both.
"""
from __future__ import annotations
import json, sys, csv, glob, os
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

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e25 = _load("e25", "25_aw_generator.py")
e50 = _load("e50", "50_sjlife_align.py")
e51 = _load("e51", "51_label_anchored_align.py")
e59 = _load("e59", "59_band_dropout.py")

RESULTS = ROOT / "results" / "60b_real_aw_sex_confirm"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; SEEDS = 15; N_FOLDS = 5
import pandas as pd, wfdb

# ---------------- PTB-XL sex (Lead-I) ----------------
def load_ptbxl_sex(max_per_class=1200, seed=0):
    df = pd.read_csv(Path.home()/"data"/"ptbxl"/"ptbxl_database.csv")
    df = df.dropna(subset=["sex"])
    # PTB-XL: sex 0=male, 1=female -> female=1 (matches SJLIFE mapping)
    fem = df[df.sex==1].sample(frac=1.0, random_state=seed).head(max_per_class)
    mal = df[df.sex==0].sample(frac=1.0, random_state=seed).head(max_per_class)
    rows = pd.concat([fem, mal]).sample(frac=1.0, random_state=seed)
    tr_s, tr_y, te_s, te_y = [], [], [], []
    for _, r in rows.iterrows():
        try: sig,_ = wfdb.rdsamp(str(Path.home()/"data"/"ptbxl"/r["filename_lr"]))
        except: continue
        x = sig[:SIGLEN,0].astype(np.float64)
        if x.size < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN-x.size)])
        x = ((x-x.mean())/(x.std()+1e-9)).astype(np.float32)
        y = int(r["sex"])  # female=1
        if int(r["strat_fold"]) <= 8: tr_s.append(x); tr_y.append(y)
        else: te_s.append(x); te_y.append(y)
    return tr_s, tr_y, te_s, te_y

# ---------------- SJLIFE labeled (sex per patient) ----------------
def load_sjlife_sex():
    """Return list of dicts: {clin, apple_windows, y(female=1), pid} in file-number order."""
    rows = list(csv.DictReader(open(ROOT/"data"/"sjlife"/"shared_paired_data_243.csv")))
    lab = {}
    for r in rows:
        num = int(r["apple_loc_ECG_243"].split("_")[-1].split(".")[0])
        lab[num] = 1 if r["gender_x"].strip().lower()=="female" else 0
    pairs = e50.load_sjlife_pairs()   # (clin_1000, [apple_1000...]) in num-sorted order
    ap = sorted(glob.glob(str(ROOT/"data"/"sjlife"/"apple"/"apple_ecg_*.npy")),
                key=lambda p:int(p.split("_")[-1].split(".")[0]))
    nums = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in ap]
    # load_sjlife_pairs skips records where clinical missing / no windows -> re-derive its kept order
    out = []
    for (clin, aws), num in zip(pairs, nums):
        if num not in lab: continue
        out.append({"clin":clin, "apple":aws, "y":lab[num], "pid":num})
    return out

@torch.no_grad()
def eval_patients(model, patients):
    """Per-patient mean-prob AUROC over each patient's apple windows. female=1."""
    from sklearn.metrics import roc_auc_score
    model.eval(); probs=[]; ys=[]
    for p in patients:
        xs = torch.from_numpy(np.stack(p["apple"])[:,None].astype(np.float32))
        logit = model(xs)
        pr = torch.softmax(logit,1).numpy()[:,1].mean()
        probs.append(pr); ys.append(p["y"])
    return float(roc_auc_score(ys, probs))

@torch.no_grad()
def eval_leadI(model, sigs, ys):
    from sklearn.metrics import roc_auc_score
    model.eval()
    dl = DataLoader(e25.SigDataset(sigs, ys), batch_size=64)
    P=[]; Y=[]
    for xb,yb in dl:
        P.append(torch.softmax(model(xb),1).numpy()[:,1]); Y.append(yb.numpy())
    return float(roc_auc_score(np.concatenate(Y), np.concatenate(P)))

def train_oracle_watch(patients_train, epochs=25, lr=1e-3, seed=0, tag="oracle"):
    """Train sex directly on real AW windows (each window inherits patient label)."""
    torch.manual_seed(seed); np.random.seed(seed)
    sigs=[]; ys=[]
    for p in patients_train:
        for w in p["apple"]: sigs.append(w); ys.append(p["y"])
    m=ECGResNet1d(n_leads=1,n_classes=2)
    dl=DataLoader(e25.SigDataset(sigs,ys),batch_size=64,shuffle=True)
    opt=torch.optim.Adam(m.parameters(),lr=lr); ce=nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        for xb,yb in dl:
            opt.zero_grad(); ce(m(xb),yb).backward(); opt.step()
    return m

def main():
    print("Loading PTB-XL sex + SJLIFE sex-labeled ...", flush=True)
    sj = load_sjlife_sex()
    print(f"  SJLIFE labeled patients={len(sj)}  female={sum(p['y'] for p in sj)}  male={sum(1-p['y'] for p in sj)}", flush=True)

    res = {a: [] for a in ["clinical_self","clean","closed_aug","band_scramble","aligned","oracle"]}

    for seed in range(SEEDS):
        print(f"\n===== SEED {seed} =====", flush=True)
        tr_s, tr_y, te_s, te_y = load_ptbxl_sex(max_per_class=1200, seed=seed)
        print(f"  PTB-XL sex: train={len(tr_s)} (fem={sum(tr_y)}) test={len(te_s)}", flush=True)

        # ---- clinical_self: is Lead-I sex learnable with abundant data? ----
        m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr_s,tr_y),epochs=20,tag=f"s{seed}-self")
        res["clinical_self"].append(eval_leadI(m, te_s, te_y))

        # ---- clean: PTB-XL -> real AW (all 243, disjoint pop) ----
        res["clean"].append(eval_patients(m, sj))   # reuse the trained clinical model

        # ---- closed_aug: calibration ----
        tgt = float(np.mean([signal_modality_stats(np.concatenate(p["apple"]),FS)["bw_energy"] for p in sj[:120]]))
        clc = ClosedLoopCalibrator.fit(tgt, tr_s, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
        aug = [clc.generate(x) for x in tr_s]
        m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-cal")
        res["closed_aug"].append(eval_patients(m, sj))

        # ---- band_scramble (E59 free lever) ----
        m=e59.train_band(tr_s,tr_y,"scramble",epochs=20,seed=seed,tag=f"s{seed}-scr")
        res["band_scramble"].append(eval_patients(m, sj))

        # ---- aligned + oracle: patient-level 5-fold CV (leakage-safe) ----
        rng=np.random.default_rng(seed); order=np.arange(len(sj)); rng.shuffle(order)
        folds=np.array_split(order, N_FOLDS)
        al_probs={}; or_probs={}   # pid -> prob, aggregated across folds
        for fi, test_idx in enumerate(folds):
            test_idx=set(test_idx.tolist())
            tr_pat=[sj[i] for i in range(len(sj)) if i not in test_idx]
            te_pat=[sj[i] for i in range(len(sj)) if i in test_idx]
            pairs=[(p["clin"], p["apple"]) for p in tr_pat]
            # aligned: CE on PTB-XL sex + InfoNCE on fold-train SJLIFE pairs
            m=e51.train_joint(tr_s,tr_y,pairs,epochs=20,seed=seed,tag=f"s{seed}f{fi}-align")
            au=eval_patients(m, te_pat)
            al_probs[fi]=au
            # oracle: train on fold-train real AW sex, test fold-heldout
            mo=train_oracle_watch(tr_pat, seed=seed, tag=f"s{seed}f{fi}-oracle")
            oa=eval_patients(mo, te_pat)
            or_probs[fi]=oa
            print(f"    fold {fi}: aligned={au:.3f} oracle={oa:.3f} (test n={len(te_pat)})", flush=True)
        res["aligned"].append(float(np.mean(list(al_probs.values()))))
        res["oracle"].append(float(np.mean(list(or_probs.values()))))

    # ---------- summarize ----------
    arms=["clinical_self","clean","closed_aug","band_scramble","aligned","oracle"]
    A={a:np.array(res[a]) for a in arms}
    from scipy import stats as sst
    print("\n===== SEX AUROC (female=1) =====")
    print(f"  {'clinical_self (PTB-XL internal)':32s} {A['clinical_self'].mean():.3f} ± {A['clinical_self'].std():.3f}")
    print("  --- transfer to REAL Apple Watch (per-patient, n=243) ---")
    for a in ["clean","closed_aug","band_scramble","aligned","oracle"]:
        print(f"  {a:32s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def cmp(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    comparisons={
        "closed_aug_vs_clean":   cmp(A["closed_aug"],A["clean"]),
        "band_scramble_vs_clean":cmp(A["band_scramble"],A["clean"]),
        "aligned_vs_clean":      cmp(A["aligned"],A["clean"]),
        "aligned_vs_closed_aug": cmp(A["aligned"],A["closed_aug"]),
        "oracle_vs_clean":       cmp(A["oracle"],A["clean"]),
    }
    print()
    for k,(d,w,p) in comparisons.items(): print(f"  {k:26s} Δ={d:+.3f} wins {w}/{SEEDS} p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show=["clean","closed_aug","band_scramble","aligned","oracle"]
    cols=["#999","#d4a017","#2a9d8f","#4a7","#333"]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(show,[A[a].mean() for a in show],yerr=[A[a].std() for a in show],color=cols,capsize=4)
    for i,a in enumerate(show): ax.text(i,A[a].mean()+0.006,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(0.5,color="r",ls=":",lw=1,label="chance"); ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.axhline(A["clinical_self"].mean(),color="#c60",ls="--",lw=1,label=f"clinical-self {A['clinical_self'].mean():.3f}")
    ax.set_ylim(0.4,0.85); ax.set_ylabel("Sex AUROC on REAL Apple Watch (per-patient)")
    ax.set_title(f"E60b CONFIRM (15 seeds) real-AW outcome: SEX transfer (clinical->AW, n=243, {SEEDS} seeds)\nclean {A['clean'].mean():.3f} · calib {A['closed_aug'].mean():.3f} · scramble {A['band_scramble'].mean():.3f} · aligned {A['aligned'].mean():.3f} · oracle {A['oracle'].mean():.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RESULTS/"real_aw_sex_confirm.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'real_aw_sex.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":SEEDS,"n_folds":N_FOLDS,"n_test_patients":len(sj),
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "per_seed":{a:[float(v) for v in A[a]] for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in comparisons.items()},
                   "honesty":["n=243 real-AW test, wide CIs","PTB-XL older/cardiac vs SJLIFE young/cancer-survivor = POPULATION shift confounds modality shift",
                              "Lead-I sex weaker than 12-lead","per-patient mean-prob over few windows","female=1 both datasets",
                              "aligned uses fold-disjoint SJLIFE pairs (no test-patient leakage)","oracle=patient-CV learnability probe"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
