#!/usr/bin/env python3
"""
E49 — Multi-lead -> single-lead DISTILLATION: inject 12-lead clinical structure
into the single-lead (watch) student. Tests the E48 pivot: the +0.041 ceiling is
information-bound, so add NEW information instead of a cleverer invariance loss.

Idea: we have abundant clinical data as FULL 12-lead. The watch only ever sees
Lead-I. Train a 12-lead TEACHER on clinical (uses all leads' structure), then
distil its knowledge (Hinton soft-label KD) into a single-lead STUDENT that only
consumes Lead-I. The student inherits multi-lead structure it could never learn
from Lead-I alone -> potentially information the augmentation ceiling can't touch.

Arms (train PTB-XL AFIB/NORM, test REAL held-out CinC AF/N, 20 seeds):
  clean        : student CE on clean Lead-I (floor, = E48 clean)
  closed_aug   : E42 winner — student on closed-loop-calibrated Lead-I
  distill      : student CE(Lead-I) + T^2 * KL(student || 12-lead teacher)
  distill_aug  : distill + calibrated input (does info-injection STACK with aug?)
Teacher = 12-lead ECGResNet1d trained ONCE on PTB-XL, frozen, reused all seeds.

HONEST FLAGS: CinC finger != AW wrist; AF/NORM easy task; single clinical train
set; KD temp/alpha picked a priori (T=3, alpha=0.5), not tuned on test; teacher
trained on same PTB-XL cohort (no separate clinical holdout).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "49_distillation"
RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; KD_T = 3.0; KD_ALPHA = 0.5

def make_student(): return ECGResNet1d(n_leads=1, n_classes=2)
def make_teacher(): return ECGResNet1d(n_leads=12, n_classes=2)

def _norm1d(x):
    x = np.asarray(x, np.float64)
    if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.shape[0])])
    return ((x[:SIGLEN] - x[:SIGLEN].mean()) / (x[:SIGLEN].std() + 1e-6)).astype(np.float32)

def _norm12(sig):
    """sig: (T,12) -> (12,SIGLEN) per-lead z-normed."""
    sig = np.asarray(sig, np.float64)
    if sig.shape[0] < SIGLEN:
        sig = np.concatenate([sig, np.zeros((SIGLEN - sig.shape[0], sig.shape[1]))], axis=0)
    sig = sig[:SIGLEN]
    mu = sig.mean(0, keepdims=True); sd = sig.std(0, keepdims=True) + 1e-6
    return ((sig - mu) / sd).T.astype(np.float32)  # (12, SIGLEN)

# ---------- teacher (12-lead) ----------
class TeacherDS(Dataset):
    def __init__(self, recs): self.recs = recs
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        return torch.from_numpy(_norm12(r["ecg12"])), torch.tensor(r["y"])

def train_teacher(recs, epochs=20, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    m = make_teacher()
    dl = DataLoader(TeacherDS(recs), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); tot=0.0; nb=0
        for xb, yb in dl:
            opt.zero_grad(); loss = ce(m(xb), yb); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  [teacher] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    m.eval()
    return m

# ---------- student with KD ----------
class KDDataset(Dataset):
    """Yields (student_input_leadI, teacher_input_12lead, y). If calib given,
    student input is the calibrated Lead-I; teacher always sees clean 12-lead."""
    def __init__(self, recs, calib=None):
        self.recs = recs; self.calib = calib
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        li = self.calib.generate(r["leadI"]) if self.calib is not None else r["leadI"]
        xs = torch.from_numpy(_norm1d(li)[None])            # (1, SIGLEN)
        xt = torch.from_numpy(_norm12(r["ecg12"]))          # (12, SIGLEN)
        return xs, xt, torch.tensor(r["y"])

def train_student_kd(teacher, recs, calib=None, epochs=20, lr=1e-3,
                     T=KD_T, alpha=KD_ALPHA, tag="distill"):
    m = make_student()
    dl = DataLoader(KDDataset(recs, calib), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr); ce = nn.CrossEntropyLoss()
    teacher.eval()
    for ep in range(epochs):
        m.train(); tot=0.0; nb=0
        for xs, xt, yb in dl:
            opt.zero_grad()
            slog = m(xs)
            with torch.no_grad():
                tlog = teacher(xt)
            hard = ce(slog, yb)
            soft = F.kl_div(F.log_softmax(slog/T, 1), F.softmax(tlog/T, 1),
                            reduction="batchmean") * (T*T)
            loss = (1-alpha)*hard + alpha*soft
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        print(f"  [{tag}] ep {ep+1}/{epochs} loss={tot/nb:.4f}", flush=True)
    return m

def run_seed(seed, recs, cinc, teacher):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tr_leadI=[r["leadI"] for r in recs]; tr_y=[r["y"] for r in recs]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw, tr_leadI, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)

    out={}
    # clean
    m=make_student(); e25.train(m,e25.SigDataset(tr_leadI,tr_y),epochs=20,tag=f"s{seed}-clean"); out["clean"]=e25.evaluate(m,ts,ty)[0]
    # closed_aug
    aug=[clc.generate(x) for x in tr_leadI]
    m=make_student(); e25.train(m,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-aug"); out["closed_aug"]=e25.evaluate(m,ts,ty)[0]
    # distill
    m=train_student_kd(teacher, recs, calib=None, epochs=20, tag=f"s{seed}-distill"); out["distill"]=e25.evaluate(m,ts,ty)[0]
    # distill + aug
    m=train_student_kd(teacher, recs, calib=clc, epochs=20, tag=f"s{seed}-distill_aug"); out["distill_aug"]=e25.evaluate(m,ts,ty)[0]
    return out

def main():
    print("Loading PTB-XL AF/NORM (12-lead) ...", flush=True)
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    recs=d["train"]   # each has leadI, y, ecg12
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL train n={len(recs)}  CinC n={len(cinc)}", flush=True)

    print("\n== training 12-lead TEACHER (once, frozen) ==", flush=True)
    teacher=train_teacher(recs, epochs=20, seed=0)
    # sanity: teacher train-acc
    with torch.no_grad():
        tl=torch.stack([torch.from_numpy(_norm12(r["ecg12"])) for r in recs])
        pred=teacher(tl).argmax(1).numpy(); ya=np.array([r["y"] for r in recs])
    print(f"  teacher train-acc={(pred==ya).mean():.3f}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,recs,cinc,teacher)

    arms=["clean","closed_aug","distill","distill_aug"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:12s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean": pair(A["closed_aug"],A["clean"]),
      "distill_vs_clean":    pair(A["distill"],A["clean"]),
      "distill_aug_vs_clean":pair(A["distill_aug"],A["clean"]),
      "distill_vs_aug":      pair(A["distill"],A["closed_aug"]),
      "distill_aug_vs_aug":  pair(A["distill_aug"],A["closed_aug"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:24s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    cols=["#999","#d4a017","#4a7","#3577c2"]
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E49 12-lead→1-lead distillation  (distill−aug Δ={cmp['distill_vs_aug'][0]:+.3f} p={cmp['distill_vs_aug'][2]:.2f})")
    fig.tight_layout(); fig.savefig(RESULTS/"distillation.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'distillation.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"kd_T":KD_T,"kd_alpha":KD_ALPHA,
                   "means":{a:float(A[a].mean()) for a in arms},
                   "stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy task","single clinical train set",
                              "KD T=3 alpha=0.5 a priori not tuned","teacher on same PTB-XL cohort"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
