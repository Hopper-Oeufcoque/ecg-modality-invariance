#!/usr/bin/env python3
"""
E61 (N3) — Per-clip test-time adaptation via cardiac-cycle self-consistency.

Realistic deployment = a single 30 s watch clip, no labels, adapt on the spot. Novel
ECG-specific TTA: the BEATS within one clip are natural same-label views (one clip = one
rhythm), so (a) they give a mini-batch that makes single-clip BN adaptation well-posed, and
(b) their predictions should be CONSISTENT. Adapt only BN affine params (γ,β), reset per clip.

Views from one clip:
  beat_consist : K crops of 1000 samples, each phase-LOCKED to a detected R-peak (circular) —
                 cardiac-cycle-aware natural augmentation.
  shift_consist: K crops by RANDOM circular shift (NOT beat-locked) — control isolating whether
                 cardiac alignment matters vs generic shift-consistency / plain TENT.
TTA loss (both): mean prediction ENTROPY (confidence, TENT) + cross-view CONSISTENCY
  (each view's softmax -> KL to the batch-mean softmax). BN affine only, few steps, per clip.
Predict = mean softmax over the clip's views.

Arms (train PTB-XL AF/N Lead-I, test real CinC, 10 seeds):
  clean         : no adaptation (floor)
  closed_aug    : calibration (E42, train-time reference — not TTA)
  shift_consist : TTA with random-shift views (generic)
  beat_consist  : TTA with beat-locked views (N3, cardiac-aware)

Read:
  beat_consist >> clean AND > shift_consist -> cardiac-cycle TTA adds real, ECG-specific value
  beat_consist ~ shift_consist > clean       -> TTA helps but cardiac locking is not the reason
  beat_consist ~ clean                        -> per-clip label-free TTA doesn't move this task

HONEST FLAGS: CinC finger != AW wrist; AF/N easy; BN-affine-only TTA; K,steps,lr a-priori;
circular-shift wrap artifact; 10 seeds.
"""
from __future__ import annotations
import json, sys, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, find_peaks
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e25 = _load("e25", "25_aw_generator.py")

RESULTS = ROOT / "results" / "61_cardiac_tta"; RESULTS.mkdir(parents=True, exist_ok=True)
FS = 100.0; SIGLEN = 1000; K_VIEWS = 6; TTA_STEPS = 5; TTA_LR = 5e-3

_bh, _ah = butter(3, 5.0/(FS/2), btype="high")

def _rpeaks(x):
    h = filtfilt(_bh, _ah, x)
    pk,_ = find_peaks(h, height=np.std(h)*1.2, distance=int(0.3*FS))
    return pk

def _znorm(x):
    x = np.asarray(x, np.float64)
    return ((x - x.mean())/(x.std()+1e-6)).astype(np.float32)

def make_views(sig, seed, beat_locked):
    """Return (K, SIGLEN) beat-locked or random circular-shift views, z-normed."""
    rng = np.random.default_rng(seed)
    x = np.asarray(sig, np.float64)
    if beat_locked:
        pk = _rpeaks(x)
        if len(pk) >= 2:
            offs = pk[:K_VIEWS] if len(pk) >= K_VIEWS else np.resize(pk, K_VIEWS)
        else:
            offs = rng.integers(0, SIGLEN, size=K_VIEWS)   # fallback: random
    else:
        offs = rng.integers(0, SIGLEN, size=K_VIEWS)
    views = [np.roll(x, -int(o)) for o in offs]
    return np.stack([_znorm(v) for v in views]).astype(np.float32)

def _bn_affine_params(model):
    ps = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d) and m.affine:
            ps += [m.weight, m.bias]
    return ps

@torch.no_grad()
def _reset_to(model, state):
    model.load_state_dict(state)

def tta_predict(model, base_state, sigs, seed, beat_locked):
    """Per-clip TTA (BN affine only, reset per clip). Returns prob-of-class-1 per clip."""
    probs = []
    for i, sig in enumerate(sigs):
        _reset_to(model, base_state)
        # freeze everything except BN affine
        for p in model.parameters(): p.requires_grad_(False)
        bn = _bn_affine_params(model)
        for p in bn: p.requires_grad_(True)
        opt = torch.optim.SGD(bn, lr=TTA_LR, momentum=0.9)
        V = torch.from_numpy(make_views(sig, seed*100003+i, beat_locked))[:,None,:]  # (K,1,L)
        model.train()   # BN uses the K-view batch statistics (well-posed since K>1)
        for _ in range(TTA_STEPS):
            opt.zero_grad()
            logit = model(V)                       # (K,2)
            p = F.softmax(logit,1)
            ent = -(p * torch.log(p+1e-8)).sum(1).mean()          # confidence (TENT)
            pbar = p.mean(0, keepdim=True)
            consist = (p * (torch.log(p+1e-8) - torch.log(pbar+1e-8))).sum(1).mean()  # KL to mean
            (ent + consist).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pr = F.softmax(model(V),1)[:,1].mean().item()
        probs.append(pr)
    return np.array(probs)

def auroc(probs, ys):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(ys, probs))

def run_seed(seed, tr, tr_y, cinc):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]

    # base clinical model
    m=ECGResNet1d(n_leads=1,n_classes=2); e25.train(m,e25.SigDataset(tr,tr_y),epochs=20,tag=f"s{seed}-base")
    base_state=copy.deepcopy(m.state_dict())
    out={}
    out["clean"]=e25.evaluate(m, ts, ty)[0]

    # calibration reference (separate train-time model)
    tgt=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt,tr,fs=FS,siglen=SIGLEN,seed=seed,n_probe=40)
    aug=[clc.generate(x) for x in tr]
    mc=ECGResNet1d(n_leads=1,n_classes=2); e25.train(mc,e25.SigDataset(aug,tr_y),epochs=20,tag=f"s{seed}-cal")
    out["closed_aug"]=e25.evaluate(mc, ts, ty)[0]

    # TTA arms (adapt the base model per clip)
    out["shift_consist"]=auroc(tta_predict(m, base_state, ts, seed, beat_locked=False), ty)
    out["beat_consist"] =auroc(tta_predict(m, base_state, ts, seed, beat_locked=True ), ty)
    print(f"  [s{seed}] clean={out['clean']:.3f} cal={out['closed_aug']:.3f} shift={out['shift_consist']:.3f} beat={out['beat_consist']:.3f}", flush=True)
    return out

def main():
    print("Loading PTB-XL AF/NORM + CinC ...", flush=True)
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL={len(tr)}  CinC={len(cinc)}  (K={K_VIEWS} views, {TTA_STEPS} steps, lr={TTA_LR})", flush=True)

    seeds=list(range(10)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc)

    arms=["clean","closed_aug","shift_consist","beat_consist"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (10 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean":    pair(A["closed_aug"],A["clean"]),
      "shift_consist_vs_clean": pair(A["shift_consist"],A["clean"]),
      "beat_consist_vs_clean":  pair(A["beat_consist"],A["clean"]),
      "beat_vs_shift":          pair(A["beat_consist"],A["shift_consist"]),
      "beat_vs_calibration":    pair(A["beat_consist"],A["closed_aug"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:24s} Δ={dl:+.3f} wins {w}/10 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols=["#999","#d4a017","#8899cc","#4a7"]
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],color=cols,capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.006,f"{A[a].mean():.3f}",ha="center",fontsize=9)
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["closed_aug"].mean(),color="#d4a017",ls=":",lw=1)
    ax.set_ylim(0.5,0.85); ax.set_ylabel("AUROC real CinC (10 seeds)")
    ax.set_title(f"E61 cardiac-cycle per-clip TTA (label-free)\nbeat {A['beat_consist'].mean():.3f} · shift {A['shift_consist'].mean():.3f} · clean {A['clean'].mean():.3f} · calib {A['closed_aug'].mean():.3f}")
    fig.tight_layout(); fig.savefig(RESULTS/"cardiac_tta.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'cardiac_tta.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"k_views":K_VIEWS,"tta_steps":TTA_STEPS,"tta_lr":TTA_LR,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["CinC finger != AW wrist","AF/NORM easy","BN-affine-only per-clip TTA",
                              "K/steps/lr a priori","circular-shift wrap artifact","10 seeds"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
