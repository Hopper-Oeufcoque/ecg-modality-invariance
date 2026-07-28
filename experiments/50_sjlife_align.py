#!/usr/bin/env python3
"""
E50 — Real paired-hardware modality-invariant pretraining (SJLIFE).

E49 showed injecting CLINICAL information (12-lead teacher) backfires: it drags the
student toward the clinical distribution. The fix per that lesson: the injected
information must be WATCH-ANCHORED / grounded across BOTH modalities. Our only real
paired asset is SJLIFE (243 patients recorded on BOTH clinical 12-lead AND Apple
Watch). Both clinical-Lead-I and Apple are SINGLE lead -> one shared encoder.

Method: contrastively pretrain the shared single-lead encoder so the SAME patient's
clinical-Lead-I window and Apple-Watch window map to the SAME point (InfoNCE, same
patient = positive, other patients in batch = negatives). This learns a
modality-INVARIANT feature space from REAL hardware, no disease labels. Then attach
an AF/NORM head, train on PTB-XL clinical Lead-I, test on REAL CinC. If real-paired
invariance pretraining helps transfer (and/or stacks with calibration), that is the
north-star "invariant features" grounded in real hardware.

Arms (test real CinC AF/N, 20 seeds; pretrained encoder shared across seeds):
  clean          : from-scratch student on clean Lead-I (floor)
  closed_aug     : E42 winner — calibrated Lead-I
  sjlife_ft      : SJLIFE-pretrained encoder -> fine-tune on clean Lead-I
  sjlife_ft_aug  : SJLIFE-pretrained encoder -> fine-tune on calibrated Lead-I

HONEST FLAGS: SJLIFE has NO disease labels (pretrain is label-free, correct); CinC
finger != AW wrist != SJLIFE wrist (3 devices); SJLIFE n=243 small for contrastive;
InfoNCE temp=0.1 a priori; AF/NORM easy task; single clinical train set.
"""
from __future__ import annotations
import os, glob, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.signal import resample
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import ECGResNet1d
from src.aw_generator import ClosedLoopCalibrator, signal_modality_stats

_spec = importlib.util.spec_from_file_location("e25", ROOT / "experiments" / "25_aw_generator.py")
e25 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e25)

RESULTS = ROOT / "results" / "50_sjlife_align"; RESULTS.mkdir(parents=True, exist_ok=True)
SJ = ROOT / "data" / "sjlife"
FS = 100.0; SIGLEN = 1000
FS_APPLE, FS_CLIN = 512.0, 500.0
TEMP = 0.1

def make_model(): return ECGResNet1d(n_leads=1, n_classes=2)

def encode(model, x):
    """Pooled 32-d feature (stem->blocks->BN/ReLU/pool/flatten), before dropout+linear."""
    x = model.stem(x); x = model.blocks(x)
    h = list(model.head)
    x = h[0](x); x = h[1](x); x = h[2](x); x = h[3](x)   # BN, ReLU, AdaptiveAvgPool1d, Flatten
    return x  # (B, 32)

# ---------- SJLIFE paired loader ----------
def _norm(x):
    x = np.asarray(x, np.float64)
    if x.shape[0] < SIGLEN: x = np.concatenate([x, np.zeros(SIGLEN - x.shape[0])])
    return ((x[:SIGLEN]-x[:SIGLEN].mean())/(x[:SIGLEN].std()+1e-6)).astype(np.float32)

def _to_common(x, fs_in):
    return resample(np.asarray(x, float), int(round(len(x)*FS/fs_in)))

def _clin_leadI(path):
    c = np.squeeze(np.load(path).astype(np.float64))
    if c.ndim == 2 and c.shape[0] in (12,15): return c[0]
    if c.ndim == 2 and c.shape[1] in (12,15): return c[:,0]
    return c.ravel()

def load_sjlife_pairs():
    """Return list of (clinical_leadI_1000, [apple_windows_1000...]) per patient, z-normed at 100 Hz."""
    ap = sorted(glob.glob(str(SJ/"apple"/"apple_ecg_*.npy")),
                key=lambda p: int(p.split("_")[-1].split(".")[0]))
    pairs = []
    for a in ap:
        num = a.split("_")[-1].split(".")[0]
        cp = SJ/"clinical"/f"clinical_ecg_{num}.npy"
        if not cp.exists(): continue
        clin = _to_common(_clin_leadI(cp), FS_CLIN)          # ~1000 samp (10 s @100)
        appl = _to_common(np.load(a).astype(np.float64).ravel(), FS_APPLE)  # ~3000 (30 s)
        # window apple into non-overlapping 1000-sample slices to match clinical length
        aw = [appl[i:i+SIGLEN] for i in range(0, len(appl)-SIGLEN+1, SIGLEN)]
        aw = [w for w in aw if len(w)==SIGLEN]
        if len(clin) < SIGLEN or not aw: continue
        pairs.append((_norm(clin), [_norm(w) for w in aw]))
    return pairs

class PairContrastDS(Dataset):
    """Each item: one patient -> (clinical_view, random apple_view). Same index = positive."""
    def __init__(self, pairs, seed=0):
        self.pairs = pairs; self.rng = np.random.default_rng(seed)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        clin, aws = self.pairs[i]
        aw = aws[self.rng.integers(len(aws))]
        return (torch.from_numpy(clin[None]), torch.from_numpy(aw[None]))

def info_nce(zc, za, temp=TEMP):
    """Symmetric InfoNCE: same-index (same patient) positive, rest negatives."""
    zc = F.normalize(zc, dim=1); za = F.normalize(za, dim=1)
    logits = zc @ za.t() / temp                 # (B,B)
    labels = torch.arange(zc.size(0))
    return 0.5*(F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

def pretrain_sjlife(pairs, epochs=60, lr=1e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = make_model()
    proj = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,32))
    dl = DataLoader(PairContrastDS(pairs, seed=seed), batch_size=64, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(list(m.parameters())+list(proj.parameters()), lr=lr)
    for ep in range(epochs):
        m.train(); proj.train(); tot=0.0; nb=0
        for xc, xa in dl:
            opt.zero_grad()
            zc = proj(encode(m, xc)); za = proj(encode(m, xa))
            loss = info_nce(zc, za); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep+1)%10==0 or ep==0:
            print(f"  [sjlife-pretrain] ep {ep+1}/{epochs} infonce={tot/max(nb,1):.4f}", flush=True)
    return m  # pretrained encoder (proj discarded)

# ---------- supervised finetune from an init state ----------
def finetune(init_state, sigs, ys, epochs=20, lr=1e-3, tag="ft"):
    m = make_model()
    if init_state is not None: m.load_state_dict(init_state, strict=True)
    e25.train(m, e25.SigDataset(sigs, ys), epochs=epochs, tag=tag)
    return m

def run_seed(seed, tr, tr_y, cinc, pre_state):
    rng=np.random.default_rng(seed); torch.manual_seed(seed); np.random.seed(seed)
    idx=np.arange(len(cinc)); rng.shuffle(idx); cut=len(cinc)//2
    ref=[cinc[i] for i in idx[:cut]]; test=[cinc[i] for i in idx[cut:]]
    ts=[r["sig"] for r in test]; ty=[r["y"] for r in test]
    tgt_bw=float(np.mean([signal_modality_stats(r["sig"],FS)["bw_energy"] for r in ref[:200]]))
    clc=ClosedLoopCalibrator.fit(tgt_bw, tr, fs=FS, siglen=SIGLEN, seed=seed, n_probe=40)
    aug=[clc.generate(x) for x in tr]

    out={}
    out["clean"]        = e25.evaluate(finetune(None, tr, tr_y, tag=f"s{seed}-clean"), ts, ty)[0]
    out["closed_aug"]   = e25.evaluate(finetune(None, aug, tr_y, tag=f"s{seed}-aug"), ts, ty)[0]
    out["sjlife_ft"]    = e25.evaluate(finetune(pre_state, tr, tr_y, tag=f"s{seed}-sjft"), ts, ty)[0]
    out["sjlife_ft_aug"]= e25.evaluate(finetune(pre_state, aug, tr_y, tag=f"s{seed}-sjft_aug"), ts, ty)[0]
    return out

def main():
    print("Loading SJLIFE real paired hardware ...", flush=True)
    pairs = load_sjlife_pairs()
    n_aw = sum(len(a) for _,a in pairs)
    print(f"  SJLIFE patients={len(pairs)}  total apple windows={n_aw}", flush=True)

    print("\n== contrastive pretraining shared encoder on REAL pairs (once) ==", flush=True)
    pre = pretrain_sjlife(pairs, epochs=60, seed=0)
    pre_state = {k: v.clone() for k,v in pre.state_dict().items()}

    print("\nLoading PTB-XL AF/NORM + CinC ...", flush=True)
    d=e25.load_ptbxl_binary(Path.home()/"data"/"ptbxl", max_per_class=700)
    tr=[r["leadI"] for r in d["train"]]; tr_y=[r["y"] for r in d["train"]]
    cA,cN=e25.load_cinc(n_per_class=700); cinc=cA+cN
    print(f"  PTB-XL n={len(tr)}  CinC n={len(cinc)}", flush=True)

    seeds=list(range(20)); per={}
    for s in seeds:
        print(f"\n== seed {s} ==", flush=True)
        per[s]=run_seed(s,tr,tr_y,cinc,pre_state)

    arms=["clean","closed_aug","sjlife_ft","sjlife_ft_aug"]
    A={a:np.array([per[s][a] for s in seeds]) for a in arms}
    from scipy import stats as sst
    print("\n===== AUROC on real CinC (20 seeds) =====")
    for a in arms: print(f"  {a:14s} {A[a].mean():.3f} ± {A[a].std():.3f}")
    def pair(x,b):
        t,p=sst.ttest_rel(x,b); return float((x-b).mean()),int((x>b).sum()),float(p)
    cmp={
      "closed_aug_vs_clean":   pair(A["closed_aug"],A["clean"]),
      "sjlife_ft_vs_clean":    pair(A["sjlife_ft"],A["clean"]),
      "sjlife_ft_vs_aug":      pair(A["sjlife_ft"],A["closed_aug"]),
      "sjlife_ft_aug_vs_aug":  pair(A["sjlife_ft_aug"],A["closed_aug"]),
      "sjlife_ft_aug_vs_clean":pair(A["sjlife_ft_aug"],A["clean"]),
    }
    print()
    for k,(dl,w,p) in cmp.items(): print(f"  {k:26s} Δ={dl:+.3f} wins {w}/20 p={p:.4f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(arms,[A[a].mean() for a in arms],yerr=[A[a].std() for a in arms],
           color=["#999","#d4a017","#4a7","#3577c2"],capsize=4)
    for i,a in enumerate(arms): ax.text(i,A[a].mean()+0.008,f"{A[a].mean():.3f}",ha="center")
    ax.axhline(A["clean"].mean(),color="k",ls=":",lw=1); ax.axhline(A["closed_aug"].mean(),color="#d4a017",ls=":",lw=1)
    ax.set_ylim(0.5,0.9); ax.set_ylabel("AUROC real CinC (20 seeds)")
    ax.set_title(f"E50 SJLIFE real-paired invariance pretraining  (sjft−aug Δ={cmp['sjlife_ft_vs_aug'][0]:+.3f} p={cmp['sjlife_ft_vs_aug'][2]:.2f})")
    fig.tight_layout(); fig.savefig(RESULTS/"sjlife_align.png",dpi=110)
    print(f"\nSaved -> {RESULTS/'sjlife_align.png'}")

    with open(RESULTS/"metrics.json","w") as f:
        json.dump({"seeds":len(seeds),"temp":TEMP,"sjlife_patients":len(pairs),"apple_windows":n_aw,
                   "means":{a:float(A[a].mean()) for a in arms},"stds":{a:float(A[a].std()) for a in arms},
                   "comparisons":{k:{"delta":v[0],"wins":v[1],"p":v[2]} for k,v in cmp.items()},
                   "honesty":["SJLIFE no disease labels (label-free pretrain, correct)",
                              "3 devices: CinC finger != AW wrist != SJLIFE wrist",
                              "SJLIFE n=243 small for contrastive","InfoNCE temp=0.1 a priori",
                              "AF/NORM easy task","single clinical train set"]}, f, indent=2)
    print(f"Saved -> {RESULTS/'metrics.json'}\nDONE.")

if __name__=="__main__":
    main()
